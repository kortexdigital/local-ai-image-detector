#!/usr/bin/env node
/**
 * Measure balanced accuracy through the extension itself.
 *
 * The point of driving the built extension in a real Chrome rather than
 * scoring in Node is that it exercises the shipped preprocessing, the shipped
 * graphs and the shipped calibration. A detector can measure well offline and
 * poorly in the browser from a single mismatched resize; running the browser
 * path is the only way to see that.
 *
 *   node eval/run-benchmark.mjs --real <dir> --ai <dir> [--threshold 0.65] [--limit N]
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const IMAGE_PATTERN = /\.(jpe?g|png|webp|bmp|gif)$/i;

function parseArgs(argv) {
  const args = { threshold: 0.65, limit: Infinity, out: null };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--real') args.real = argv[++i];
    else if (key === '--ai') args.ai = argv[++i];
    else if (key === '--threshold') args.threshold = Number(argv[++i]);
    else if (key === '--limit') args.limit = Number(argv[++i]);
    else if (key === '--out') args.out = argv[++i];
  }
  if (!args.real || !args.ai) {
    throw new Error('usage: run-benchmark.mjs --real <dir> --ai <dir> [--threshold 0.65] [--limit N]');
  }
  return args;
}

async function listImages(dir, limit) {
  const out = [];
  async function walk(current) {
    for (const entry of await fs.readdir(current, { withFileTypes: true })) {
      if (out.length >= limit) return;
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (IMAGE_PATTERN.test(entry.name)) out.push(full);
    }
  }
  await walk(dir);
  return out;
}

async function launch() {
  const browser = await puppeteer.launch({
    headless: 'new',
    // The first score pays for loading an 89 MB model, which outlasts the
    // default 180 s protocol timeout on a cold WASM start.
    protocolTimeout: 600000,
    args: [
      `--disable-extensions-except=${DIST}`,
      `--load-extension=${DIST}`,
      '--no-sandbox',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
    ],
  });

  // The extension id is whatever Chrome assigned to the unpacked build.
  const target = await browser.waitForTarget(
    (t) => t.type() === 'service_worker' && t.url().startsWith('chrome-extension://'),
    { timeout: 30000 },
  );
  const extensionId = new URL(target.url()).host;
  return { browser, extensionId };
}

function summarize(records, threshold) {
  const byLabel = { real: [], ai: [] };
  for (const record of records) byLabel[record.label].push(record);

  const tnr = byLabel.real.length
    ? byLabel.real.filter((r) => r.confidence < threshold).length / byLabel.real.length
    : 0;
  const tpr = byLabel.ai.length
    ? byLabel.ai.filter((r) => r.confidence >= threshold).length / byLabel.ai.length
    : 0;

  const deadZone =
    records.filter((r) => r.confidence >= 0.35 && r.confidence < threshold).length /
    Math.max(1, records.length);

  return {
    balancedAccuracy: (tpr + tnr) / 2,
    tpr,
    tnr,
    counts: { real: byLabel.real.length, ai: byLabel.ai.length },
    confusion: {
      realCorrect: byLabel.real.filter((r) => r.confidence < threshold).length,
      realWrong: byLabel.real.filter((r) => r.confidence >= threshold).length,
      aiCorrect: byLabel.ai.filter((r) => r.confidence >= threshold).length,
      aiWrong: byLabel.ai.filter((r) => r.confidence < threshold).length,
    },
    deadZone,
  };
}

function curve(records) {
  const points = [];
  for (let t = 0.05; t <= 0.95; t += 0.05) {
    const s = summarize(records, t);
    points.push({ threshold: Number(t.toFixed(2)), balancedAccuracy: Number(s.balancedAccuracy.toFixed(4)) });
  }
  return points;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  await fs.access(path.join(DIST, 'manifest.json')).catch(() => {
    throw new Error('dist/ not found. Run `npm run build` first.');
  });

  const realFiles = await listImages(args.real, args.limit);
  const aiFiles = await listImages(args.ai, args.limit);
  if (realFiles.length === 0 || aiFiles.length === 0) {
    throw new Error('both --real and --ai must contain images');
  }
  process.stdout.write(`Scoring ${realFiles.length} real and ${aiFiles.length} AI images\n`);

  const { browser, extensionId } = await launch();
  const page = await browser.newPage();
  page.on('console', (message) => {
    if (message.type() === 'error') process.stderr.write(`  page error: ${message.text()}\n`);
  });

  await page.goto(`chrome-extension://${extensionId}/eval/runner.html`, {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForFunction(
    () => globalThis.__aiidLoaded === true || globalThis.__aiidLoadError,
    { timeout: 180000 },
  );
  const loadError = await page.evaluate(() => globalThis.__aiidLoadError);
  if (loadError) throw new Error(`inference host failed to load: ${loadError}`);

  const status = await page.evaluate(() => globalThis.__aiidStatus());
  process.stdout.write(`Backend: ${status.backend}\n`);

  const records = [];
  const started = Date.now();

  for (const [label, files] of [
    ['real', realFiles],
    ['ai', aiFiles],
  ]) {
    for (const [index, file] of files.entries()) {
      const base64 = (await fs.readFile(file)).toString('base64');
      try {
        const result = await page.evaluate(
          (b64, threshold) => globalThis.__aiidScore(b64, { threshold }),
          base64,
          args.threshold,
        );
        records.push({ label, file, ...result });
      } catch (error) {
        process.stderr.write(`  skipped ${path.basename(file)}: ${error.message}\n`);
      }
      if ((index + 1) % 100 === 0) {
        const rate = records.length / ((Date.now() - started) / 1000);
        process.stdout.write(`  ${label}: ${index + 1}/${files.length} (${rate.toFixed(1)} img/s)\n`);
      }
    }
  }

  await browser.close();

  const summary = summarize(records, args.threshold);
  const elapsed = (Date.now() - started) / 1000;

  const report = {
    backend: status.backend,
    threshold: args.threshold,
    scored: records.length,
    ...summary,
    curve: curve(records),
    imagesPerSecond: Number((records.length / elapsed).toFixed(2)),
  };

  process.stdout.write('\n');
  process.stdout.write(`Balanced accuracy at ${args.threshold}: ${summary.balancedAccuracy.toFixed(4)}\n`);
  process.stdout.write(`  true positive rate (AI detected): ${summary.tpr.toFixed(4)}\n`);
  process.stdout.write(`  true negative rate (real kept):   ${summary.tnr.toFixed(4)}\n`);
  process.stdout.write(`  scored ${records.length} images at ${report.imagesPerSecond} img/s on ${status.backend}\n`);

  if (args.out) {
    await fs.writeFile(args.out, `${JSON.stringify({ ...report, records: records.map(({ file, label, confidence }) => ({ file, label, confidence })) }, null, 2)}\n`);
    process.stdout.write(`  wrote ${args.out}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`\n${error.stack ?? error.message}\n`);
  process.exit(1);
});
