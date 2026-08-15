#!/usr/bin/env node
/**
 * Prove the extension needs no network.
 *
 * The browser is put offline before the extension ever loads a model, and
 * every request that is not a chrome-extension: URL is aborted outright. If
 * scoring still works under those conditions, nothing about inference depends
 * on reaching anything.
 *
 * This is the behavioural counterpart to the static check in
 * eval/tests/build.test.mjs, which greps shipped code for remote fetches. A
 * grep can be fooled by a constructed URL; this cannot.
 *
 *   node eval/offline-check.mjs
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const BENCH = path.join(ROOT, 'data', 'benchmark');

async function firstImage(dir) {
  const entries = await fs.readdir(dir);
  const name = entries.find((f) => /\.(jpe?g|png)$/i.test(f));
  if (!name) throw new Error(`no image in ${dir}`);
  return path.join(dir, name);
}

async function main() {
  await fs.access(path.join(DIST, 'manifest.json')).catch(() => {
    throw new Error('dist/ not found. Run `npm run build` first.');
  });

  const browser = await puppeteer.launch({
    headless: 'new',
    protocolTimeout: 600000,
    args: [
      `--disable-extensions-except=${DIST}`,
      `--load-extension=${DIST}`,
      '--no-sandbox',
    ],
  });

  try {
    const target = await browser.waitForTarget(
      (t) => t.type() === 'service_worker' && t.url().startsWith('chrome-extension://'),
      { timeout: 30000 },
    );
    const extensionId = new URL(target.url()).host;

    const page = await browser.newPage();

    // Offline before anything loads, and every non-extension request refused,
    // so a cached response cannot quietly stand in for a live one.
    const blocked = [];
    await page.setRequestInterception(true);
    page.on('request', (request) => {
      const url = request.url();
      if (url.startsWith('chrome-extension://') || url.startsWith('data:')) {
        request.continue().catch(() => {});
      } else {
        blocked.push(url);
        request.abort().catch(() => {});
      }
    });
    await page.setOfflineMode(true);

    process.stdout.write('Browser offline, non-extension requests blocked\n');

    await page.goto(`chrome-extension://${extensionId}/eval/runner.html`, {
      waitUntil: 'domcontentloaded',
    });
    await page.waitForFunction(
      () => globalThis.__aiidLoaded === true || globalThis.__aiidLoadError,
      { timeout: 300000 },
    );

    const loadError = await page.evaluate(() => globalThis.__aiidLoadError);
    if (loadError) {
      throw new Error(`models failed to load offline: ${loadError}`);
    }

    const status = await page.evaluate(() => globalThis.__aiidStatus());
    process.stdout.write(`  models loaded offline, backend ${status.backend}\n`);

    const results = [];
    for (const [label, dir] of [
      ['real', path.join(BENCH, 'real')],
      ['ai', path.join(BENCH, 'ai')],
    ]) {
      const file = await firstImage(dir);
      const base64 = (await fs.readFile(file)).toString('base64');
      const scored = await page.evaluate((b64) => globalThis.__aiidScore(b64), base64);
      results.push({ label, confidence: scored.confidence, verdict: scored.verdict });
      process.stdout.write(
        `  scored a ${label} image offline: ${(scored.confidence * 100).toFixed(1)}% -> ${scored.verdict}\n`,
      );
    }

    if (results.length !== 2 || !results.every((r) => Number.isFinite(r.confidence))) {
      throw new Error('offline scoring did not produce usable confidences');
    }

    if (blocked.length > 0) {
      process.stdout.write(
        `\n  ${blocked.length} non-extension request(s) were blocked and inference still worked:\n`,
      );
      for (const url of [...new Set(blocked)].slice(0, 5)) {
        process.stdout.write(`    ${url.slice(0, 100)}\n`);
      }
    } else {
      process.stdout.write('\n  the extension made no non-extension request at all\n');
    }

    process.stdout.write('\nOFFLINE OK\n');
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`\nOFFLINE CHECK FAILED\n${error.stack ?? error.message}\n`);
  process.exit(1);
});
