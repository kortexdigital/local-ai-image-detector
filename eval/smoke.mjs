#!/usr/bin/env node
/**
 * End-to-end check that the extension actually works while browsing.
 *
 * The benchmark harness exercises the inference path directly. This exercises
 * everything around it: the content script finding images, the service worker
 * queueing them, the offscreen document scoring them, and badges appearing on
 * the page with real numbers.
 *
 * A throwaway HTTP server serves the fixture page, because content scripts do
 * not run on data: or chrome-extension: URLs. It is test scaffolding: the
 * extension has no localhost dependency, and the server is gone before this
 * script exits.
 *
 *   node eval/smoke.mjs
 */

import { createServer } from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const FIXTURES = path.join(ROOT, 'eval', 'fixtures');
const BENCH = path.join(ROOT, 'data', 'benchmark');

/**
 * Prefer the exported benchmark when it exists, otherwise fall back to the
 * checked-in fixtures. These checks prove the pipeline runs, not how accurate
 * it is, so any decodable image will do; the fixtures are generated rather
 * than copied so the repository redistributes no dataset image.
 */
async function imageDir(preferred) {
  try {
    const entries = await fs.readdir(preferred);
    if (entries.some((f) => /\.(jpe?g|png)$/i.test(f))) return preferred;
  } catch {
    // fall through
  }
  return FIXTURES;
}

async function pickImages(dir, count) {
  const entries = await fs.readdir(dir);
  const chosen = entries.filter((f) => /\.jpe?g$/i.test(f)).slice(0, count);
  return Promise.all(
    chosen.map(async (name) => ({ name, bytes: await fs.readFile(path.join(dir, name)) })),
  );
}

function buildPage(images) {
  const tags = images
    .map(
      (img, index) =>
        `<figure><img id="img${index}" src="/img/${encodeURIComponent(img.name)}" width="360" alt="" /><figcaption>${img.name.split('__')[0]}</figcaption></figure>`,
    )
    .join('\n');
  return `<!doctype html><html><head><meta charset="utf-8"><title>smoke</title>
<style>body{font:14px system-ui;margin:24px}figure{margin:0 0 24px}img{display:block}</style>
</head><body><h1>Extension smoke test</h1>${tags}</body></html>`;
}

async function main() {
  await fs.access(path.join(DIST, 'manifest.json')).catch(() => {
    throw new Error('dist/ not found. Run `npm run build` first.');
  });

  const realDir = await imageDir(path.join(BENCH, 'real'));
  const aiDir = await imageDir(path.join(BENCH, 'ai'));
  const images =
    realDir === aiDir
      ? await pickImages(realDir, 4)
      : [...(await pickImages(realDir, 3)), ...(await pickImages(aiDir, 3))];
  if (images.length === 0) throw new Error('no images to test with');
  const byName = new Map(images.map((i) => [i.name, i.bytes]));
  const page = buildPage(images);

  const server = createServer((request, response) => {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') {
      response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      response.end(page);
      return;
    }
    const name = decodeURIComponent(url.pathname.replace('/img/', ''));
    const bytes = byName.get(name);
    if (!bytes) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { 'content-type': 'image/jpeg' });
    response.end(bytes);
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;

  const browser = await puppeteer.launch({
    headless: 'new',
    // The first score pays for loading an 89 MB model, which outlasts the
    // default 180 s protocol timeout on a cold WASM start.
    protocolTimeout: 600000,
    args: [
      `--disable-extensions-except=${DIST}`,
      `--load-extension=${DIST}`,
      '--no-sandbox',
    ],
  });

  try {
    await browser.waitForTarget(
      (t) => t.type() === 'service_worker' && t.url().startsWith('chrome-extension://'),
      { timeout: 30000 },
    );

    const tab = await browser.newPage();
    const errors = [];
    tab.on('pageerror', (error) => errors.push(String(error)));

    await tab.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle2' });

    // Analysis is deliberately lazy: an image is only scored once it comes
    // near the viewport. Scrolling the page is what a user does, and without
    // it only the first image or two would ever be analyzed.
    await tab.evaluate(async () => {
      for (let y = 0; y < document.body.scrollHeight; y += 400) {
        window.scrollTo(0, y);
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
      window.scrollTo(0, 0);
    });

    process.stdout.write(`Waiting for badges on ${images.length} images\n`);
    await tab.waitForFunction(
      (expected) =>
        document.querySelectorAll('.aiid-badge.aiid-ai, .aiid-badge.aiid-real').length >= expected,
      { timeout: 240000 },
      images.length,
    );

    const badges = await tab.evaluate(() =>
      [...document.querySelectorAll('.aiid-badge')].map((b) => ({
        text: b.textContent,
        title: b.title,
        classes: [...b.classList],
      })),
    );

    process.stdout.write(`\nBadges rendered: ${badges.length}\n`);
    for (const badge of badges) {
      process.stdout.write(`  ${badge.text.padEnd(12)} ${badge.title.split('\n')[0]}\n`);
    }

    const failures = [];
    if (badges.length < images.length) {
      failures.push(`expected at least ${images.length} badges, saw ${badges.length}`);
    }
    if (!badges.every((b) => /\d+%/.test(b.text))) {
      failures.push('every badge must show a percentage');
    }
    if (!badges.every((b) => b.classes.includes('aiid-ai') || b.classes.includes('aiid-real'))) {
      failures.push('every badge must reach a verdict');
    }
    if (errors.length > 0) failures.push(`page errors: ${errors.join('; ')}`);

    // The page must remain usable: a detector that blocks clicks is worse
    // than no detector.
    const overlayBlocks = await tab.evaluate(() => {
      const image = document.querySelector('img');
      const rect = image.getBoundingClientRect();
      const atCentre = document.elementFromPoint(
        rect.left + rect.width / 2,
        rect.top + rect.height / 2,
      );
      return atCentre !== image;
    });
    if (overlayBlocks) failures.push('badge overlay intercepts clicks over the image');

    if (failures.length > 0) {
      process.stderr.write(`\nSMOKE FAILED\n${failures.map((f) => `  - ${f}`).join('\n')}\n`);
      process.exitCode = 1;
      return;
    }
    process.stdout.write('\nSMOKE OK\n');
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((error) => {
  process.stderr.write(`\n${error.stack ?? error.message}\n`);
  process.exit(1);
});
