#!/usr/bin/env node
/**
 * Drive a real image search with the extension loaded and capture what a user
 * would see.
 *
 * This is a demonstration, not a measurement: the labels on the page are
 * unknown, so nothing here says anything about accuracy. It exists to show
 * the extension working on images it has never met, served by a site nobody
 * involved controls.
 *
 *   node eval/demo-screenshot.mjs "midjourney portrait" --out shot.png
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');

function parseArgs(argv) {
  const args = {
    query: null,
    out: path.join(ROOT, 'demo.png'),
    wait: 90000,
    headless: false,
    engine: 'bing',
  };
  const rest = [];
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--out') args.out = argv[++i];
    else if (argv[i] === '--wait') args.wait = Number(argv[++i]);
    else if (argv[i] === '--headless') args.headless = 'new';
    else if (argv[i] === '--engine') args.engine = argv[++i];
    else rest.push(argv[i]);
  }
  args.query = rest.join(' ') || 'ai generated art';
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await fs.access(path.join(DIST, 'manifest.json')).catch(() => {
    throw new Error('dist/ not found. Run `npm run build` first.');
  });

  const browser = await puppeteer.launch({
    // A visible window with automation flags stripped. Search engines refuse
    // headless traffic outright, and this demo has to run against the real
    // site rather than a stand-in.
    headless: args.headless,
    protocolTimeout: 600000,
    defaultViewport: { width: 1440, height: 1400 },
    args: [
      `--disable-extensions-except=${DIST}`,
      `--load-extension=${DIST}`,
      '--no-sandbox',
      '--window-size=1440,1400',
      '--lang=en-US',
      '--disable-blink-features=AutomationControlled',
      '--exclude-switches=enable-automation',
    ],
    ignoreDefaultArgs: ['--enable-automation'],
  });

  try {
    await browser.waitForTarget(
      (t) => t.type() === 'service_worker' && t.url().startsWith('chrome-extension://'),
      { timeout: 30000 },
    );

    const page = await browser.newPage();
    await page.evaluateOnNewDocument(() => {
      Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    });
    await page.setUserAgent(
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
        '(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36',
    );

    // Google refuses automated traffic outright and answers with a CAPTCHA;
    // its anti-bot measures are not something to work around. Bing and
    // DuckDuckGo serve the same kind of page and allow it.
    const engines = {
      google: (q) => `https://www.google.com/search?tbm=isch&hl=en&q=${encodeURIComponent(q)}`,
      bing: (q) => `https://www.bing.com/images/search?q=${encodeURIComponent(q)}&form=HDRSC2`,
      duckduckgo: (q) => `https://duckduckgo.com/?q=${encodeURIComponent(q)}&iax=images&ia=images`,
    };
    const build = engines[args.engine] ?? engines.bing;
    const url = build(args.query);
    process.stdout.write(`Opening ${args.engine} images for "${args.query}"\n`);
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 90000 });

    // Consent interstitials appear in some regions and cover the results.
    for (const label of ['Accept all', 'I agree', 'Aceitar tudo', 'Reject all']) {
      const button = await page
        .$$(`button ::-p-text(${label})`)
        .then((found) => found[0])
        .catch(() => null);
      if (button) {
        await button.click().catch(() => {});
        await page.waitForNavigation({ timeout: 15000 }).catch(() => {});
        break;
      }
    }

    // Analysis is lazy by design; scrolling is what brings images into range.
    await page.evaluate(async () => {
      for (let y = 0; y < 2600; y += 350) {
        window.scrollTo(0, y);
        await new Promise((resolve) => setTimeout(resolve, 400));
      }
      window.scrollTo(0, 0);
    });

    // Wait for the queue to drain rather than for some arbitrary number of
    // badges. Stopping at the first few produces a screenshot of work in
    // progress, which reads as the extension having skipped everything below
    // the fold when it had simply not got there yet.
    process.stdout.write('Waiting for scoring to settle\n');
    const deadline = Date.now() + args.wait;
    const countBadges = () =>
      page.evaluate(() => ({
        scored: document.querySelectorAll('.aiid-ai, .aiid-real, .aiid-uncertain').length,
        pending: document.querySelectorAll('.aiid-pending').length,
      }));

    let count = 0;
    let stable = 0;
    while (Date.now() < deadline) {
      const { scored, pending } = await countBadges();
      if (scored === count && pending === 0 && scored > 0) {
        stable += 1;
        if (stable >= 3) break;
      } else {
        stable = 0;
      }
      count = scored;
      process.stdout.write(`  ${scored} scored, ${pending} in flight\r`);
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
    process.stdout.write('\n');

    const badges = await page.evaluate(() =>
      [...document.querySelectorAll('.aiid-badge')].map((b) => b.textContent),
    );
    process.stdout.write(`  ${count} images scored\n`);
    for (const badge of badges.slice(0, 12)) process.stdout.write(`    ${badge}\n`);

    await page.screenshot({ path: args.out, fullPage: false });
    process.stdout.write(`\nSaved ${args.out}\n`);

    if (count === 0) {
      process.stderr.write('No badges appeared. The page may have blocked automation.\n');
      process.exitCode = 1;
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`\n${error.stack ?? error.message}\n`);
  process.exit(1);
});
