#!/usr/bin/env node
/**
 * Prove the browser and the training pipeline compute the same features.
 *
 * This is the check that catches the failure mode this project was designed
 * around: a model that measures well in Python and badly in Chrome because
 * one side resized, normalized or ordered channels differently. A single
 * mismatched interpolation is invisible in every other test and costs twenty
 * points of accuracy.
 *
 *   node eval/parity.mjs --images <dir> [--tolerance 1e-3]
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import puppeteer from 'puppeteer';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const PYTHON = path.join(ROOT, '.venv', 'bin', 'python');
const IMAGE_PATTERN = /\.(jpe?g|png|webp)$/i;

function parseArgs(argv) {
  const args = { tolerance: 1e-3, limit: 25 };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--images') args.images = argv[++i];
    else if (argv[i] === '--tolerance') args.tolerance = Number(argv[++i]);
    else if (argv[i] === '--limit') args.limit = Number(argv[++i]);
  }
  if (!args.images) throw new Error('usage: parity.mjs --images <dir> [--tolerance 1e-3]');
  return args;
}

function pythonEmbeddings(files) {
  return new Promise((resolve, reject) => {
    const script = `
import json, sys
from pathlib import Path
from PIL import Image
import numpy as np
from training.config import CONFIG
from training.features.backbone import FeatureExtractor, backbone_by_key
from training.head.train import l2_normalize

extractor = FeatureExtractor(backbone_by_key("clip-vit-b32-int8"), CONFIG.models_dir)
out = {}
for raw in json.load(sys.stdin):
    vector = extractor.embed(Image.open(raw))
    out[raw] = l2_normalize(vector.reshape(1, -1))[0].tolist()
json.dump(out, sys.stdout)
`;
    const child = spawn(PYTHON, ['-c', script], {
      cwd: ROOT,
      env: { ...process.env, PYTHONPATH: ROOT },
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => (stdout += chunk));
    child.stderr.on('data', (chunk) => (stderr += chunk));
    child.on('close', (code) => {
      if (code !== 0) reject(new Error(`python extractor failed:\n${stderr}`));
      else resolve(JSON.parse(stdout));
    });
    child.stdin.write(JSON.stringify(files));
    child.stdin.end();
  });
}

async function browserEmbeddings(files) {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      `--disable-extensions-except=${DIST}`,
      `--load-extension=${DIST}`,
      '--no-sandbox',
    ],
  });
  const target = await browser.waitForTarget(
    (t) => t.type() === 'service_worker' && t.url().startsWith('chrome-extension://'),
    { timeout: 30000 },
  );
  const extensionId = new URL(target.url()).host;

  const page = await browser.newPage();
  await page.goto(`chrome-extension://${extensionId}/eval/runner.html`, {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForFunction(() => globalThis.__aiidLoaded === true || globalThis.__aiidLoadError, {
    timeout: 180000,
  });

  const out = {};
  for (const file of files) {
    const base64 = (await fs.readFile(file)).toString('base64');
    const result = await page.evaluate(
      (b64) => globalThis.__aiidScore(b64, { includeEmbedding: true }),
      base64,
    );
    out[file] = result.embedding;
  }
  await browser.close();
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const entries = await fs.readdir(args.images, { withFileTypes: true });
  const files = entries
    .filter((e) => e.isFile() && IMAGE_PATTERN.test(e.name))
    .slice(0, args.limit)
    .map((e) => path.join(args.images, e.name));

  if (files.length === 0) throw new Error(`no images found in ${args.images}`);
  process.stdout.write(`Comparing ${files.length} images\n`);

  const [fromPython, fromBrowser] = await Promise.all([
    pythonEmbeddings(files),
    browserEmbeddings(files),
  ]);

  let worst = 0;
  let worstFile = '';
  let worstCosine = 1;

  for (const file of files) {
    const a = fromPython[file];
    const b = fromBrowser[file];
    if (!a || !b) throw new Error(`missing embedding for ${file}`);
    if (a.length !== b.length) {
      throw new Error(`dimension mismatch for ${file}: python ${a.length}, browser ${b.length}`);
    }

    let maxDiff = 0;
    let dot = 0;
    for (let i = 0; i < a.length; i += 1) {
      maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
      dot += a[i] * b[i];
    }
    if (maxDiff > worst) {
      worst = maxDiff;
      worstFile = path.basename(file);
    }
    worstCosine = Math.min(worstCosine, dot);
  }

  process.stdout.write(`  worst absolute difference: ${worst.toExponential(3)} (${worstFile})\n`);
  process.stdout.write(`  lowest cosine similarity:  ${worstCosine.toFixed(6)}\n`);

  if (worst > args.tolerance) {
    process.stderr.write(
      `\nPARITY FAILED: ${worst.toExponential(3)} exceeds tolerance ${args.tolerance}.\n` +
        'Training and browser features have drifted. Check the preprocessing graph,\n' +
        'the crop, and the channel order before trusting any accuracy number.\n',
    );
    process.exit(1);
  }
  process.stdout.write('PARITY OK\n');
}

main().catch((error) => {
  process.stderr.write(`\n${error.stack ?? error.message}\n`);
  process.exit(1);
});
