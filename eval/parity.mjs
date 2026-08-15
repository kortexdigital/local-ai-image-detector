#!/usr/bin/env node
/**
 * Prove the browser and the training pipeline agree where they must.
 *
 * This is the check for the failure mode this project was designed around: a
 * model that measures well in Python and badly in Chrome because one side
 * resized, cropped, normalized or ordered channels differently. A single
 * mismatched interpolation is invisible in every other test and can cost
 * twenty points of accuracy.
 *
 * Two stages, held to different standards, because only one of them can be
 * exact:
 *
 *   1. Decode, crop and preprocess. STRICT. Both sides run the same ONNX
 *      graph over the same pixels, so any drift is a defect.
 *
 *   2. The backbone embedding. LOOSE. The shipped backbone is int8, and
 *      quantized kernels requantize differently between onnxruntime's native
 *      CPU build and its WebAssembly build. That difference is inherent to
 *      shipping a quantized model, not a bug, and it was measured to cost
 *      nothing: browser balanced accuracy came out slightly above the Python
 *      figure on the same images. It is still worth watching, because a large
 *      jump would mean something else broke.
 *
 *   node eval/parity.mjs --images <dir> [--limit 25]
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

/** Preprocessing runs the same graph on both sides, so it must be exact. */
const PREPROCESS_TOLERANCE = 1e-4;
/** Quantized kernels differ across execution providers; this bounds by how much. */
const EMBEDDING_MIN_COSINE = 0.95;

function parseArgs(argv) {
  const args = { limit: 25 };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--images') args.images = argv[++i];
    else if (argv[i] === '--limit') args.limit = Number(argv[++i]);
  }
  if (!args.images) throw new Error('usage: parity.mjs --images <dir> [--limit 25]');
  return args;
}

const PYTHON_SCRIPT = `
import json, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

from training.config import CONFIG
from training.features.backbone import FeatureExtractor, backbone_by_key
from training.features.preprocess_graph import center_square_crop
from training.head.train import l2_normalize

backbone = backbone_by_key("clip-vit-b32-int8")
pre_path = CONFIG.models_dir / "backbones" / backbone.key / "preprocess.onnx"
pre = ort.InferenceSession(str(pre_path), providers=["CPUExecutionProvider"])
extractor = FeatureExtractor(backbone, CONFIG.models_dir)

out = {}
for file in json.load(sys.stdin):
    image = Image.open(file)
    image.load()
    image = image.convert("RGB")
    cropped = center_square_crop(image)
    pixels = np.asarray(cropped, dtype=np.uint8)[None, ...]
    values = pre.run(None, {"pixels": pixels})[0].reshape(-1)
    embedding = l2_normalize(extractor.embed(image).reshape(1, -1))[0]
    out[file] = {
        "side": int(cropped.size[0]),
        "pixelSum": int(pixels.sum()),
        "values": values[:4096].tolist(),
        "embedding": embedding.tolist(),
    }
json.dump(out, sys.stdout)
`;

function pythonSide(files) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, ['-c', PYTHON_SCRIPT], {
      cwd: ROOT,
      env: { ...process.env, PYTHONPATH: ROOT },
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => (stdout += chunk));
    child.stderr.on('data', (chunk) => (stderr += chunk));
    child.on('close', (code) => {
      if (code !== 0) reject(new Error(`python pipeline failed:\n${stderr}`));
      else resolve(JSON.parse(stdout));
    });
    child.stdin.write(JSON.stringify(files));
    child.stdin.end();
  });
}

async function browserSide(files) {
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
  const loadError = await page.evaluate(() => globalThis.__aiidLoadError);
  if (loadError) throw new Error(`inference host failed to load: ${loadError}`);

  const out = {};
  for (const file of files) {
    const base64 = (await fs.readFile(file)).toString('base64');
    const pre = await page.evaluate((b64) => globalThis.__aiidPreprocess(b64), base64);
    const scored = await page.evaluate(
      (b64) => globalThis.__aiidScore(b64, { includeEmbedding: true }),
      base64,
    );
    out[file] = { ...pre, embedding: scored.embedding };
  }
  const backend = await page.evaluate(() => globalThis.__aiidStatus());
  await browser.close();
  return { results: out, backend: backend.backend };
}

function compare(a, b) {
  let maxDiff = 0;
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i += 1) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  return maxDiff;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i += 1) dot += a[i] * b[i];
  return dot;
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

  const [fromPython, browser] = await Promise.all([pythonSide(files), browserSide(files)]);
  const fromBrowser = browser.results;

  let worstPixel = 0;
  let worstPre = 0;
  let worstPreFile = '';
  let worstCosine = 1;
  let worstCosineFile = '';

  for (const file of files) {
    const py = fromPython[file];
    const br = fromBrowser[file];
    if (!py || !br) throw new Error(`missing result for ${file}`);

    if (py.side !== br.side) {
      throw new Error(`crop size differs for ${path.basename(file)}: ${py.side} vs ${br.side}`);
    }
    worstPixel = Math.max(worstPixel, Math.abs(py.pixelSum - br.pixelSum));

    const preDiff = compare(py.values, br.values.slice(0, py.values.length));
    if (preDiff > worstPre) {
      worstPre = preDiff;
      worstPreFile = path.basename(file);
    }

    const similarity = cosine(py.embedding, br.embedding);
    if (similarity < worstCosine) {
      worstCosine = similarity;
      worstCosineFile = path.basename(file);
    }
  }

  process.stdout.write(`  backend: ${browser.backend}\n`);
  process.stdout.write(`  decode + crop, worst pixel-sum difference: ${worstPixel}\n`);
  process.stdout.write(
    `  preprocess, worst absolute difference: ${worstPre.toExponential(3)} (${worstPreFile})\n`,
  );
  process.stdout.write(
    `  embedding, lowest cosine similarity:   ${worstCosine.toFixed(6)} (${worstCosineFile})\n`,
  );

  const failures = [];
  if (worstPixel !== 0) {
    failures.push(
      `decode or crop drifted: pixel sums differ by ${worstPixel}. ` +
        'Check colorSpaceConversion, premultiplyAlpha and the crop arithmetic.',
    );
  }
  if (worstPre > PREPROCESS_TOLERANCE) {
    failures.push(
      `preprocessing drifted: ${worstPre.toExponential(3)} exceeds ${PREPROCESS_TOLERANCE}. ` +
        'Both sides run the same graph, so this is a real defect.',
    );
  }
  if (worstCosine < EMBEDDING_MIN_COSINE) {
    failures.push(
      `embeddings drifted further than quantization explains: cosine ${worstCosine.toFixed(4)} ` +
        `below ${EMBEDDING_MIN_COSINE}. Expect roughly 0.98 for the int8 backbone.`,
    );
  }

  if (failures.length > 0) {
    process.stderr.write(`\nPARITY FAILED\n${failures.map((f) => `  - ${f}`).join('\n')}\n`);
    process.exit(1);
  }
  process.stdout.write('PARITY OK\n');
}

main().catch((error) => {
  process.stderr.write(`\n${error.stack ?? error.message}\n`);
  process.exit(1);
});
