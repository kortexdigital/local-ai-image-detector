#!/usr/bin/env node
/**
 * Assemble dist/, the directory Chrome loads.
 *
 * Everything the extension needs at runtime is placed here at build time. The
 * extension itself never fetches a model, a weight, or any other inference
 * asset: after installation it can run with the network switched off.
 *
 * The backbone is downloaded once here and verified against a pinned SHA-256,
 * so a build either reproduces exactly the model that was measured or fails.
 */

import { createHash } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const HEAD_DIR = path.join(ROOT, 'models', 'clip-vit-b32-int8');
const PREPROCESS = path.join(
  ROOT,
  'models',
  'backbones',
  'clip-vit-b32-int8',
  'preprocess.onnx',
);
const CACHE = path.join(ROOT, '.build-cache');

const assets = JSON.parse(
  await fs.readFile(path.join(ROOT, 'scripts', 'assets.json'), 'utf8'),
);

function log(message) {
  process.stdout.write(`${message}\n`);
}

async function sha256(file) {
  const hash = createHash('sha256');
  const handle = await fs.open(file, 'r');
  try {
    const stream = handle.createReadStream();
    for await (const chunk of stream) hash.update(chunk);
  } finally {
    await handle.close();
  }
  return hash.digest('hex');
}

async function exists(target) {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

async function download(url, target) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(`download failed for ${url} (${response.status})`);
  }
  await pipeline(response.body, createWriteStream(target));
}

async function fetchPinned({ url, sha256: expected, bytes }) {
  const cached = path.join(CACHE, expected);
  if (!(await exists(cached))) {
    log(`  downloading ${url}`);
    await download(url, cached);
  }

  const actual = await sha256(cached);
  if (actual !== expected) {
    await fs.rm(cached, { force: true });
    throw new Error(
      `checksum mismatch for ${url}\n  expected ${expected}\n  actual   ${actual}\n` +
        'The upstream artifact changed. Verify the new file before updating scripts/assets.json.',
    );
  }

  if (bytes != null) {
    const stat = await fs.stat(cached);
    if (stat.size !== bytes) {
      throw new Error(`size mismatch for ${url}: expected ${bytes}, got ${stat.size}`);
    }
  }

  return cached;
}

async function copyDir(from, to) {
  await fs.cp(from, to, { recursive: true });
}

async function requireFile(target, hint) {
  if (!(await exists(target))) {
    throw new Error(`missing ${path.relative(ROOT, target)}\n  ${hint}`);
  }
}

async function main() {
  log('Building dist/');

  await fs.rm(DIST, { recursive: true, force: true });
  await fs.mkdir(DIST, { recursive: true });

  // 1. Extension sources.
  log('  copying extension sources');
  await copyDir(path.join(ROOT, 'extension'), DIST);

  // 2. Trained artifacts, produced by the offline pipeline and versioned.
  log('  copying trained head and calibration');
  const modelsOut = path.join(DIST, 'models');
  await fs.mkdir(modelsOut, { recursive: true });

  await requireFile(
    path.join(HEAD_DIR, 'head.onnx'),
    'Run: .venv/bin/python -m training.cli train clip-vit-b32-int8',
  );
  await requireFile(
    path.join(HEAD_DIR, 'calibration.json'),
    'Run: .venv/bin/python -m training.cli train clip-vit-b32-int8',
  );
  await requireFile(
    PREPROCESS,
    'Run: .venv/bin/python -m training.cli extract clip-vit-b32-int8',
  );

  await fs.copyFile(path.join(HEAD_DIR, 'head.onnx'), path.join(modelsOut, 'head.onnx'));
  await fs.copyFile(
    path.join(HEAD_DIR, 'calibration.json'),
    path.join(modelsOut, 'calibration.json'),
  );
  await fs.copyFile(PREPROCESS, path.join(modelsOut, 'preprocess.onnx'));

  // 3. Frozen backbone, pinned by hash.
  log('  fetching backbone');
  const backbone = await fetchPinned(assets.backbone);
  await fs.copyFile(backbone, path.join(DIST, assets.backbone.dest));

  // 4. ONNX Runtime Web, copied from the locked devDependency.
  log('  vendoring ONNX Runtime Web');
  const ortSource = path.join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
  await requireFile(ortSource, 'Run: npm ci');
  const ortOut = path.join(DIST, 'vendor', 'ort');
  await fs.mkdir(ortOut, { recursive: true });
  for (const file of assets.ortFiles) {
    await requireFile(path.join(ortSource, file), 'Run: npm ci');
    await fs.copyFile(path.join(ortSource, file), path.join(ortOut, file));
  }

  // 5. License travels with the package.
  await fs.copyFile(path.join(ROOT, 'LICENSE'), path.join(DIST, 'LICENSE'));

  const total = await directorySize(DIST);
  log(`Done. dist/ is ${(total / 1e6).toFixed(1)} MB`);
  log('Load it with chrome://extensions -> Developer mode -> Load unpacked -> dist/');
}

async function directorySize(dir) {
  let total = 0;
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    total += entry.isDirectory() ? await directorySize(full) : (await fs.stat(full)).size;
  }
  return total;
}

main().catch((error) => {
  process.stderr.write(`\nBuild failed: ${error.message}\n`);
  process.exit(1);
});
