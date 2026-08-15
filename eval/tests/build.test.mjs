/**
 * What ships must match what was measured, and must not reach the network.
 *
 * These run against dist/, so they only mean anything after `npm run build`.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DIST = path.join(ROOT, 'dist');

const assets = JSON.parse(
  await fs.readFile(path.join(ROOT, 'scripts', 'assets.json'), 'utf8'),
);

async function built() {
  try {
    await fs.access(path.join(DIST, 'manifest.json'));
    return true;
  } catch {
    return false;
  }
}

const HAVE_DIST = await built();
const skip = HAVE_DIST ? false : 'run `npm run build` first';

async function walk(dir) {
  const out = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(full)));
    else out.push(full);
  }
  return out;
}

test('the built package declares Manifest V3', { skip }, async () => {
  const manifest = JSON.parse(
    await fs.readFile(path.join(DIST, 'manifest.json'), 'utf8'),
  );
  assert.equal(manifest.manifest_version, 3);
  assert.ok(manifest.background.service_worker);
  assert.match(manifest.content_security_policy.extension_pages, /wasm-unsafe-eval/);
  assert.ok(!/unsafe-eval[^-]/.test(manifest.content_security_policy.extension_pages));
});

test('every model the runtime loads is present', { skip }, async () => {
  for (const file of ['vision_model.onnx', 'head.onnx', 'preprocess.onnx', 'calibration.json']) {
    await fs.access(path.join(DIST, 'models', file));
  }
});

test('the bundled backbone is the pinned one', { skip }, async () => {
  const bytes = await fs.readFile(path.join(DIST, assets.backbone.dest));
  const digest = createHash('sha256').update(bytes).digest('hex');
  assert.equal(digest, assets.backbone.sha256);
});

test('the ONNX Runtime files the host needs are vendored', { skip }, async () => {
  for (const file of assets.ortFiles) {
    await fs.access(path.join(DIST, 'vendor', 'ort', file));
  }
});

test('no remote URL is loaded by shipped code', { skip }, async () => {
  const files = (await walk(DIST)).filter((f) => /\.(js|mjs|html)$/.test(f));
  const offenders = [];

  for (const file of files) {
    // The vendored runtime is third-party code we do not rewrite; it is
    // covered by the separate check that it never reaches the network for
    // model assets, since wasmPaths is set to an extension URL.
    if (file.includes(`${path.sep}vendor${path.sep}`)) continue;

    const text = await fs.readFile(file, 'utf8');
    for (const [index, line] of text.split('\n').entries()) {
      const code = line.split('//')[0];
      if (/\bfetch\s*\(\s*['"`]https?:/.test(code) || /new\s+URL\(\s*['"`]https?:/.test(code)) {
        offenders.push(`${path.relative(DIST, file)}:${index + 1}`);
      }
    }
  }

  assert.deepEqual(offenders, [], `shipped code must not fetch remote URLs`);
});

test('no build or training tooling is shipped', { skip }, async () => {
  const files = await walk(DIST);
  const forbidden = files.filter((f) =>
    /\.(py|pyc|map)$/.test(f) || f.includes('node_modules') || f.includes('.venv'),
  );
  assert.deepEqual(forbidden, []);
});

test('the calibration ships the decision threshold it was fitted for', { skip }, async () => {
  const calibration = JSON.parse(
    await fs.readFile(path.join(DIST, 'models', 'calibration.json'), 'utf8'),
  );
  assert.equal(calibration.decision_confidence, 0.65);
  assert.equal(typeof calibration.a, 'number');
  assert.equal(typeof calibration.b, 'number');
  assert.ok(calibration.dim > 0);
});

test('the license travels with the package', { skip }, async () => {
  const license = await fs.readFile(path.join(DIST, 'LICENSE'), 'utf8');
  assert.match(license, /MIT License/);
});
