# Chrome Extension and Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Manifest V3 Chrome extension that scores every image on a page for being AI-generated, running all inference inside the browser, plus a harness that measures its balanced accuracy through the extension's real code path.

**Architecture:** A content script finds images and draws badges. The service worker fetches image bytes and reads generation metadata. A single offscreen document hosts ONNX Runtime Web and runs the same three graphs the training pipeline used: preprocess, frozen CLIP backbone, trained head. Everything is bundled at build time; the extension makes no network request at runtime.

**Tech Stack:** Manifest V3, ONNX Runtime Web 1.20.1 (WebGPU with WASM fallback), plain ES modules (no bundler), Node only as build tooling and test harness, Puppeteer for evaluation.

**Spec:** `docs/superpowers/specs/2026-08-14-ai-image-detector-design.md`
**Gate result this builds on:** `docs/superpowers/reports/2026-08-14-phase1-gate.md`

## Global Constraints

These come from the challenge rules and are non-negotiable. Every task inherits them.

- **No network at runtime.** After installation the extension must not fetch any model, weight, or inference asset. All assets are bundled into `dist/` at build time. A test asserts no `fetch`/`XMLHttpRequest` to a remote origin exists in shipped inference code.
- **No cloud inference, no external API, no image data leaving the device.** The only `fetch` the extension performs is for the bytes of an image already loaded on the page the user is viewing.
- **No local backend.** No Python, Node, Flask or localhost dependency at runtime. Node appears only in `scripts/` (build) and `eval/` (measurement), neither of which ships in `dist/`.
- **No hardcoded benchmark hashes or lookup tables.** No code path keyed on image identity.
- **MIT license**, `LICENSE` at repository root, and every bundled weight under a compatible permissive license recorded in `models/LICENSES.md`.
- **Reproducible from source.** `npm run build` produces `dist/` from a clean checkout, with SHA-256 verification of every downloaded artifact.
- **Confidence score displayed for every analyzed image**, and the flag threshold is 0.65 to match the evaluation.
- Public artifacts (README, code comments, commit messages) in **English**. No mention of bounties, prizes or money anywhere in the repository.
- Manifest V3 CSP for extension pages: `script-src 'self' 'wasm-unsafe-eval'; object-src 'self'`. No remote script, no `eval`.
- WASM threads stay off (`ort.env.wasm.numThreads = 1`): threading needs cross-origin isolation, which extension pages cannot reliably obtain.

---

### Task 1: Extension scaffold and the shared scoring contract

**Files:**
- Create: `extension/manifest.json`
- Create: `extension/shared/constants.js`
- Create: `extension/shared/scoring.js`
- Create: `LICENSE`
- Create: `package.json`
- Test: `eval/tests/scoring.test.mjs`

**Interfaces:**
- Produces `extension/shared/constants.js`: `DECISION_CONFIDENCE = 0.65`, `MIN_IMAGE_SIDE = 128`, `MAX_CONCURRENT = 2`, `MODEL_DIR = 'models/'`.
- Produces `extension/shared/scoring.js`:
  - `l2Normalize(Float32Array) -> Float32Array`
  - `applyCalibration({a, b}, logit) -> number` in `[0,1]`
  - `classify(confidence, threshold) -> 'ai' | 'real'`
  - `fuseMetadata(confidence, signal) -> {confidence, reason}` where `signal` is `{generatorTag: boolean, cameraExif: boolean}`

- [ ] **Step 1: Write the failing test**

```js
// eval/tests/scoring.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {l2Normalize, applyCalibration, classify, fuseMetadata} from '../../extension/shared/scoring.js';

test('l2Normalize gives a unit vector', () => {
  const out = l2Normalize(Float32Array.from([3, 4]));
  assert.ok(Math.abs(Math.hypot(out[0], out[1]) - 1) < 1e-6);
});

test('l2Normalize leaves a zero vector finite', () => {
  const out = l2Normalize(Float32Array.from([0, 0]));
  assert.ok(out.every(Number.isFinite));
});

test('calibration maps the fitted threshold onto 0.65', () => {
  const t = -2.4192640769475693;
  const cal = {a: 0.22343252720203058, b: 1.1595814950877068};
  assert.ok(Math.abs(applyCalibration(cal, t) - 0.65) < 1e-6);
});

test('calibration is monotonic and bounded', () => {
  const cal = {a: 0.5, b: 0.1};
  let prev = -1;
  for (let s = -50; s <= 50; s += 0.5) {
    const c = applyCalibration(cal, s);
    assert.ok(c >= prev - 1e-12 && c >= 0 && c <= 1);
    prev = c;
  }
});

test('classify flags at or above the threshold', () => {
  assert.equal(classify(0.65, 0.65), 'ai');
  assert.equal(classify(0.6499, 0.65), 'real');
});

test('a generation tag overrides toward ai', () => {
  const {confidence} = fuseMetadata(0.10, {generatorTag: true, cameraExif: false});
  assert.ok(confidence >= 0.95);
});

test('camera exif only nudges and never decides alone', () => {
  const {confidence} = fuseMetadata(0.90, {generatorTag: false, cameraExif: true});
  assert.ok(confidence < 0.90 && confidence > 0.65,
    'exif must not be able to flip a confident ai verdict');
});

test('no metadata leaves the model score untouched', () => {
  const {confidence} = fuseMetadata(0.42, {generatorTag: false, cameraExif: false});
  assert.equal(confidence, 0.42);
});
```

- [ ] **Step 2: Run it and watch it fail** ;  `node --test eval/tests/` fails on the missing module.
- [ ] **Step 3: Implement** `extension/shared/scoring.js` and `constants.js`, write `manifest.json` with the permissions listed in Global Constraints, `LICENSE` (MIT, current year), and `package.json` with `scripts.build`, `scripts.test`, `scripts.eval`.
- [ ] **Step 4: Run the test** ;  all pass.
- [ ] **Step 5: Commit** `feat: add extension scaffold and shared scoring contract`.

---

### Task 2: Build script that bundles every runtime asset

**Files:**
- Create: `scripts/build.mjs`
- Create: `scripts/assets.json`
- Test: `eval/tests/build.test.mjs`

**Interfaces:**
- `scripts/assets.json` lists each downloaded artifact with `url`, `dest`, `sha256`.
- `npm run build` produces `dist/` containing the extension plus `dist/models/{vision_model.onnx, preprocess.onnx, head.onnx, calibration.json}` and `dist/vendor/ort/*`.

- [ ] **Step 1: Write the failing test** asserting, after a build: `dist/manifest.json` exists; every file named in `assets.json` exists under `dist/` with the declared SHA-256; `dist/` contains no `.py`, no `node_modules`, no `.map`; and `grep` finds no `http://` or `https://` fetch target inside `dist/**/*.js` other than in comments.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement** `scripts/build.mjs`: copy `extension/` to `dist/`, copy `models/clip-vit-b32-int8/{head.onnx,calibration.json}` and the preprocess graph, download the backbone ONNX and the ORT Web distribution to their pinned SHA-256, and fail loudly on any hash mismatch.
- [ ] **Step 4: Run the test.**
- [ ] **Step 5: Commit** `feat: add reproducible build that bundles all runtime assets`.

---

### Task 3: Offscreen inference host

**Files:**
- Create: `extension/offscreen/offscreen.html`
- Create: `extension/offscreen/offscreen.js`
- Test: covered end to end by Task 7

**Interfaces:**
- `initSessions()` loads the three ONNX graphs once and reports the chosen backend (`'webgpu'` or `'wasm'`).
- `scoreImageBytes(ArrayBuffer) -> {logit, confidence, backend, dims}`.
- Message protocol: `{type: 'score', id, bytes}` in, `{type: 'scored', id, confidence, logit, backend}` out.

Preprocessing must reproduce the training path exactly: decode with `createImageBitmap`, integer centre-square crop via `drawImage` source rectangle (a pure crop, no resampling), read RGBA with `getImageData`, drop alpha into a `[1, side, side, 3]` uint8 tensor, then let `preprocess.onnx` do the resize and normalization. Resize and normalization stay inside ONNX precisely so the browser and Python cannot drift.

- [ ] **Step 1: Implement** `offscreen.html` (loads `vendor/ort/ort.min.js`, then `offscreen.js` as a module) and `offscreen.js`.
- [ ] **Step 2: Verify manually** by loading the unpacked extension and checking the service worker log reports a backend and a first score.
- [ ] **Step 3: Commit** `feat: add offscreen ONNX Runtime Web inference host`.

---

### Task 4: Service worker orchestration and metadata reading

**Files:**
- Create: `extension/background/service-worker.js`
- Create: `extension/background/metadata.js`
- Test: `eval/tests/metadata.test.mjs`

**Interfaces:**
- `extension/background/metadata.js`: `readGenerationSignals(ArrayBuffer) -> {generatorTag, cameraExif, details}`.
  - `generatorTag` true for: a PNG `tEXt`/`iTXt` chunk whose keyword is `parameters`, `prompt`, or `workflow`; an EXIF/XMP `Software` value naming a known generator; a JUMBF/C2PA box.
  - `cameraExif` true only when EXIF carries `Make` **and** `Model` **and** an exposure field.
- Service worker: keeps one offscreen document, a request queue capped at `MAX_CONCURRENT`, and a per-URL result cache.

- [ ] **Step 1: Write the failing test** over synthetic byte fixtures: a PNG with a `parameters` tEXt chunk reports `generatorTag`; a plain PNG does not; a JPEG with `Make`/`Model`/`ExposureTime` reports `cameraExif`; a JPEG with only `Make` does not; garbage bytes return all false without throwing.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement** both files.
- [ ] **Step 4: Run the test.**
- [ ] **Step 5: Commit** `feat: add service worker orchestration and generation metadata reading`.

---

### Task 5: Content script and badge overlay

**Files:**
- Create: `extension/content/content.js`
- Create: `extension/content/content.css`

**Interfaces:**
- Observes the DOM with `MutationObserver` and visibility with `IntersectionObserver`.
- Skips images whose rendered or natural side is below `MIN_IMAGE_SIDE`.
- Renders one badge per analyzed image showing the confidence as a percentage, coloured by verdict, with a title attribute carrying the backend and any metadata reason.
- Badges reposition on scroll and resize and never intercept clicks (`pointer-events: none` except on the badge itself).

- [ ] **Step 1: Implement** both files.
- [ ] **Step 2: Verify** on a local test page and on a real site, confirming badges appear and the page stays interactive.
- [ ] **Step 3: Commit** `feat: add content script with per-image confidence badges`.

---

### Task 6: Popup and options

**Files:**
- Create: `extension/popup/popup.html`, `popup.js`, `popup.css`
- Create: `extension/options/options.html`, `options.js`

**Interfaces:**
- Popup lists the current tab's analyzed images with confidence, sorted most-suspicious first, and shows the active backend.
- Options persist `enabled`, `threshold` (default 0.65), and `minImageSide` to `chrome.storage.sync`.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify** the popup reflects a real page and options round-trip.
- [ ] **Step 3: Commit** `feat: add popup summary and options page`.

---

### Task 7: Evaluation harness and the parity test

This is the task that proves the number. It drives the shipped extension, not a reimplementation.

**Files:**
- Create: `eval/run-benchmark.mjs`
- Create: `eval/runner.html`
- Create: `eval/parity.mjs`
- Create: `eval/README.md`

**Interfaces:**
- `node eval/run-benchmark.mjs --real <dir> --ai <dir> [--threshold 0.65]` loads the built extension into headless Chrome, opens `chrome-extension://<id>/eval/runner.html`, pushes image bytes in, and collects confidences from the extension's own `scoreImageBytes`.
- Reports balanced accuracy at the threshold, confusion matrix, per-directory breakdown, abstention rate under the alternative reading of the threshold, and latency per backend.
- `node eval/parity.mjs --images <dir>` compares the browser's feature vector against the Python one for the same images and fails if the maximum absolute difference exceeds `1e-3`.

- [ ] **Step 1: Implement** `runner.html` (an extension page that exposes `window.__score(bytes)`), then `run-benchmark.mjs` and `parity.mjs`.
- [ ] **Step 2: Run parity** against a fixed image set. Any real divergence blocks the rest.
- [ ] **Step 3: Run the benchmark** over the held-out generators used in the Phase 1 gate and confirm the in-browser balanced accuracy matches the Python figure within a point.
- [ ] **Step 4: Commit** `feat: add in-browser evaluation harness and preprocessing parity test`.

---

### Task 8: Documentation, compliance audit and release

**Files:**
- Create: `README.md`
- Create: `docs/COMPLIANCE.md`
- Modify: `models/LICENSES.md`

- [ ] **Step 1: Write `README.md`**: what it does, how it works, exact build and install steps, how to run the harness, model provenance and licenses, and a plain statement of the privacy properties.
- [ ] **Step 2: Write `docs/COMPLIANCE.md`** mapping each challenge rule to the file and mechanism that satisfies it.
- [ ] **Step 3: Run the audit**: fresh clone into a temporary directory, `npm ci && npm run build`, load unpacked in a clean Chrome profile, confirm badges appear, then disable the network and confirm scoring still works.
- [ ] **Step 4: Commit and push** the public repository.

---

## Self-Review

**Rule coverage.** Every rule in the challenge maps to a task: no cloud inference and no external API (Tasks 2, 3, 7 and the build test that scans for remote fetch targets), no backend process (Task 2 keeps Node in build and eval only), no post-install downloads (Task 2 bundles everything), automatic analysis (Task 5), a confidence score for every analyzed image (Tasks 5, 6), build and install instructions (Task 8), reproducible from source (Task 2 hash pinning plus Task 8 clean-clone audit), MIT (Task 1), and no hardcoded benchmark hashes (no task introduces one; Task 8 audits for it).

**Interface consistency.** `applyCalibration({a, b}, logit)` matches the `a`/`b`/`t_star` fields written by `training/export/head_onnx.py`. `scoreImageBytes` returns the `logit` the head's ONNX `score` output produces, which is the same quantity `raw_scores` returns in Python. `DECISION_CONFIDENCE` is the single source of the 0.65 threshold across content script, popup, options default and harness.

**Open risk.** The backbone that ships is the int8 export, so the head must be trained on int8 features rather than fp32 ones; otherwise the browser runs an approximation of a model trained on something else. That extraction and gate rerun happen before Task 2 pins the asset.
