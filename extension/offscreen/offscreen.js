/**
 * Inference host.
 *
 * Runs the same three ONNX graphs the training pipeline ran, in the same
 * order, with the same arithmetic:
 *
 *   preprocess.onnx   uint8 [1,H,W,3]  ->  float32 [1,3,224,224]
 *   vision_model.onnx float32 tensor    ->  512-d image embedding
 *   head.onnx         512-d (L2 normed) ->  a single logit
 *
 * Resize and normalization live inside preprocess.onnx rather than in this
 * file on purpose. Canvas resampling and PIL resampling do not agree, so
 * doing it in host code would leave the browser computing slightly different
 * features from the ones the head was trained on. Keeping those operations in
 * a shared graph removes the possibility rather than testing for it later.
 *
 * The only work done here in host code is the centre-square crop, which is
 * integer arithmetic and a pure copy, so both sides agree exactly.
 */

import { DECISION_CONFIDENCE, IMAGE_SIZE, MODEL_DIR } from '../shared/constants.js';
import { readGenerationSignals } from '../shared/metadata.js';
import { applyCalibration, classify, fuseMetadata, l2Normalize } from '../shared/scoring.js';

const ort = globalThis.ort;

ort.env.wasm.wasmPaths = chrome.runtime.getURL('vendor/ort/');
// Threads need cross-origin isolation, which extension pages cannot reliably
// obtain. Single-threaded SIMD is enough for the per-image budget here.
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';

/** @type {{preprocess: any, backbone: any, head: any, calibration: any, backend: string} | null} */
let state = null;
let initPromise = null;

async function fetchLocal(relativePath) {
  const response = await fetch(chrome.runtime.getURL(relativePath));
  if (!response.ok) {
    throw new Error(`missing bundled asset ${relativePath} (${response.status})`);
  }
  return response;
}

async function createSessions(providers) {
  const options = { executionProviders: providers, graphOptimizationLevel: 'all' };
  const [preprocessBytes, backboneBytes, headBytes] = await Promise.all([
    (await fetchLocal(`${MODEL_DIR}preprocess.onnx`)).arrayBuffer(),
    (await fetchLocal(`${MODEL_DIR}vision_model.onnx`)).arrayBuffer(),
    (await fetchLocal(`${MODEL_DIR}head.onnx`)).arrayBuffer(),
  ]);
  return {
    // The preprocessing graph stays on WASM: it is tiny, and its Resize op is
    // the least uniformly supported node across WebGPU builds. Running it on
    // CPU keeps the numbers identical to training on every machine.
    preprocess: await ort.InferenceSession.create(preprocessBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }),
    backbone: await ort.InferenceSession.create(backboneBytes, options),
    head: await ort.InferenceSession.create(headBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }),
  };
}

async function init() {
  const calibration = await (await fetchLocal(`${MODEL_DIR}calibration.json`)).json();

  let sessions;
  let backend;
  const webgpuAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator;

  if (webgpuAvailable) {
    try {
      sessions = await createSessions(['webgpu']);
      backend = 'webgpu';
    } catch (error) {
      console.warn('WebGPU unavailable, falling back to WASM:', error);
    }
  }
  if (!sessions) {
    sessions = await createSessions(['wasm']);
    backend = 'wasm';
  }

  state = { ...sessions, calibration, backend };
  return state;
}

function ensureReady() {
  if (!initPromise) initPromise = init();
  return initPromise;
}

/**
 * Decode bytes and produce the uint8 NHWC tensor the preprocessing graph
 * expects, cropped to a centred square.
 */
async function toPixelTensor(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const side = Math.min(bitmap.width, bitmap.height);
  const left = Math.floor((bitmap.width - side) / 2);
  const top = Math.floor((bitmap.height - side) / 2);

  const canvas = new OffscreenCanvas(side, side);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  // Source and destination rectangles are the same size, so this is a copy
  // rather than a resample. All scaling happens inside the ONNX graph.
  context.drawImage(bitmap, left, top, side, side, 0, 0, side, side);
  const { data } = context.getImageData(0, 0, side, side);
  bitmap.close();

  // RGBA to RGB. Training images were all re-encoded to JPEG and carry no
  // alpha, so images with transparency land on the canvas default of black.
  const rgb = new Uint8Array(side * side * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j] = data[i];
    rgb[j + 1] = data[i + 1];
    rgb[j + 2] = data[i + 2];
  }

  return {
    tensor: new ort.Tensor('uint8', rgb, [1, side, side, 3]),
    width: bitmap.width,
    height: bitmap.height,
  };
}

/**
 * Score raw image bytes.
 * @param {ArrayBuffer|Uint8Array} bytes
 * @returns {Promise<{logit: number, confidence: number, backend: string, width: number, height: number, embedding?: Float32Array}>}
 */
export async function scoreImageBytes(bytes, { includeEmbedding = false } = {}) {
  const ready = await ensureReady();
  const { tensor, width, height } = await toPixelTensor(bytes);

  const preOut = await ready.preprocess.run({ pixels: tensor });
  const pixelValues = preOut[ready.preprocess.outputNames[0]];

  const backboneInput = ready.backbone.inputNames[0];
  const backboneOut = await ready.backbone.run({ [backboneInput]: pixelValues });
  const embedding = backboneOut[ready.backbone.outputNames[0]].data;

  const normalized = l2Normalize(embedding);
  const headOut = await ready.head.run({
    features: new ort.Tensor('float32', normalized, [1, normalized.length]),
  });
  const logit = headOut[ready.head.outputNames[0]].data[0];
  const confidence = applyCalibration(ready.calibration, logit);

  const result = {
    logit,
    confidence,
    backend: ready.backend,
    width,
    height,
  };
  if (includeEmbedding) result.embedding = Array.from(normalized);
  return result;
}

/**
 * Fetch an image by URL and score it, metadata included.
 *
 * The fetch happens here rather than in the service worker because
 * chrome.runtime.sendMessage serializes as JSON: an ArrayBuffer does not
 * survive the trip, and converting to base64 would cost a third more bytes
 * and a copy per image. This document runs at the extension origin and
 * carries the same host permissions, so it can fetch the bytes directly.
 */
export async function scoreImageUrl(url, threshold) {
  const response = await fetch(url, { credentials: 'omit', cache: 'force-cache' });
  if (!response.ok) throw new Error(`image fetch failed (${response.status})`);
  const bytes = await response.arrayBuffer();

  const signals = readGenerationSignals(bytes);
  const scored = await scoreImageBytes(bytes);
  const fused = fuseMetadata(scored.confidence, signals);

  return {
    url,
    confidence: fused.confidence,
    modelConfidence: scored.confidence,
    logit: scored.logit,
    verdict: classify(fused.confidence, threshold ?? DECISION_CONFIDENCE),
    reason: fused.reason || signals.details,
    backend: scored.backend,
    width: scored.width,
    height: scored.height,
  };
}

/** Exposed so the evaluation harness drives the shipped code path, not a copy. */
globalThis.__aiidScoreBytes = async (bytes, options) => scoreImageBytes(bytes, options);
globalThis.__aiidReady = async () => {
  const ready = await ensureReady();
  return { backend: ready.backend, threshold: DECISION_CONFIDENCE, imageSize: IMAGE_SIZE };
};

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== 'offscreen') return false;

  if (message.type === 'scoreUrl') {
    scoreImageUrl(message.url, message.threshold)
      .then((result) => sendResponse({ ok: true, ...result }))
      .catch((error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }

  if (message.type === 'status') {
    ensureReady()
      .then((ready) => sendResponse({ ok: true, backend: ready.backend }))
      .catch((error) => sendResponse({ ok: false, error: String(error?.message ?? error) }));
    return true;
  }

  return false;
});

// Warm the sessions as soon as the document exists so the first image on a
// page does not pay the model load.
ensureReady().catch((error) => console.error('inference host failed to start', error));
