/**
 * Bridge between an automation driver and the extension's real scoring code.
 *
 * The harness hands over base64 because that is what survives the driver
 * boundary; it is decoded here and fed to exactly the functions the offscreen
 * document uses when the user browses.
 */

import { DECISION_CONFIDENCE } from '../shared/constants.js';
import { readGenerationSignals } from '../shared/metadata.js';
import { classify, fuseMetadata } from '../shared/scoring.js';
import { preprocessBytes, scoreImageBytes } from '../offscreen/offscreen.js';

function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** Score one image given as base64, through the shipped path. */
async function score(base64, { threshold = DECISION_CONFIDENCE, includeEmbedding = false } = {}) {
  const bytes = base64ToBytes(base64);
  const signals = readGenerationSignals(bytes.buffer);
  const scored = await scoreImageBytes(bytes, { includeEmbedding });
  const fused = fuseMetadata(scored.confidence, signals);

  return {
    confidence: fused.confidence,
    modelConfidence: scored.confidence,
    logit: scored.logit,
    verdict: classify(fused.confidence, threshold),
    backend: scored.backend,
    reason: fused.reason || signals.details,
    width: scored.width,
    height: scored.height,
    embedding: scored.embedding,
  };
}

/**
 * Preprocessing only: decode, crop, and run preprocess.onnx.
 *
 * The parity check compares this against the Python pipeline. It is the stage
 * the design guarantees to be identical, because both sides execute the same
 * graph, so any drift here is a real defect rather than a numerical artifact
 * of a quantized kernel.
 */
async function preprocess(base64) {
  const result = await preprocessBytes(base64ToBytes(base64));
  return {
    width: result.width,
    height: result.height,
    side: result.side,
    pixelSum: result.pixelSum,
    values: Array.from(result.values),
  };
}

globalThis.__aiidScore = score;
globalThis.__aiidPreprocess = preprocess;
globalThis.__aiidStatus = async () => globalThis.__aiidReady();

globalThis.__aiidReady().then(
  (info) => {
    document.getElementById('state').textContent = `ready on ${info.backend}`;
    globalThis.__aiidLoaded = true;
  },
  (error) => {
    document.getElementById('state').textContent = `failed: ${error.message}`;
    globalThis.__aiidLoadError = String(error.message ?? error);
  },
);
