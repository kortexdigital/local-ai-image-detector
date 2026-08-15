/**
 * Bridge between an automation driver and the extension's real scoring code.
 *
 * The harness hands over base64 because that is what survives the driver
 * boundary; it is decoded here and fed to exactly the function the offscreen
 * document uses when the user browses.
 */

import { DECISION_CONFIDENCE } from '../shared/constants.js';
import { readGenerationSignals } from '../shared/metadata.js';
import { classify, fuseMetadata } from '../shared/scoring.js';
import { scoreImageBytes } from '../offscreen/offscreen.js';

function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/**
 * Score one image given as base64.
 * @returns {Promise<{confidence: number, modelConfidence: number, logit: number,
 *   verdict: string, backend: string, reason: string, embedding?: number[]}>}
 */
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

globalThis.__aiidScore = score;
globalThis.__aiidStatus = async () => {
  const probe = await globalThis.__aiidReady();
  return probe;
};

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
