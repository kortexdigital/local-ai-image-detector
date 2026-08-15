/**
 * Turning a model output into the number the user sees.
 *
 * The head emits a logit. Calibration is a monotonic map from that logit onto
 * a confidence in [0, 1], fitted during training under the constraint that the
 * threshold maximizing balanced accuracy lands on DECISION_CONFIDENCE. Keeping
 * the map monotonic matters: it means thresholding the confidence and
 * thresholding the raw logit produce identical decisions.
 */

/** Scale a vector to unit length, leaving a zero vector finite. */
export function l2Normalize(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) sum += vector[i] * vector[i];
  const norm = Math.max(Math.sqrt(sum), 1e-8);
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i += 1) out[i] = vector[i] / norm;
  return out;
}

/**
 * Map a raw logit onto a calibrated confidence.
 * @param {{a: number, b: number}} calibration
 * @param {number} logit
 */
export function applyCalibration(calibration, logit) {
  const z = calibration.a * logit + calibration.b;
  // Clamp before exp so a large magnitude logit saturates instead of
  // producing Infinity and then NaN.
  const clamped = Math.max(-60, Math.min(60, z));
  return 1 / (1 + Math.exp(-clamped));
}

/** @returns {'ai' | 'real'} */
export function classify(confidence, threshold) {
  return confidence >= threshold ? 'ai' : 'real';
}

/**
 * Combine the model confidence with what the file's own metadata says.
 *
 * The two directions are deliberately asymmetric. An embedded generation tag
 * (an Automatic1111 parameters chunk, a ComfyUI workflow, a C2PA manifest, a
 * generator named in Software) is close to proof, so it overrides. Camera EXIF
 * is only weak evidence: it is trivially forged and routinely added by
 * re-saving, so it nudges and is bounded so it can never flip a confident
 * AI verdict to real. Calling a real photograph AI-generated is the more
 * damaging error, which is why only the AI direction gets an override.
 *
 * @param {number} confidence model confidence in [0, 1]
 * @param {{generatorTag: boolean, cameraExif: boolean}} signals
 * @returns {{confidence: number, reason: string}}
 */
export function fuseMetadata(confidence, signals) {
  if (signals.generatorTag) {
    return {
      confidence: Math.max(confidence, 0.98),
      reason: 'embedded generation metadata',
    };
  }
  if (signals.cameraExif) {
    // A bounded multiplicative nudge. At most it moves a 0.90 to 0.855, so a
    // confident detection survives and a borderline one tips toward real.
    const nudged = confidence * 0.95;
    return {
      confidence: Math.max(0, Math.min(1, nudged)),
      reason: 'camera EXIF present',
    };
  }
  return { confidence, reason: '' };
}
