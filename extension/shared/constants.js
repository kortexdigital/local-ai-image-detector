/**
 * Values shared by the content script, service worker, offscreen host,
 * popup, options page and evaluation harness.
 */

/**
 * Confidence at or above which an image is reported as AI-generated.
 *
 * The trained head is calibrated so that its balanced-accuracy-optimal
 * decision boundary lands exactly on this number, so the operating point of
 * the product and the operating point of the classifier are the same value.
 * Changing it here without recalibrating moves the boundary off the optimum.
 */
export const DECISION_CONFIDENCE = 0.65;

/** Images smaller than this on either side are skipped as icons or spacers. */
export const MIN_IMAGE_SIDE = 128;

/** Concurrent inference requests. Kept low so browsing stays responsive. */
export const MAX_CONCURRENT = 2;

/** Location of the bundled model files inside the extension package. */
export const MODEL_DIR = 'models/';

/** Square input the preprocessing graph resizes to. */
export const IMAGE_SIZE = 224;

/** Defaults for anything the user can change on the options page. */
export const DEFAULT_SETTINGS = Object.freeze({
  enabled: true,
  threshold: DECISION_CONFIDENCE,
  minImageSide: MIN_IMAGE_SIDE,
  showRealBadges: true,
});
