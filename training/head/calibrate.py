"""Map raw scores onto a confidence whose decision point is 0.65.

Evaluation thresholds confidence at 0.65 rather than 0.5. A conventionally
calibrated model puts its optimal boundary at 0.5, so thresholding it at 0.65
shifts predictions toward the negative class and costs balanced accuracy.

The fix is a monotonic affine-in-logit map constrained so that the raw
threshold maximizing balanced accuracy lands exactly on 0.65. The extension
flags images at that same 0.65, so the product's operating point and the
classifier's optimal operating point are the same number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    a: float
    b: float
    t_star: float
    decision_confidence: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    parts = []
    for cls in (0, 1):
        mask = labels == cls
        if mask.any():
            parts.append(float((predictions[mask] == cls).mean()))
    return float(np.mean(parts))


def family_weights(families: np.ndarray) -> np.ndarray:
    """Weight each generator family equally, whatever its sample count.

    The held-out set is not evenly split: one hard family with many images
    drags the threshold toward appeasing it, at the cost of every other
    family and of the real images. Equal weight per family stops the mixture
    of the validation set from deciding the operating point.
    """
    families = np.asarray(families)
    weights = np.ones(len(families), dtype=np.float64)
    for name in set(families.tolist()):
        mask = families == name
        weights[mask] = 1.0 / float(mask.sum())
    return weights * (len(families) / weights.sum())


def _weighted_balanced_accuracy(labels, predictions, weights) -> float:
    parts = []
    for cls in (0, 1):
        mask = labels == cls
        if mask.any():
            total = weights[mask].sum()
            hit = weights[mask][predictions[mask] == cls].sum()
            parts.append(float(hit / total) if total else 0.0)
    return float(np.mean(parts))


def best_threshold(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> float:
    if weights is not None:
        candidates = np.unique(scores)
        if candidates.size > 2000:
            candidates = np.quantile(scores, np.linspace(0.0, 1.0, 2000))
        best_t, best_score = float(candidates[0]), -1.0
        for t in candidates:
            score = _weighted_balanced_accuracy(labels, (scores >= t).astype(int), weights)
            if score > best_score:
                best_score, best_t = score, float(t)
        return best_t

    candidates = np.unique(scores)
    if candidates.size > 2000:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 2000))
    best_t, best_score = float(candidates[0]), -1.0
    for t in candidates:
        score = _balanced_accuracy(labels, (scores >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def fit_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    decision_confidence: float,
    sharpness: float = 3.0,
    families: np.ndarray | None = None,
) -> Calibration:
    """Fit the map from logit to confidence.

    `sharpness` scales the slope while holding the crossing at t_star, so it
    changes how many images land in the low-confidence band without moving a
    single prediction. The default is above 1 deliberately: at 1.0 roughly a
    third of images sat between 0.35 and 0.65, which is fine when the
    threshold is read as a decision boundary and bad if it is read as an
    abstention band. Raising it to 3 cuts that to under a tenth and leaves
    balanced accuracy identical.
    """
    weights = None if families is None else family_weights(families)
    t_star = best_threshold(scores, labels, weights)

    # Slope from the spread of the scores, then scaled by the sharpness knob.
    spread = float(np.std(scores)) or 1.0
    a = sharpness * (2.0 / spread)

    # Constrain the map so that t_star lands on the decision confidence.
    b = _logit(decision_confidence) - a * t_star

    return Calibration(
        a=a, b=b, t_star=t_star, decision_confidence=decision_confidence
    )


def apply(cal: Calibration, scores: np.ndarray) -> np.ndarray:
    return _sigmoid(cal.a * np.asarray(scores, dtype=np.float64) + cal.b)


def dead_zone_fraction(
    confidences: np.ndarray, low: float = 0.35, high: float = 0.65
) -> float:
    inside = (confidences >= low) & (confidences < high)
    return float(inside.mean())
