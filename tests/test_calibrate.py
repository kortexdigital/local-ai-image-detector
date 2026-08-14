import numpy as np

from training.head.calibrate import (
    apply,
    best_threshold,
    dead_zone_fraction,
    fit_calibration,
)
from training.head.train import balanced_accuracy


def _scores_and_labels(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = rng.normal(-1.0, 1.0, n)
    scores[labels == 1] = rng.normal(1.5, 1.0, n // 2)
    return scores, labels


def test_best_threshold_lands_between_the_two_modes():
    scores, labels = _scores_and_labels()
    t = best_threshold(scores, labels)
    assert scores[labels == 0].mean() < t < scores[labels == 1].mean()


def test_calibration_maps_the_optimal_threshold_onto_the_decision_confidence():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    mapped = apply(cal, np.array([cal.t_star]))[0]
    assert abs(mapped - 0.65) < 1e-6


def test_calibration_is_monotonic():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    grid = np.linspace(-8, 8, 400)
    out = apply(cal, grid)
    assert np.all(np.diff(out) >= -1e-12)


def test_confidences_stay_inside_the_unit_interval():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    out = apply(cal, np.linspace(-100, 100, 1000))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_thresholding_at_065_matches_thresholding_the_raw_score_at_t_star():
    scores, labels = _scores_and_labels()
    cal = fit_calibration(scores, labels, decision_confidence=0.65)
    from_raw = (scores >= cal.t_star).astype(int)
    from_confidence = (apply(cal, scores) >= 0.65).astype(int)
    assert np.array_equal(from_raw, from_confidence)


def test_sharpness_shrinks_the_dead_zone_without_moving_the_boundary():
    scores, labels = _scores_and_labels()
    soft = fit_calibration(scores, labels, 0.65, sharpness=1.0)
    sharp = fit_calibration(scores, labels, 0.65, sharpness=3.0)

    soft_predictions = (apply(soft, scores) >= 0.65).astype(int)
    sharp_predictions = (apply(sharp, scores) >= 0.65).astype(int)
    assert np.array_equal(soft_predictions, sharp_predictions)
    assert balanced_accuracy(labels, soft_predictions) == balanced_accuracy(
        labels, sharp_predictions
    )

    assert dead_zone_fraction(apply(sharp, scores)) < dead_zone_fraction(
        apply(soft, scores)
    )
