import numpy as np

from training.report import evaluate, render_markdown


def test_evaluate_reports_balanced_accuracy_at_the_decision_confidence():
    labels = np.array([0, 0, 1, 1])
    confidences = np.array([0.10, 0.20, 0.90, 0.80])
    generators = ["real", "real", "flux", "flux"]
    result = evaluate(confidences, labels, generators, 0.65)
    assert result.balanced_accuracy == 1.0
    assert result.tpr == 1.0
    assert result.tnr == 1.0
    assert result.n == 4


def test_confidence_just_below_the_threshold_counts_as_real():
    labels = np.array([1, 0])
    confidences = np.array([0.6499, 0.10])
    result = evaluate(confidences, labels, ["flux", "real"], 0.65)
    assert result.tpr == 0.0
    assert result.tnr == 1.0
    assert result.balanced_accuracy == 0.5


def test_per_generator_breakdown_uses_recall_within_each_generator():
    labels = np.array([1, 1, 1, 1, 0, 0])
    confidences = np.array([0.9, 0.9, 0.2, 0.2, 0.1, 0.1])
    generators = ["flux", "flux", "sd", "sd", "real", "real"]
    result = evaluate(confidences, labels, generators, 0.65)
    assert result.per_generator["flux"] == 1.0
    assert result.per_generator["sd"] == 0.0
    assert result.per_generator["real"] == 1.0


def test_threshold_curve_covers_the_unit_interval():
    labels = np.array([0, 1])
    result = evaluate(np.array([0.2, 0.8]), labels, ["real", "flux"], 0.65)
    xs = [t for t, _ in result.threshold_curve]
    assert min(xs) <= 0.05 and max(xs) >= 0.95
    assert all(0.0 <= ba <= 1.0 for _, ba in result.threshold_curve)


def test_render_markdown_states_the_headline_number():
    labels = np.array([0, 1])
    result = evaluate(np.array([0.2, 0.8]), labels, ["real", "flux"], 0.65)
    text = render_markdown(result, "Phase 1")
    assert "Phase 1" in text
    assert "1.0000" in text or "1.000" in text
