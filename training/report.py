"""Turn confidences into the numbers that decide whether this approach works."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.head.calibrate import dead_zone_fraction


@dataclass(frozen=True)
class EvalResult:
    balanced_accuracy: float
    tpr: float
    tnr: float
    dead_zone: float
    per_generator: dict[str, float]
    threshold_curve: list[tuple[float, float]]
    n: int


def _rates(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    positive, negative = labels == 1, labels == 0
    tpr = float((predictions[positive] == 1).mean()) if positive.any() else 0.0
    tnr = float((predictions[negative] == 0).mean()) if negative.any() else 0.0
    return tpr, tnr


def evaluate(
    confidences: np.ndarray,
    labels: np.ndarray,
    generators: list[str],
    decision_confidence: float,
) -> EvalResult:
    confidences = np.asarray(confidences, dtype=np.float64)
    labels = np.asarray(labels)
    predictions = (confidences >= decision_confidence).astype(int)
    tpr, tnr = _rates(labels, predictions)

    generators_arr = np.asarray(generators)
    per_generator: dict[str, float] = {}
    for name in sorted(set(generators)):
        mask = generators_arr == name
        expected = int(labels[mask][0])
        per_generator[name] = float((predictions[mask] == expected).mean())

    curve = []
    for t in np.linspace(0.02, 0.98, 49):
        t_predictions = (confidences >= t).astype(int)
        a, b = _rates(labels, t_predictions)
        curve.append((round(float(t), 4), round((a + b) / 2.0, 4)))

    return EvalResult(
        balanced_accuracy=(tpr + tnr) / 2.0,
        tpr=tpr,
        tnr=tnr,
        dead_zone=dead_zone_fraction(confidences, 0.35, decision_confidence),
        per_generator=per_generator,
        threshold_curve=curve,
        n=int(labels.size),
    )


def render_markdown(result: EvalResult, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Images evaluated: {result.n}",
        f"- **Balanced accuracy at the decision threshold: {result.balanced_accuracy:.4f}**",
        f"- True positive rate (synthetic detected): {result.tpr:.4f}",
        f"- True negative rate (real kept): {result.tnr:.4f}",
        f"- Fraction in the low-confidence dead zone: {result.dead_zone:.4f}",
        "",
        "## Per generator",
        "",
        "| Generator | Accuracy |",
        "|---|---|",
    ]
    for name, value in sorted(result.per_generator.items(), key=lambda kv: kv[1]):
        lines.append(f"| {name} | {value:.4f} |")
    lines += [
        "",
        "## Balanced accuracy versus threshold",
        "",
        "| Threshold | BA |",
        "|---|---|",
    ]
    lines += [f"| {t:.2f} | {ba:.4f} |" for t, ba in result.threshold_curve]
    return "\n".join(lines) + "\n"
