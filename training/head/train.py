"""Train a shallow head on frozen features.

The head stays linear on purpose. A large head on frozen features overfits
the generators present in training, which is exactly the failure mode this
project is built to avoid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from training.head.splits import Split

log = logging.getLogger(__name__)

C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class TrainedHead:
    weights: np.ndarray
    bias: float
    dim: int
    C: float


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def raw_scores(head: TrainedHead, features: np.ndarray) -> np.ndarray:
    return features @ head.weights + head.bias


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for cls in (0, 1):
        mask = labels == cls
        if not mask.any():
            continue
        scores.append(float((predictions[mask] == cls).mean()))
    return float(np.mean(scores))


def train_head(
    features: np.ndarray, labels: np.ndarray, split: Split, seed: int
) -> TrainedHead:
    normalized = l2_normalize(features.astype(np.float32))
    x_train, y_train = normalized[split.train], labels[split.train]
    x_val, y_val = normalized[split.val_seen], labels[split.val_seen]

    best: TrainedHead | None = None
    best_score = -1.0
    for c in C_GRID:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=seed
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        score = balanced_accuracy(y_val, predictions)
        log.info("C=%s val_seen balanced accuracy=%.4f", c, score)
        if score > best_score:
            best_score = score
            best = TrainedHead(
                weights=model.coef_.reshape(-1).astype(np.float32),
                bias=float(model.intercept_[0]),
                dim=int(normalized.shape[1]),
                C=c,
            )
    assert best is not None
    return best
