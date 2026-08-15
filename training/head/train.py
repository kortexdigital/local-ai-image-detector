"""Train a shallow head on frozen features.

The head is a stack of dense layers with ReLU between them and no activation
on the output, so a linear head is just the one-layer case and both export to
the same ONNX shape.

A linear head was the starting point, on the reasoning that more capacity
overfits the generators present in training. Measurement disagreed: with
GAN-family and diffusion generators both in training, a linear head could not
hold both axes at once and lost seven points on held-out FLUX when GAN data
was added. One hidden layer recovers that and lifts held-out balanced
accuracy from 0.82 to 0.85.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from training.head.splits import Split

log = logging.getLogger(__name__)

C_GRID = (0.1, 1.0, 10.0)
HIDDEN_SIZES = (256, 512)


@dataclass(frozen=True)
class TrainedHead:
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    dim: int
    kind: str
    hyperparams: dict[str, Any] = field(default_factory=dict)


def linear_head(weights: np.ndarray, bias: float) -> TrainedHead:
    weights = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    return TrainedHead(
        layers=((weights, np.array([bias], dtype=np.float32)),),
        dim=int(weights.shape[0]),
        kind="linear",
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def raw_scores(head: TrainedHead, features: np.ndarray) -> np.ndarray:
    """Run the layer stack, returning the pre-sigmoid logit."""
    activations = np.asarray(features, dtype=np.float32)
    last = len(head.layers) - 1
    for index, (weights, bias) in enumerate(head.layers):
        activations = activations @ weights + bias
        if index < last:
            activations = np.maximum(activations, 0.0)
    return activations.reshape(-1)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for cls in (0, 1):
        mask = labels == cls
        if not mask.any():
            continue
        scores.append(float((predictions[mask] == cls).mean()))
    return float(np.mean(scores))


def _from_logistic(model: LogisticRegression) -> TrainedHead:
    head = linear_head(model.coef_.reshape(-1), float(model.intercept_[0]))
    return TrainedHead(
        layers=head.layers, dim=head.dim, kind="linear", hyperparams={"C": model.C}
    )


def _from_mlp(model: MLPClassifier) -> TrainedHead:
    layers = tuple(
        (
            np.asarray(w, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
        )
        for w, b in zip(model.coefs_, model.intercepts_)
    )
    return TrainedHead(
        layers=layers,
        dim=int(layers[0][0].shape[0]),
        kind="mlp",
        hyperparams={"hidden_layer_sizes": list(model.hidden_layer_sizes)},
    )


def train_head(
    features: np.ndarray, labels: np.ndarray, split: Split, seed: int
) -> TrainedHead:
    normalized = l2_normalize(features.astype(np.float32))
    x_train, y_train = normalized[split.train], labels[split.train]
    x_val, y_val = normalized[split.val_seen], labels[split.val_seen]

    candidates: list[TrainedHead] = []

    for c in C_GRID:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=seed
        )
        model.fit(x_train, y_train)
        candidates.append(_from_logistic(model))

    for hidden in HIDDEN_SIZES:
        model = MLPClassifier(
            hidden_layer_sizes=(hidden,),
            max_iter=400,
            random_state=seed,
            early_stopping=True,
        )
        model.fit(x_train, y_train)
        candidates.append(_from_mlp(model))

    best: TrainedHead | None = None
    best_score = -1.0
    for head in candidates:
        predictions = (raw_scores(head, x_val) > 0).astype(int)
        score = balanced_accuracy(y_val, predictions)
        log.info("%s %s val_seen balanced accuracy=%.4f", head.kind, head.hyperparams, score)
        if score > best_score:
            best_score, best = score, head

    assert best is not None
    log.info("selected %s %s", best.kind, best.hyperparams)
    return best
