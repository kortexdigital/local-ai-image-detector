"""Train a shallow head on frozen features.

The head is a stack of dense layers with ReLU between them and no activation
on the output, so a linear head is just the one-layer case and both export to
the same ONNX shape.

Three things here were driven by measurement rather than by taste:

The head is an MLP, not linear. A linear head could not represent GAN
detection and diffusion detection at once, losing seven points on held-out
FLUX when GAN data was added.

Several heads are trained with different seeds and their logits averaged. A
single seed of an early-stopped MLP is noisy; averaging five is worth about a
point of balanced accuracy and costs nothing at inference beyond a slightly
wider graph.

The input carries the log of the embedding norm alongside the L2-normalized
embedding. Normalizing throws the norm away, and the norm turns out to be
mildly informative.
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
ENSEMBLE_SEEDS = 5


@dataclass(frozen=True)
class TrainedHead:
    """One or more layer stacks whose logits are averaged."""

    members: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]
    dim: int
    kind: str
    hyperparams: dict[str, Any] = field(default_factory=dict)

    @property
    def layers(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """The single member's layers, for the one-member case."""
        if len(self.members) != 1:
            raise ValueError("head has several members; use .members")
        return self.members[0]


def linear_head(weights: np.ndarray, bias: float) -> TrainedHead:
    weights = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    return TrainedHead(
        members=(((weights, np.array([bias], dtype=np.float32)),),),
        dim=int(weights.shape[0]),
        kind="linear",
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def build_features(embeddings: np.ndarray) -> np.ndarray:
    """The head's input: the unit-length embedding plus its log magnitude.

    L2 normalization discards the norm, which carries a little signal of its
    own. Appending its log costs one dimension and measured about a third of a
    point.
    """
    embeddings = embeddings.astype(np.float32)
    normalized = l2_normalize(embeddings)
    log_norm = np.log(np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return np.hstack([normalized, log_norm.astype(np.float32)])


def _run_member(layers, features: np.ndarray) -> np.ndarray:
    activations = np.asarray(features, dtype=np.float32)
    last = len(layers) - 1
    for index, (weights, bias) in enumerate(layers):
        activations = activations @ weights + bias
        if index < last:
            activations = np.maximum(activations, 0.0)
    return activations.reshape(-1)


def raw_scores(head: TrainedHead, features: np.ndarray) -> np.ndarray:
    """Average the members' pre-sigmoid logits."""
    total = np.zeros(len(features), dtype=np.float64)
    for layers in head.members:
        total += _run_member(layers, features)
    return (total / len(head.members)).astype(np.float32)


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for cls in (0, 1):
        mask = labels == cls
        if not mask.any():
            continue
        scores.append(float((predictions[mask] == cls).mean()))
    return float(np.mean(scores))


def _logistic_member(model: LogisticRegression):
    return (
        (
            model.coef_.reshape(-1, 1).astype(np.float32),
            np.array([float(model.intercept_[0])], dtype=np.float32),
        ),
    )


def _mlp_member(model: MLPClassifier):
    return tuple(
        (np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32))
        for w, b in zip(model.coefs_, model.intercepts_)
    )


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    split: Split,
    seed: int,
    selection_indices: np.ndarray | None = None,
) -> TrainedHead:
    """Fit candidates and keep the one that generalizes best.

    `selection_indices` chooses which images decide the winner. Passing the
    calibration half of the held-out generators picks the head that does best
    on generators it never saw, which is what the deployment faces, while
    leaving the reporting half untouched. Defaults to the in-distribution
    validation set.
    """
    x_train, y_train = features[split.train], labels[split.train]
    chooser = split.val_seen if selection_indices is None else selection_indices
    x_val, y_val = features[chooser], labels[chooser]

    candidates: list[TrainedHead] = []

    for c in C_GRID:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=seed
        )
        model.fit(x_train, y_train)
        candidates.append(
            TrainedHead(
                members=(_logistic_member(model),),
                dim=int(features.shape[1]),
                kind="linear",
                hyperparams={"C": c},
            )
        )

    for hidden in HIDDEN_SIZES:
        members = []
        for offset in range(ENSEMBLE_SEEDS):
            model = MLPClassifier(
                hidden_layer_sizes=(hidden,),
                max_iter=400,
                random_state=seed + offset,
                early_stopping=True,
            )
            model.fit(x_train, y_train)
            members.append(_mlp_member(model))
        candidates.append(
            TrainedHead(
                members=tuple(members),
                dim=int(features.shape[1]),
                kind="mlp",
                hyperparams={"hidden_layer_sizes": [hidden], "members": ENSEMBLE_SEEDS},
            )
        )

    best: TrainedHead | None = None
    best_score = -1.0
    for head in candidates:
        predictions = (raw_scores(head, x_val) > 0).astype(int)
        score = balanced_accuracy(y_val, predictions)
        log.info("%s %s selection balanced accuracy=%.4f", head.kind, head.hyperparams, score)
        if score > best_score:
            best_score, best = score, head

    assert best is not None
    log.info("selected %s %s", best.kind, best.hyperparams)
    return best
