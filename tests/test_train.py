import numpy as np

from training.head.splits import Split
from training.head.train import (
    TrainedHead,
    balanced_accuracy,
    build_features,
    l2_normalize,
    raw_scores,
    train_head,
)


def test_l2_normalize_gives_unit_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(x)
    assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-6
    assert np.all(np.isfinite(out[1]))  # zero row must not produce NaN


def test_balanced_accuracy_ignores_class_imbalance():
    labels = np.array([0] * 90 + [1] * 10)
    always_zero = np.zeros(100, dtype=int)
    assert balanced_accuracy(labels, always_zero) == 0.5


def test_balanced_accuracy_is_one_for_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    assert balanced_accuracy(labels, labels) == 1.0


def _separable(n: int = 400, dim: int = 16, seed: int = 0):
    """Two classes separated along one axis.

    The offset has to be large relative to the vector norm because the head
    L2-normalizes its input, which discards magnitude and keeps only
    direction. A small offset survives normalization only weakly.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    offset = np.zeros(dim, dtype=np.float32)
    offset[0] = 6.0
    features = rng.normal(0, 1, (n, dim)).astype(np.float32)
    features[labels == 1] += offset
    return features, labels


def test_train_head_learns_a_separable_problem():
    features, labels = _separable()
    idx = np.arange(len(labels))
    split = Split(train=idx[::2], val_seen=idx[1::4], val_unseen=idx[3::4])
    inputs = build_features(features)
    head = train_head(inputs, labels, split, seed=0)

    assert isinstance(head, TrainedHead)
    assert head.dim == inputs.shape[1]

    scores = raw_scores(head, inputs[split.val_unseen])
    predictions = (scores > 0).astype(int)
    assert balanced_accuracy(labels[split.val_unseen], predictions) > 0.9


def test_raw_scores_are_logits_not_probabilities():
    features, labels = _separable()
    idx = np.arange(len(labels))
    split = Split(train=idx[::2], val_seen=idx[1::4], val_unseen=idx[3::4])
    inputs = build_features(features)
    head = train_head(inputs, labels, split, seed=0)
    scores = raw_scores(head, inputs)
    assert scores.min() < 0.0 < scores.max()
