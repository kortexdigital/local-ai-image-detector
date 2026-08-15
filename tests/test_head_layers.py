"""The head may be linear or an MLP; both are the same layer stack."""
import numpy as np

from training.head.splits import Split
from training.head.train import TrainedHead, linear_head, raw_scores, train_head


def _linear(w, b) -> TrainedHead:
    return linear_head(np.asarray(w, dtype=np.float32), float(b))


def test_linear_head_is_a_single_layer():
    head = _linear([1.0, -2.0], 0.5)
    assert len(head.layers) == 1
    assert head.kind == "linear"
    assert head.dim == 2


def test_raw_scores_of_a_linear_head():
    head = _linear([1.0, -2.0], 0.5)
    x = np.array([[3.0, 1.0]], dtype=np.float32)
    assert abs(float(raw_scores(head, x)[0]) - (3.0 - 2.0 + 0.5)) < 1e-6


def test_raw_scores_apply_relu_between_layers_but_not_after_the_last():
    # Hidden layer forces one unit negative; ReLU must clamp it to zero.
    w1 = np.array([[1.0, -1.0]], dtype=np.float32)  # dim=1 -> hidden=2
    b1 = np.array([0.0, 0.0], dtype=np.float32)
    w2 = np.array([[1.0], [1.0]], dtype=np.float32)  # hidden=2 -> 1
    b2 = np.array([-0.25], dtype=np.float32)
    head = TrainedHead(layers=((w1, b1), (w2, b2)), dim=1, kind="mlp", hyperparams={})

    # x=2 -> hidden (2, -2) -> relu (2, 0) -> 2 - 0.25
    assert abs(float(raw_scores(head, np.array([[2.0]], dtype=np.float32))[0]) - 1.75) < 1e-6
    # A negative output would prove the last layer got clamped, which it must not.
    assert float(raw_scores(head, np.array([[0.0]], dtype=np.float32))[0]) == -0.25


def _separable(n=600, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    features = rng.normal(0, 1, (n, dim)).astype(np.float32)
    offset = np.zeros(dim, dtype=np.float32)
    offset[0] = 6.0
    features[labels == 1] += offset
    return features, labels


def test_train_head_returns_a_usable_head_of_either_kind():
    features, labels = _separable()
    idx = np.arange(len(labels))
    split = Split(train=idx[::2], val_seen=idx[1::4], val_unseen=idx[3::4])
    head = train_head(features, labels, split, seed=0)
    assert head.kind in {"linear", "mlp"}
    assert head.dim == features.shape[1]
    scores = raw_scores(head, features)
    assert scores.shape == (len(labels),)
    assert scores.min() < 0.0 < scores.max()
