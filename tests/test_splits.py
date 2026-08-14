import numpy as np

from training.head.splits import make_splits

GENS = (
    ["real"] * 40 + ["stable-diffusion-1"] * 20 + ["glide"] * 20 + ["flux"] * 20
)
LABELS = np.array([0] * 40 + [1] * 60)
HELD = ("flux",)


def test_held_out_generator_never_appears_in_train():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    assert all(GENS[i] != "flux" for i in split.train)


def test_held_out_generator_appears_in_val_unseen():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    assert any(GENS[i] == "flux" for i in split.val_unseen)


def test_val_unseen_contains_real_images_too():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    assert any(LABELS[i] == 0 for i in split.val_unseen)
    assert any(LABELS[i] == 1 for i in split.val_unseen)


def test_splits_are_disjoint_and_cover_everything():
    split = make_splits(GENS, LABELS, HELD, seed=1)
    joined = np.concatenate([split.train, split.val_seen, split.val_unseen])
    assert len(joined) == len(set(joined.tolist())) == len(GENS)


def test_splits_are_deterministic_for_a_seed():
    a = make_splits(GENS, LABELS, HELD, seed=42)
    b = make_splits(GENS, LABELS, HELD, seed=42)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val_unseen, b.val_unseen)
