import numpy as np
from PIL import Image

from training.features.augment import AUG_PROBABILITIES, augment


def _photo(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))


def test_output_is_rgb_pil_image():
    out = augment(_photo(), np.random.default_rng(1))
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_same_seed_gives_identical_output():
    a = augment(_photo(), np.random.default_rng(7))
    b = augment(_photo(), np.random.default_rng(7))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_different_seeds_give_different_output():
    a = augment(_photo(), np.random.default_rng(1))
    b = augment(_photo(), np.random.default_rng(2))
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_augmentation_actually_changes_pixels_over_many_seeds():
    original = np.asarray(_photo())
    changed = sum(
        not np.array_equal(np.asarray(augment(_photo(), np.random.default_rng(s))), original)
        for s in range(20)
    )
    assert changed >= 18, "augmentation is firing too rarely to build robustness"


def test_all_declared_probabilities_are_in_range():
    assert AUG_PROBABILITIES
    assert all(0.0 <= p <= 1.0 for p in AUG_PROBABILITIES.values())
