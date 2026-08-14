import numpy as np
import pytest
from PIL import Image

from training.config import CONFIG
from training.features.backbone import BACKBONES, FeatureExtractor, download_backbone


def test_every_backbone_declares_a_permissive_license():
    permissive = {"MIT", "Apache-2.0", "BSD-3-Clause"}
    assert BACKBONES
    for b in BACKBONES:
        assert b.license_name in permissive, f"{b.key} license is not MIT-compatible"


def test_backbone_keys_are_unique():
    keys = [b.key for b in BACKBONES]
    assert len(keys) == len(set(keys))


@pytest.mark.slow
def test_extractor_produces_a_deterministic_fixed_length_vector():
    backbone = BACKBONES[0]
    download_backbone(backbone, CONFIG.models_dir)
    extractor = FeatureExtractor(backbone, CONFIG.models_dir)

    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 256, (321, 456, 3), dtype=np.uint8))

    first = extractor.embed(im)
    second = extractor.embed(im)

    assert first.ndim == 1
    assert first.shape[0] >= 256
    assert first.dtype == np.float32
    assert np.allclose(first, second, atol=0)


@pytest.mark.slow
def test_extractor_accepts_images_of_any_aspect_ratio():
    backbone = BACKBONES[0]
    download_backbone(backbone, CONFIG.models_dir)
    extractor = FeatureExtractor(backbone, CONFIG.models_dir)
    dim = None
    for size in ((64, 512), (512, 64), (224, 224)):
        vec = extractor.embed(Image.new("RGB", size, (120, 30, 200)))
        dim = dim or vec.shape[0]
        assert vec.shape[0] == dim


@pytest.mark.slow
def test_different_images_produce_different_embeddings():
    backbone = BACKBONES[0]
    download_backbone(backbone, CONFIG.models_dir)
    extractor = FeatureExtractor(backbone, CONFIG.models_dir)
    a = extractor.embed(Image.new("RGB", (224, 224), (10, 10, 10)))
    b = extractor.embed(Image.new("RGB", (224, 224), (240, 30, 90)))
    assert not np.allclose(a, b, atol=1e-4)
