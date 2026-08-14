import io

import pytest
from PIL import Image

from training.config import CONFIG
from training.datasets.normalize import CacheError, normalize_for_cache


def _encode(im: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def test_png_input_becomes_jpeg():
    im = Image.new("RGB", (300, 200), (10, 120, 240))
    out = normalize_for_cache(_encode(im, "PNG"))
    assert Image.open(io.BytesIO(out)).format == "JPEG"


def test_large_image_is_downscaled_to_max_side():
    im = Image.new("RGB", (2000, 1000), (5, 5, 5))
    out = Image.open(io.BytesIO(normalize_for_cache(_encode(im, "JPEG"))))
    assert max(out.size) == CONFIG.cache_max_side
    assert out.size == (CONFIG.cache_max_side, CONFIG.cache_max_side // 2)


def test_small_image_is_never_upscaled():
    im = Image.new("RGB", (64, 48), (200, 100, 50))
    out = Image.open(io.BytesIO(normalize_for_cache(_encode(im, "PNG"))))
    assert out.size == (64, 48)


def test_grayscale_and_rgba_are_converted_to_rgb():
    for mode, fmt in (("L", "PNG"), ("RGBA", "PNG")):
        im = Image.new(mode, (50, 50))
        out = Image.open(io.BytesIO(normalize_for_cache(_encode(im, fmt))))
        assert out.mode == "RGB"


def test_normalizing_twice_is_stable_in_size():
    im = Image.new("RGB", (800, 600), (33, 66, 99))
    once = normalize_for_cache(_encode(im, "PNG"))
    twice = normalize_for_cache(once)
    assert Image.open(io.BytesIO(once)).size == Image.open(io.BytesIO(twice)).size


def test_undecodable_bytes_raise_cache_error():
    with pytest.raises(CacheError):
        normalize_for_cache(b"this is not an image")
