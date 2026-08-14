"""Degradations that mimic what the web does to images.

Evaluation samples are described as web-realistic, which in practice means
recompressed, resized and rescaled. A detector trained only on pristine files
learns artifacts that these operations destroy.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

AUG_PROBABILITIES: dict[str, float] = {
    "jpeg": 0.60,
    "webp": 0.15,
    "downscale": 0.45,
    "crop": 0.35,
    "blur": 0.15,
    "noise": 0.15,
}


def _recompress(im: Image.Image, fmt: str, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def augment(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    im = im.convert("RGB")

    if rng.random() < AUG_PROBABILITIES["crop"]:
        width, height = im.size
        keep = float(rng.uniform(0.70, 1.0))
        new_w, new_h = max(16, int(width * keep)), max(16, int(height * keep))
        left = int(rng.integers(0, max(1, width - new_w + 1)))
        top = int(rng.integers(0, max(1, height - new_h + 1)))
        im = im.crop((left, top, left + new_w, top + new_h))

    if rng.random() < AUG_PROBABILITIES["downscale"]:
        width, height = im.size
        factor = float(rng.uniform(0.40, 1.0))
        small = (max(16, int(width * factor)), max(16, int(height * factor)))
        im = im.resize(small, Image.BICUBIC).resize((width, height), Image.BICUBIC)

    if rng.random() < AUG_PROBABILITIES["blur"]:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.9))))

    if rng.random() < AUG_PROBABILITIES["noise"]:
        arr = np.asarray(im, dtype=np.float32)
        arr += rng.normal(0.0, float(rng.uniform(1.0, 5.0)), arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if rng.random() < AUG_PROBABILITIES["webp"]:
        im = _recompress(im, "WEBP", int(rng.integers(40, 96)))

    if rng.random() < AUG_PROBABILITIES["jpeg"]:
        im = _recompress(im, "JPEG", int(rng.integers(30, 96)))

    return im
