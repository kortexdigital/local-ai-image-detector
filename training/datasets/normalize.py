"""Single entry point for writing an image into the local cache.

Both real and synthetic images pass through this exact function, so neither
class carries a re-encoding signature the classifier could exploit.
"""
from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from training.config import CONFIG


class CacheError(Exception):
    """Raised when input bytes cannot be decoded as an image."""


def normalize_for_cache(raw: bytes) -> bytes:
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CacheError("input bytes are not a decodable image") from exc

    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")

    width, height = im.size
    longest = max(width, height)
    if longest > CONFIG.cache_max_side:
        scale = CONFIG.cache_max_side / longest
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        im = im.resize(target, Image.BICUBIC)

    out = io.BytesIO()
    im.save(
        out,
        format="JPEG",
        quality=CONFIG.cache_jpeg_quality,
        subsampling=0,
        optimize=False,
    )
    return out.getvalue()
