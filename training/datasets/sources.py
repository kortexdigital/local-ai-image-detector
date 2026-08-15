"""Registry of dataset sources.

Every repo_id here was verified against the HuggingFace datasets-server by
scripts/survey_sources.py, which also recorded the row count and the image
column. Do not add an entry without running that script: a dataset existing
on the Hub does not mean it is queryable, and several candidates that exist
return HTTP 500 or 501 from the rows endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    repo_id: str
    label: int  # 0 real, 1 synthetic
    generator: str  # "real" for real sources, otherwise the generator name
    split: str
    image_column: str
    target_count: int
    total_rows: int


# Real images, deliberately spread across scenes, objects, textures, faces and
# animals. A real set made only of faces teaches the head that "face means
# real", which collapses the moment it meets a synthetic portrait.
_REAL: tuple[Source, ...] = (
    Source("coco", "bitmind/MS-COCO", 0, "real", "train", "image", 1800, 566747),
    Source("openimages", "bitmind/open-images-v7-subset", 0, "real", "train", "image", 1800, 1743042),
    Source("caltech256", "bitmind/caltech-256", 0, "real", "train", "image", 1200, 30607),
    Source("dtd", "bitmind/dtd", 0, "real", "train", "image", 800, 5640),
    Source("ffhq", "bitmind/ffhq-256", 0, "real", "train", "image", 1200, 70000),
    Source("bm_real", "bitmind/bm-real", 0, "real", "train", "image", 1200, 28393),
    Source("lfw", "bitmind/lfw", 0, "real", "train", "image", 500, 13233),
    Source("afhq", "bitmind/AFHQ", 0, "real", "train", "image", 500, 15803),
)

# Synthetic images from generators the head is allowed to learn from.
_SYNTHETIC_SEEN: tuple[Source, ...] = (
    Source("wukong", "bitmind/GenImage_wukong", 1, "wukong", "train", "image", 600, 162000),
    Source("glide", "bitmind/GenImage_glide", 1, "glide", "train", "image", 600, 162000),
    Source("vqdm", "bitmind/GenImage_VQDM", 1, "vqdm", "train", "image", 600, 162000),
    Source("adm", "bitmind/GenImage_ADM", 1, "adm", "train", "image", 600, 162000),
    Source("mj_genimage", "bitmind/GenImage_MidJourney", 1, "midjourney-genimage", "train", "image", 600, 162000),
    Source("journeydb", "bitmind/JourneyDB", 1, "midjourney", "train", "image", 600, 670368),
    Source("sdxl", "bitmind/bm-subnet-sdxl-256", 1, "sdxl", "train", "image", 600, 25218),
    Source("realvis", "bitmind/realvis-xl___individual-files", 1, "realvis-xl", "train", "image", 500, 6207),
    Source("sd_flickr", "bitmind/flickr30k-stable-diffusion", 1, "sd-flickr", "train", "image", 600, 24811),
    Source("mobius", "bitmind/bm-mobius", 1, "mobius", "train", "image", 500, 9831),
    Source("imagine", "bitmind/bm-imagine", 1, "imagine", "train", "image", 500, 8707),
    Source("leonardo", "bitmind/Deepfake-leonardo-stablecog", 1, "leonardo", "train", "image", 400, 2434),
    # GAN-era generators. Without these every seen generator is a diffusion or
    # transformer model, and the head never learns what a GAN artifact looks
    # like. The first measured gate scored 0.2767 on held-out BigGAN for
    # exactly that reason: it was being asked to extrapolate to an entire
    # family it had zero examples of.
    # gojay/StyleGAN2-Face is deliberately absent: its parquet carries
    # `bytes: None` with an hf:// path, so the images live as separate repo
    # files that shard fetching cannot reach.
    Source("stylegan3", "34data/stylegan3_T_FFHQU_processed", 1, "stylegan3", "train", "image", 600, 8000),
    Source("stargan", "34data/STARGAN", 1, "stargan", "train", "image", 500, 5648),
)

# Synthetic images from the held-out generators. These are downloaded like any
# other source; the split logic in training/head/splits.py is what keeps them
# out of training.
_SYNTHETIC_HELD_OUT: tuple[Source, ...] = (
    Source("flux_coco", "bitmind/MS-COCO-unique___FLUX.1-dev", 1, "flux", "train", "image", 800, 112907),
    Source("flux_faces", "bitmind/celeb-a-hq___FLUX.1-dev", 1, "flux", "train", "image", 400, 30000),
    Source("nano_banana", "bitmind/nano-banana", 1, "nano-banana", "train", "image", 800, 9457),
    Source("biggan", "bitmind/GenImage_BigGAN", 1, "biggan", "train", "image", 600, 162000),
)

SOURCES: tuple[Source, ...] = _REAL + _SYNTHETIC_SEEN + _SYNTHETIC_HELD_OUT
