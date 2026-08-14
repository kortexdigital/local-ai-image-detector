"""Central configuration for the offline training pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generators deliberately excluded from training. Balanced accuracy on these is
# the only honest estimate of performance on an unseen evaluation set.
HELD_OUT_GENERATORS: tuple[str, ...] = ("flux", "midjourney-v6", "stylegan3")


@dataclass(frozen=True)
class Config:
    data_dir: Path
    cache_dir: Path
    features_dir: Path
    models_dir: Path
    seed: int
    cache_max_side: int
    cache_jpeg_quality: int
    image_size: int
    decision_confidence: float


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


CONFIG = Config(
    data_dir=_path_from_env("AIID_DATA_DIR", REPO_ROOT / "data"),
    cache_dir=_path_from_env("AIID_CACHE_DIR", REPO_ROOT / "data" / "cache"),
    features_dir=_path_from_env("AIID_FEATURES_DIR", REPO_ROOT / "data" / "features"),
    models_dir=_path_from_env("AIID_MODELS_DIR", REPO_ROOT / "models"),
    seed=20260814,
    cache_max_side=512,
    cache_jpeg_quality=95,
    image_size=224,
    decision_confidence=0.65,
)
