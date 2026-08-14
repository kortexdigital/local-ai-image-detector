"""Extract and cache features for every image in the manifest.

Features are computed once and written to disk because the head is retrained
many times during tuning, and re-running the backbone for each of those runs
would dominate the cost.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from training.datasets.manifest import read_manifest
from training.features.augment import augment
from training.features.backbone import FeatureExtractor, backbone_by_key

log = logging.getLogger(__name__)


def extract_all(
    backbone_key: str,
    manifest_path: Path,
    cache_dir: Path,
    out_dir: Path,
    models_dir: Path,
    seed: int,
    augment_fraction: float,
) -> Path:
    backbone = backbone_by_key(backbone_key)
    extractor = FeatureExtractor(backbone, models_dir)
    rows = read_manifest(manifest_path)
    rng = np.random.default_rng(seed)

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    kept: list[tuple[str, str, int, int]] = []

    started = time.time()
    for index, row in enumerate(rows):
        path = cache_dir / row.relpath
        if not path.exists():
            continue
        try:
            im = Image.open(path)
            im.load()
        except Exception:
            log.warning("undecodable cached file %s", path)
            continue

        augmented = rng.random() < augment_fraction
        if augmented:
            # Seeded per index so a rerun reproduces the same degradations.
            im = augment(im, np.random.default_rng(seed + index))

        vectors.append(extractor.embed(im))
        labels.append(row.label)
        kept.append((row.relpath, row.generator, row.label, int(augmented)))

        if index and index % 500 == 0:
            rate = index / max(1e-9, time.time() - started)
            remaining = (len(rows) - index) / max(1e-9, rate)
            log.info(
                "%s: %d/%d at %.1f img/s, ~%.0f s left",
                backbone_key,
                index,
                len(rows),
                rate,
                remaining,
            )

    if not vectors:
        raise RuntimeError(f"no features extracted for {backbone_key}")

    target = out_dir / backbone_key
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "features.npy", np.stack(vectors).astype(np.float32))
    np.save(target / "labels.npy", np.asarray(labels, dtype=np.int64))
    with (target / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("relpath", "generator", "label", "augmented"))
        writer.writerows(kept)
    log.info("wrote %d feature vectors to %s", len(vectors), target)
    return target
