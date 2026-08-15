"""Partition data so validation measures generalization, not memorization.

Random splits leak: the same scene, prompt or capture session lands on both
sides. Held-out generators are stronger still, because they estimate what
happens on a generator the training never saw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val_seen: np.ndarray
    val_unseen: np.ndarray


def split_for_calibration(
    indices: np.ndarray,
    generators: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Halve a validation set into calibration and reporting parts.

    Fitting the decision threshold on the same images the score is reported
    against inflates that score: the threshold is chosen knowing the answers.
    Splitting per generator keeps both halves representative, so the reported
    number is what an unseen benchmark would actually see.
    """
    rng = np.random.default_rng(seed)
    calibration: list[np.ndarray] = []
    report: list[np.ndarray] = []

    generators = np.asarray(generators)
    for generator in sorted(set(generators.tolist())):
        for cls in (0, 1):
            mask = (generators == generator) & (labels == cls)
            group = np.asarray(indices)[mask] if len(indices) == len(mask) else None
            if group is None:
                group = np.flatnonzero(mask)
            if group.size == 0:
                continue
            shuffled = rng.permutation(group)
            cut = len(shuffled) // 2
            calibration.append(shuffled[:cut])
            report.append(shuffled[cut:])

    return (
        np.sort(np.concatenate(calibration)) if calibration else np.array([], dtype=int),
        np.sort(np.concatenate(report)) if report else np.array([], dtype=int),
    )


def make_splits(
    generators: list[str],
    labels: np.ndarray,
    held_out: tuple[str, ...],
    seed: int,
    val_fraction: float = 0.2,
) -> Split:
    generators_arr = np.asarray(generators)
    rng = np.random.default_rng(seed)

    unseen_mask = np.isin(generators_arr, held_out)
    seen_indices = np.flatnonzero(~unseen_mask)
    unseen_indices = np.flatnonzero(unseen_mask)

    # Real images are reserved for val_unseen too, so that split has both
    # classes and a balanced accuracy computed on it is meaningful.
    real_seen = seen_indices[labels[seen_indices] == 0]
    reserved_real = rng.permutation(real_seen)[: max(1, len(unseen_indices))]

    remaining = np.setdiff1d(seen_indices, reserved_real, assume_unique=False)
    shuffled = rng.permutation(remaining)
    cut = int(len(shuffled) * val_fraction)

    return Split(
        train=np.sort(shuffled[cut:]),
        val_seen=np.sort(shuffled[:cut]),
        val_unseen=np.sort(np.concatenate([unseen_indices, reserved_real])),
    )
