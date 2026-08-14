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
