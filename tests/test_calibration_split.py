"""The threshold must never be fitted on the images it is reported against."""
import numpy as np

from training.head.splits import split_for_calibration


GENS = np.array(["flux"] * 100 + ["biggan"] * 60 + ["real"] * 160)
LABELS = np.array([1] * 160 + [0] * 160)
IDX = np.arange(len(GENS))


def test_halves_are_disjoint_and_cover_everything():
    calib, report = split_for_calibration(IDX, GENS, LABELS, seed=1)
    joined = np.concatenate([calib, report])
    assert len(joined) == len(set(joined.tolist())) == len(IDX)


def test_every_generator_appears_in_both_halves():
    calib, report = split_for_calibration(IDX, GENS, LABELS, seed=1)
    for half in (calib, report):
        assert set(GENS[half]) == set(GENS)


def test_both_halves_carry_both_classes():
    calib, report = split_for_calibration(IDX, GENS, LABELS, seed=1)
    for half in (calib, report):
        assert set(LABELS[half].tolist()) == {0, 1}


def test_halves_are_roughly_equal_in_size():
    calib, report = split_for_calibration(IDX, GENS, LABELS, seed=1)
    assert abs(len(calib) - len(report)) <= len(set(GENS))


def test_split_is_deterministic_for_a_seed():
    a1, b1 = split_for_calibration(IDX, GENS, LABELS, seed=7)
    a2, b2 = split_for_calibration(IDX, GENS, LABELS, seed=7)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
