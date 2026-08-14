from pathlib import Path

from training.config import CONFIG, Config, HELD_OUT_GENERATORS


def test_config_defaults_are_absolute_paths():
    assert isinstance(CONFIG, Config)
    for p in (CONFIG.data_dir, CONFIG.cache_dir, CONFIG.features_dir, CONFIG.models_dir):
        assert isinstance(p, Path)
        assert p.is_absolute()


def test_decision_confidence_is_the_evaluation_threshold():
    assert CONFIG.decision_confidence == 0.65


def test_cache_settings_match_the_disk_budget():
    assert CONFIG.cache_max_side == 512
    assert CONFIG.cache_jpeg_quality == 95


def test_held_out_generators_are_declared_and_non_empty():
    assert len(HELD_OUT_GENERATORS) >= 3
    assert all(isinstance(g, str) and g for g in HELD_OUT_GENERATORS)
