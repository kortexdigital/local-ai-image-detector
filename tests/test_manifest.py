from pathlib import Path

from training.config import HELD_OUT_GENERATORS
from training.datasets.manifest import ManifestRow, read_manifest, write_manifest
from training.datasets.sources import SOURCES


def test_manifest_roundtrips(tmp_path: Path):
    rows = [
        ManifestRow("a" * 64, "real/coco/aaa.jpg", "coco", "real", 0),
        ManifestRow("b" * 64, "fake/flux/bbb.jpg", "flux", "flux", 1),
    ]
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    assert read_manifest(path) == rows


def test_sources_cover_both_classes():
    assert any(s.label == 0 for s in SOURCES)
    assert any(s.label == 1 for s in SOURCES)


def test_source_keys_are_unique():
    keys = [s.key for s in SOURCES]
    assert len(keys) == len(set(keys))


def test_real_sources_use_the_real_generator_marker():
    for s in SOURCES:
        assert (s.generator == "real") == (s.label == 0)


def test_every_held_out_generator_has_a_source():
    available = {s.generator for s in SOURCES}
    for generator in HELD_OUT_GENERATORS:
        assert generator in available, f"no source provides held-out generator {generator}"


def test_synthetic_sources_span_multiple_generators():
    synthetic = {s.generator for s in SOURCES if s.label == 1}
    assert len(synthetic) >= 6, "cross-generator generalization needs generator diversity"
