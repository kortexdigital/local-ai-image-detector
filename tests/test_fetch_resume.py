"""Resuming an interrupted download must not lose what was already cached."""
from pathlib import Path

from training.datasets.fetch import merge_manifest_rows, remaining_for_source
from training.datasets.manifest import ManifestRow
from training.datasets.sources import Source

SRC = Source("flux_coco", "some/repo", 1, "flux", "train", "image", 800, 100000)


def _row(digest: str, source_key: str = "flux_coco") -> ManifestRow:
    return ManifestRow(digest, f"flux/{digest}.jpg", source_key, "flux", 1)


def test_remaining_is_the_full_target_when_nothing_is_cached():
    assert remaining_for_source(SRC, []) == 800


def test_remaining_shrinks_by_what_the_manifest_already_holds():
    existing = [_row(f"{i:064x}") for i in range(300)]
    assert remaining_for_source(SRC, existing) == 500


def test_remaining_is_zero_for_a_completed_source():
    existing = [_row(f"{i:064x}") for i in range(800)]
    assert remaining_for_source(SRC, existing) == 0


def test_rows_from_other_sources_do_not_count():
    existing = [_row(f"{i:064x}", source_key="flux_faces") for i in range(400)]
    assert remaining_for_source(SRC, existing) == 800


def test_merge_keeps_existing_rows_and_appends_new_ones():
    existing = [_row("a" * 64), _row("b" * 64)]
    fresh = [_row("c" * 64)]
    merged = merge_manifest_rows(existing, fresh)
    assert [r.sha256 for r in merged] == ["a" * 64, "b" * 64, "c" * 64]


def test_merge_drops_duplicates_by_digest():
    existing = [_row("a" * 64)]
    fresh = [_row("a" * 64), _row("d" * 64)]
    merged = merge_manifest_rows(existing, fresh)
    assert [r.sha256 for r in merged] == ["a" * 64, "d" * 64]


def test_merge_of_nothing_new_preserves_everything():
    existing = [_row("a" * 64), _row("b" * 64)]
    assert merge_manifest_rows(existing, []) == existing
