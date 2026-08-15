import hashlib
import io
from pathlib import Path

from PIL import Image

from training.datasets.fetch import cache_image, image_bytes_from_cell
from training.datasets.sources import Source

SRC = Source("t", "some/repo", 1, "flux", "train", "image", 10, 1000)


def _png(color=(1, 2, 3), size=(40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_cache_image_writes_a_file_and_returns_a_row(tmp_path: Path):
    row = cache_image(_png(), SRC, tmp_path)
    assert row is not None
    assert (tmp_path / row.relpath).exists()
    assert row.label == 1
    assert row.generator == "flux"
    assert row.source_key == "t"


def test_relpath_is_partitioned_by_generator(tmp_path: Path):
    row = cache_image(_png(), SRC, tmp_path)
    assert row.relpath.startswith("flux/")
    assert row.relpath.endswith(".jpg")


def test_sha256_is_of_the_normalized_bytes_not_the_input(tmp_path: Path):
    raw = _png()
    row = cache_image(raw, SRC, tmp_path)
    stored = (tmp_path / row.relpath).read_bytes()
    assert row.sha256 == hashlib.sha256(stored).hexdigest()
    assert row.sha256 != hashlib.sha256(raw).hexdigest()


def test_duplicate_image_is_skipped_when_its_digest_is_already_known(tmp_path: Path):
    raw = _png()
    known: set[str] = set()
    first = cache_image(raw, SRC, tmp_path, known)
    assert first is not None
    assert first.sha256 in known
    assert cache_image(raw, SRC, tmp_path, known) is None


def test_a_file_on_disk_without_a_manifest_entry_is_still_recovered(tmp_path: Path):
    """An interrupted run leaves images written but unrecorded.

    If presence on disk were treated as proof of a manifest entry, those
    images could never be recovered: every retry would report zero new
    images while the manifest stayed short.
    """
    raw = _png()
    orphan = cache_image(raw, SRC, tmp_path, known=None)
    assert orphan is not None
    assert (tmp_path / orphan.relpath).exists()

    # Same bytes, fresh run whose manifest does not list this digest yet.
    recovered = cache_image(raw, SRC, tmp_path, known=set())
    assert recovered is not None
    assert recovered.sha256 == orphan.sha256


def test_undecodable_image_is_skipped_not_raised(tmp_path: Path):
    assert cache_image(b"garbage", SRC, tmp_path) is None


def test_image_bytes_from_cell_reads_the_huggingface_image_struct():
    raw = _png()
    assert image_bytes_from_cell({"bytes": raw, "path": "x.png"}) == raw


def test_image_bytes_from_cell_accepts_plain_bytes():
    raw = _png()
    assert image_bytes_from_cell(raw) == raw


def test_image_bytes_from_cell_returns_none_for_unusable_cells():
    assert image_bytes_from_cell(None) is None
    assert image_bytes_from_cell({"path": "x.png", "bytes": None}) is None
    assert image_bytes_from_cell(42) is None
