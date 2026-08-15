"""Image cells arrive in several shapes across dataset publishers."""
import base64
import io

from PIL import Image

from training.datasets.fetch import image_bytes_from_cell


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (7, 8, 9)).save(buf, format="PNG")
    return buf.getvalue()


def test_huggingface_image_struct():
    raw = _png()
    assert image_bytes_from_cell({"bytes": raw, "path": "a.png"}) == raw


def test_plain_bytes():
    raw = _png()
    assert image_bytes_from_cell(raw) == raw


def test_base64_string():
    """Some publishers store the image as a base64 string, not as bytes."""
    raw = _png()
    encoded = base64.b64encode(raw).decode("ascii")
    assert image_bytes_from_cell(encoded) == raw


def test_base64_data_uri():
    raw = _png()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    assert image_bytes_from_cell(uri) == raw


def test_unusable_cells_return_none():
    assert image_bytes_from_cell(None) is None
    assert image_bytes_from_cell(42) is None
    assert image_bytes_from_cell({"path": "a.png", "bytes": None}) is None
    assert image_bytes_from_cell("not base64 at all !!!") is None
