from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from training.features.preprocess_graph import (
    CLIP_SPEC,
    SIGLIP_SPEC,
    build_preprocess_graph,
    center_square_crop,
)


def _session(spec, tmp_path: Path) -> ort.InferenceSession:
    path = build_preprocess_graph(spec, tmp_path / "pre.onnx")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_output_shape_and_dtype(tmp_path: Path):
    sess = _session(CLIP_SPEC, tmp_path)
    pixels = np.zeros((1, 137, 91, 3), dtype=np.uint8)
    out = sess.run(None, {"pixels": pixels})[0]
    assert out.shape == (1, 3, CLIP_SPEC.size, CLIP_SPEC.size)
    assert out.dtype == np.float32


def test_normalization_matches_a_numpy_reference(tmp_path: Path):
    sess = _session(CLIP_SPEC, tmp_path)
    # A constant image isolates normalization from resampling.
    pixels = np.full((1, 300, 300, 3), 128, dtype=np.uint8)
    out = sess.run(None, {"pixels": pixels})[0]
    expected = [(128 / 255.0 - m) / s for m, s in zip(CLIP_SPEC.mean, CLIP_SPEC.std)]
    for channel, value in enumerate(expected):
        assert np.allclose(out[0, channel], value, atol=1e-5)


def test_arbitrary_input_sizes_are_accepted(tmp_path: Path):
    sess = _session(SIGLIP_SPEC, tmp_path)
    for h, w in ((32, 512), (999, 17), (224, 224)):
        out = sess.run(None, {"pixels": np.zeros((1, h, w, 3), dtype=np.uint8)})[0]
        assert out.shape == (1, 3, SIGLIP_SPEC.size, SIGLIP_SPEC.size)


def test_channel_order_is_preserved(tmp_path: Path):
    sess = _session(CLIP_SPEC, tmp_path)
    pixels = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    pixels[..., 0] = 255  # pure red in RGB
    out = sess.run(None, {"pixels": pixels})[0]
    red = (1.0 - CLIP_SPEC.mean[0]) / CLIP_SPEC.std[0]
    green = (0.0 - CLIP_SPEC.mean[1]) / CLIP_SPEC.std[1]
    assert abs(float(out[0, 0].mean()) - red) < 1e-3
    assert abs(float(out[0, 1].mean()) - green) < 1e-3


def test_center_square_crop_produces_a_square_of_the_shorter_side():
    im = Image.new("RGB", (300, 200))
    cropped = center_square_crop(im)
    assert cropped.size == (200, 200)


def test_center_square_crop_is_centered():
    im = Image.new("RGB", (5, 1), (0, 0, 0))
    im.putpixel((2, 0), (255, 255, 255))
    assert center_square_crop(im).getpixel((0, 0)) == (255, 255, 255)
