"""Build the preprocessing ONNX graph shared by training and the browser.

Keeping resize and normalization inside ONNX means onnxruntime in Python and
onnxruntime-web in the extension execute the same arithmetic. Doing this in
host code instead would leave resampling to PIL on one side and to canvas on
the other, which do not agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

OPSET = 17


@dataclass(frozen=True)
class PreprocessSpec:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    size: int
    square_crop: bool


# OpenAI CLIP normalization constants.
CLIP_SPEC = PreprocessSpec(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    size=224,
    square_crop=True,
)

# SigLIP squashes to a square without cropping and normalizes to [-1, 1].
SIGLIP_SPEC = PreprocessSpec(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    size=224,
    square_crop=False,
)


def center_square_crop(im: Image.Image) -> Image.Image:
    """Crop to a centered square of the shorter side, in integer arithmetic."""
    width, height = im.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return im.crop((left, top, left + side, top + side))


def build_preprocess_graph(spec: PreprocessSpec, path: Path) -> Path:
    mean = np.array(spec.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(spec.std, dtype=np.float32).reshape(1, 3, 1, 1)

    initializers = [
        numpy_helper.from_array(np.array([255.0], dtype=np.float32), name="scale255"),
        numpy_helper.from_array(mean, name="mean"),
        numpy_helper.from_array(std, name="std"),
        numpy_helper.from_array(np.array([], dtype=np.float32), name="roi"),
        numpy_helper.from_array(np.array([], dtype=np.float32), name="scales"),
        numpy_helper.from_array(
            np.array([1, 3, spec.size, spec.size], dtype=np.int64), name="sizes"
        ),
    ]

    nodes = [
        helper.make_node("Cast", ["pixels"], ["as_float"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["as_float"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "Resize",
            ["nchw", "roi", "scales", "sizes"],
            ["resized"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            nearest_mode="floor",
        ),
        helper.make_node("Div", ["resized", "scale255"], ["unit"]),
        helper.make_node("Sub", ["unit", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "std"], ["pixel_values"]),
    ]

    graph = helper.make_graph(
        nodes,
        "preprocess",
        inputs=[
            helper.make_tensor_value_info("pixels", TensorProto.UINT8, [1, "H", "W", 3])
        ],
        outputs=[
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, [1, 3, spec.size, spec.size]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
