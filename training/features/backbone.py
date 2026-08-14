"""Frozen vision backbones, downloaded as ONNX and run under onnxruntime.

Running the backbone through onnxruntime here, rather than through PyTorch,
means training features and browser features come out of the same graph. The
usual failure of this kind of project is a model that scores well offline and
poorly in the browser because the two preprocessing paths disagree; sharing
the graph removes the disagreement instead of testing for it later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

from training.features.preprocess_graph import (
    CLIP_SPEC,
    SIGLIP_SPEC,
    PreprocessSpec,
    build_preprocess_graph,
    center_square_crop,
)


@dataclass(frozen=True)
class Backbone:
    key: str
    repo_id: str
    filename: str
    spec: PreprocessSpec
    license_name: str


BACKBONES: tuple[Backbone, ...] = (
    Backbone(
        key="clip-vit-b32",
        repo_id="Xenova/clip-vit-base-patch32",
        filename="onnx/vision_model.onnx",
        spec=CLIP_SPEC,
        license_name="MIT",
    ),
    Backbone(
        key="siglip-base-p16",
        repo_id="Xenova/siglip-base-patch16-224",
        filename="onnx/vision_model.onnx",
        spec=SIGLIP_SPEC,
        license_name="Apache-2.0",
    ),
)


def backbone_by_key(key: str) -> Backbone:
    for backbone in BACKBONES:
        if backbone.key == key:
            return backbone
    raise KeyError(f"unknown backbone {key!r}")


def download_backbone(backbone: Backbone, models_dir: Path) -> Path:
    target_dir = models_dir / "backbones" / backbone.key
    target_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=backbone.repo_id,
            filename=backbone.filename,
            local_dir=str(target_dir),
        )
    )


class FeatureExtractor:
    """Runs preprocessing and the frozen backbone, returning one vector."""

    def __init__(self, backbone: Backbone, models_dir: Path) -> None:
        self.backbone = backbone
        model_path = download_backbone(backbone, models_dir)
        pre_path = build_preprocess_graph(
            backbone.spec, models_dir / "backbones" / backbone.key / "preprocess.onnx"
        )
        providers = ["CPUExecutionProvider"]
        self._pre = ort.InferenceSession(str(pre_path), providers=providers)
        self._net = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._net.get_inputs()[0].name
        self._output_name = self._pick_output()

    def _pick_output(self) -> str:
        """Prefer a pooled embedding over the token sequence."""
        names = [o.name for o in self._net.get_outputs()]
        for preferred in ("image_embeds", "pooler_output"):
            if preferred in names:
                return preferred
        return names[0]

    def embed(self, im: Image.Image) -> np.ndarray:
        im = im.convert("RGB")
        if self.backbone.spec.square_crop:
            im = center_square_crop(im)
        pixels = np.asarray(im, dtype=np.uint8)[None, ...]
        pixel_values = self._pre.run(None, {"pixels": pixels})[0]
        raw = self._net.run([self._output_name], {self._input_name: pixel_values})[0]
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 3:  # token sequence: mean-pool it
            return array.reshape(-1, array.shape[-1]).mean(axis=0).astype(np.float32)
        return array.reshape(-1).astype(np.float32)
