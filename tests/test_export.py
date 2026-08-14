import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from training.export.head_onnx import export_calibration, export_head
from training.head.calibrate import Calibration
from training.head.train import TrainedHead, raw_scores

HEAD = TrainedHead(
    weights=np.array([0.5, -1.5, 2.0], dtype=np.float32), bias=0.25, dim=3, C=1.0
)
CAL = Calibration(a=1.7, b=-0.4, t_star=0.35, decision_confidence=0.65)


def test_exported_graph_matches_the_python_scores(tmp_path: Path):
    path = export_head(HEAD, tmp_path / "head.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    for _ in range(20):
        features = rng.normal(0, 1, (1, 3)).astype(np.float32)
        onnx_score = sess.run(None, {"features": features})[0].reshape(-1)[0]
        python_score = raw_scores(HEAD, features)[0]
        assert abs(float(onnx_score) - float(python_score)) < 1e-5


def test_exported_graph_declares_the_expected_io(tmp_path: Path):
    path = export_head(HEAD, tmp_path / "head.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["features"]
    assert [o.name for o in sess.get_outputs()] == ["score"]


def test_calibration_json_carries_everything_the_runtime_needs(tmp_path: Path):
    path = export_calibration(CAL, HEAD, "clip-vit-b32", tmp_path / "calibration.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["a"] == CAL.a
    assert payload["b"] == CAL.b
    assert payload["t_star"] == CAL.t_star
    assert payload["decision_confidence"] == 0.65
    assert payload["dim"] == 3
    assert payload["backbone_key"] == "clip-vit-b32"
    assert payload["l2_normalize"] is True
