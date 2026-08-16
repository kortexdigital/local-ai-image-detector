import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from training.export.head_onnx import export_calibration, export_head
from training.head.calibrate import Calibration
from training.head.train import TrainedHead, linear_head, raw_scores

LINEAR = linear_head(np.array([0.5, -1.5, 2.0], dtype=np.float32), 0.25)

_W1 = np.array([[0.4, -0.2, 0.9, 0.1], [0.3, 0.7, -0.5, 0.2], [-0.8, 0.1, 0.6, -0.3]], dtype=np.float32)
_B1 = np.array([0.1, -0.2, 0.05, 0.0], dtype=np.float32)
_W2 = np.array([[1.1], [-0.6], [0.4], [0.9]], dtype=np.float32)
_B2 = np.array([-0.15], dtype=np.float32)
MLP = TrainedHead(members=(((_W1, _B1), (_W2, _B2)),), dim=3, kind="mlp", hyperparams={})

# Two members whose logits the exported graph must average.
ENSEMBLE = TrainedHead(
    members=(((_W1, _B1), (_W2, _B2)), ((_W1 * 0.5, _B1), (_W2 * -0.8, _B2 + 0.3))),
    dim=3,
    kind="mlp",
    hyperparams={},
)

CAL = Calibration(a=1.7, b=-0.4, t_star=0.35, decision_confidence=0.65)


def _run(path: Path, features: np.ndarray) -> float:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return float(sess.run(None, {"features": features})[0].reshape(-1)[0])


def test_exported_linear_graph_matches_the_python_scores(tmp_path: Path):
    path = export_head(LINEAR, tmp_path / "head.onnx")
    rng = np.random.default_rng(0)
    for _ in range(20):
        features = rng.normal(0, 1, (1, 3)).astype(np.float32)
        assert abs(_run(path, features) - float(raw_scores(LINEAR, features)[0])) < 1e-5


def test_exported_mlp_graph_matches_the_python_scores(tmp_path: Path):
    """The ReLU has to land between the layers in ONNX exactly as in numpy."""
    path = export_head(MLP, tmp_path / "mlp.onnx")
    rng = np.random.default_rng(1)
    for _ in range(20):
        features = rng.normal(0, 1, (1, 3)).astype(np.float32)
        assert abs(_run(path, features) - float(raw_scores(MLP, features)[0])) < 1e-5


def test_exported_mlp_reaches_negative_outputs(tmp_path: Path):
    """Guards against a stray ReLU on the output clamping every score to zero."""
    path = export_head(MLP, tmp_path / "mlp.onnx")
    rng = np.random.default_rng(2)
    values = [
        _run(path, rng.normal(0, 3, (1, 3)).astype(np.float32)) for _ in range(50)
    ]
    assert min(values) < 0.0


def test_exported_ensemble_averages_its_members(tmp_path: Path):
    """A five-seed ensemble ships as one graph; the average must be exact."""
    path = export_head(ENSEMBLE, tmp_path / "ens.onnx")
    rng = np.random.default_rng(3)
    for _ in range(15):
        features = rng.normal(0, 1, (1, 3)).astype(np.float32)
        assert abs(_run(path, features) - float(raw_scores(ENSEMBLE, features)[0])) < 1e-5


def test_calibration_records_the_ensemble_size(tmp_path: Path):
    path = export_calibration(CAL, ENSEMBLE, "clip-vit-b32-int8", tmp_path / "c.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["ensemble_members"] == 2
    assert payload["append_log_norm"] is True


def test_exported_graph_declares_the_expected_io(tmp_path: Path):
    path = export_head(LINEAR, tmp_path / "head.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["features"]
    assert [o.name for o in sess.get_outputs()] == ["score"]


def test_calibration_json_carries_everything_the_runtime_needs(tmp_path: Path):
    path = export_calibration(CAL, LINEAR, "clip-vit-b32", tmp_path / "calibration.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["a"] == CAL.a
    assert payload["b"] == CAL.b
    assert payload["t_star"] == CAL.t_star
    assert payload["decision_confidence"] == 0.65
    assert payload["dim"] == 3
    assert payload["backbone_key"] == "clip-vit-b32"
    assert payload["l2_normalize"] is True
    assert payload["head_kind"] == "linear"
