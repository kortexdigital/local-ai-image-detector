"""Write the runtime artifacts the browser extension consumes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from training.head.calibrate import Calibration
from training.head.train import TrainedHead

OPSET = 17


def export_head(head: TrainedHead, path: Path) -> Path:
    weights = head.weights.astype(np.float32).reshape(head.dim, 1)
    bias = np.array([head.bias], dtype=np.float32)

    nodes = [
        helper.make_node("MatMul", ["features", "W"], ["projected"]),
        helper.make_node("Add", ["projected", "B"], ["score"]),
    ]
    graph = helper.make_graph(
        nodes,
        "head",
        inputs=[
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, head.dim])
        ],
        outputs=[helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])],
        initializer=[
            numpy_helper.from_array(weights, name="W"),
            numpy_helper.from_array(bias, name="B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def export_calibration(
    cal: Calibration, head: TrainedHead, backbone_key: str, path: Path
) -> Path:
    payload = {
        "a": cal.a,
        "b": cal.b,
        "t_star": cal.t_star,
        "decision_confidence": cal.decision_confidence,
        "dim": head.dim,
        "backbone_key": backbone_key,
        "l2_normalize": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
