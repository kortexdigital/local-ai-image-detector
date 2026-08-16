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
    """Emit the head as ONNX.

    Each ensemble member becomes its own chain of dense layers with ReLU
    between them and none after, and the members' logits are averaged into a
    single output. Exporting the ensemble as one graph keeps the browser to a
    single session and a single call.
    """
    nodes = []
    initializers = []
    member_outputs = []

    for member_index, layers in enumerate(head.members):
        current = "features"
        last = len(layers) - 1
        for layer_index, (weights, bias) in enumerate(layers):
            w_name = f"W{member_index}_{layer_index}"
            b_name = f"B{member_index}_{layer_index}"
            initializers.append(
                numpy_helper.from_array(np.asarray(weights, dtype=np.float32), name=w_name)
            )
            initializers.append(
                numpy_helper.from_array(
                    np.asarray(bias, dtype=np.float32).reshape(-1), name=b_name
                )
            )
            matmul_out = f"mm{member_index}_{layer_index}"
            add_out = f"add{member_index}_{layer_index}"
            nodes.append(helper.make_node("MatMul", [current, w_name], [matmul_out]))
            nodes.append(helper.make_node("Add", [matmul_out, b_name], [add_out]))
            if layer_index < last:
                relu_out = f"relu{member_index}_{layer_index}"
                nodes.append(helper.make_node("Relu", [add_out], [relu_out]))
                current = relu_out
            else:
                member_outputs.append(add_out)

    if len(member_outputs) == 1:
        nodes.append(helper.make_node("Identity", member_outputs, ["score"]))
    else:
        nodes.append(helper.make_node("Mean", member_outputs, ["score"]))

    graph = helper.make_graph(
        nodes,
        "head",
        inputs=[
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, head.dim])
        ],
        outputs=[helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])],
        initializer=initializers,
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
        "append_log_norm": True,
        "head_kind": head.kind,
        "ensemble_members": len(head.members),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
