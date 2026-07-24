"""Export ResNet-18 to ONNX (dynamic batch), verify parity vs PyTorch."""
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from torchvision.models import ResNet18_Weights

ART = "artifacts"
TOL = 1e-4
OPSET = 13


def build_model():
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()          # BN -> running stats, deterministic
    return model


def export(model, path, fold):
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        do_constant_folding=fold,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        # Legacy TorchScript tracer. The dynamo exporter runs a decomposition
        # pass before ONNX translation, which shreds BatchNorm and re-fuses it
        # into Conv regardless of do_constant_folding, and it silently falls
        # back to opset 18 when the requested opset conversion fails.
        dynamo=False,
    )
    m = onnx.load(path)
    onnx.checker.check_model(m)

    actual_opset = m.opset_import[0].version
    assert actual_opset == OPSET, f"opset {actual_opset} != requested {OPSET}"

    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"{path}  opset={actual_opset}  nodes={len(m.graph.node)}  "
          f"BN={ops.get('BatchNormalization', 0)}  Conv={ops.get('Conv', 0)}")
    return ops


def parity(model, path, batch):
    x = torch.randn(batch, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x.numpy()})[0]

    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    diff = float(np.abs(ref - got).max())
    status = "PASS" if diff < TOL else "FAIL"
    print(f"  batch={batch:<3} max_abs_diff={diff:.3e}  {status}")
    assert diff < TOL, f"parity fail at batch={batch}: {diff}"
    return diff


def main():
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    model = build_model()

    folded = os.path.join(ART, "resnet18.onnx")
    nofold = os.path.join(ART, "resnet18_nofold.onnx")

    print("-- export --")
    ops_f = export(model, folded, fold=True)
    ops_n = export(model, nofold, fold=False)

    # Guard the two-graph contract: quantize.py needs the folded graph,
    # fold_conv_bn.py needs BN nodes to actually be present.
    assert ops_f.get("BatchNormalization", 0) == 0, \
        "folded graph still has BN nodes -- constant folding did not run"
    assert ops_n.get("BatchNormalization", 0) > 0, \
        "nofold graph has no BN nodes -- nothing left for fold_conv_bn.py to fold"

    report = {
        "tol": TOL,
        "opset": OPSET,
        "nodes_folded": sum(ops_f.values()),
        "nodes_nofold": sum(ops_n.values()),
        "bn_folded": ops_f.get("BatchNormalization", 0),
        "bn_nofold": ops_n.get("BatchNormalization", 0),
        "parity": {},
    }

    for path in (folded, nofold):
        print(f"-- parity: {os.path.basename(path)} --")
        report["parity"][os.path.basename(path)] = {
            str(b): parity(model, path, b) for b in (1, 4, 8)
        }

    with open(os.path.join(ART, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nall parity checks passed")


if __name__ == "__main__":
    main()
