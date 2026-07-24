"""Fold BatchNorm into the preceding Conv, directly on the ONNX graph.

Math (per output channel c):
    s        = gamma / sqrt(var + eps)
    W_folded = s * W          (broadcast over axis 0)
    b_folded = s * (b - mu) + beta

Conv is linear and BN at inference is affine, so the composition is a single Conv.
"""

import json
import os
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ART = "artifacts"
SRC = os.path.join(ART, "resnet18_nofold.onnx")
DST = os.path.join(ART, "resnet18_folded.onnx")
REF = os.path.join(ART, "resnet18.onnx")      # torch's own folded graph, for comparison
TOL = 1e-4


def count_ops(graph):
    return Counter(n.op_type for n in graph.node)


def build_indices(graph):
    """producer: tensor name -> node emitting it.  consumers: tensor name -> use count."""
    producer, consumers = {}, Counter()
    for node in graph.node:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            consumers[inp] += 1
    return producer, consumers


def build_constants(graph):
    """Tensor name -> numpy array. Resolves initializers, Constant nodes, and
    Identity passthroughs."""

    const = {}
    for init in graph.initializer:
        const[init.name] = numpy_helper.to_array(init)

    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const[node.output[0]] = numpy_helper.to_array(attr.t)

    for _ in range(len(graph.node)):
        changed = False
        for node in graph.node:
            if node.op_type != "Identity":
                continue
            out, src = node.output[0], node.input[0]
            if out not in const and src in const:
                const[out] = const[src]
                changed = True
        if not changed:
            break

    return const


def get_eps(bn_node):
    for attr in bn_node.attribute:
        if attr.name == "epsilon":
            return attr.f
    return 1e-5          # ONNX default


def find_pairs(graph, const):
    producer, consumers = build_indices(graph)
    pairs, skipped = [], []
    for node in graph.node:
        if node.op_type != "BatchNormalization":
            continue
        conv = producer.get(node.input[0])

        if conv is None:
            skipped.append((node.name, f"input[0]={node.input[0]} has no producer node"))
            continue
        if conv.op_type != "Conv":
            skipped.append((node.name, f"producer is {conv.op_type}, not Conv"))
            continue
        if consumers[conv.output[0]] != 1:
            skipped.append((node.name, f"conv output has {consumers[conv.output[0]]} "
                                       f"consumers, not 1"))
            continue
        missing = [n for n in node.input[1:5] if n not in const]
        if missing:
            skipped.append((node.name, f"BN params not constant: {missing}"))
            continue

        pairs.append((conv, node))

    if skipped:
        print(f"skipped {len(skipped)} BN nodes:")
        for name, why in skipped:
            print(f"  {name or '<unnamed>'}: {why}")
    return pairs


def sweep(graph):
    """Remove nodes and initializers nothing references any more."""

    n_const = n_ident = 0
    while True:
        used = {name for node in graph.node for name in node.input}
        used |= {o.name for o in graph.output}
        dead = [n for n in graph.node
                if n.op_type in ("Constant", "Identity") and n.output[0] not in used]
        if not dead:
            break
        for n in dead:
            if n.op_type == "Constant":
                n_const += 1
            else:
                n_ident += 1
            graph.node.remove(n)

    used = {name for node in graph.node for name in node.input}
    used |= {o.name for o in graph.output}
    dead_init = [i for i in graph.initializer if i.name not in used]
    for i in dead_init:
        graph.initializer.remove(i)

    return len(dead_init), n_const, n_ident


def fold(model):
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    const = build_constants(graph)

    pairs = find_pairs(graph, const)
    print(f"found {len(pairs)} Conv->BN pairs to fold")

    removed = []
    for conv, bn in pairs:
        W = const[conv.input[1]].astype(np.float64)
        if len(conv.input) > 2 and conv.input[2] in const:
            b = const[conv.input[2]].astype(np.float64)
        else:
            b = np.zeros(W.shape[0], dtype=np.float64)   

        gamma = const[bn.input[1]].astype(np.float64)
        beta  = const[bn.input[2]].astype(np.float64)
        mu    = const[bn.input[3]].astype(np.float64)
        var   = const[bn.input[4]].astype(np.float64)
        eps = get_eps(bn)

        s = gamma / np.sqrt(var + eps)
        W_fold = (W * s.reshape(-1, 1, 1, 1)).astype(np.float32)
        b_fold = (s * (b - mu) + beta).astype(np.float32)

        inits[conv.input[1]].CopyFrom(numpy_helper.from_array(W_fold, conv.input[1]))

        bias_name = (conv.name or conv.output[0]) + "_folded_bias"
        graph.initializer.append(numpy_helper.from_array(b_fold, bias_name))
        if len(conv.input) > 2:
            conv.input[2] = bias_name
        else:
            conv.input.append(bias_name)

        conv.output[0] = bn.output[0]
        removed.append(bn)

    for bn in removed:
        graph.node.remove(bn)

    dead_init, dead_const, dead_ident = sweep(graph)
    print(f"removed {len(removed)} BN nodes, {dead_init} orphaned initializers, "
          f"{dead_const} orphaned Constant nodes, {dead_ident} orphaned Identity nodes")

    return model


def equivalence(path_a, path_b, batch, tol=TOL):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch, 3, 224, 224)).astype(np.float32)

    outs = []
    for p in (path_a, path_b):
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        outs.append(sess.run(["logits"], {"input": x})[0])

    diff = float(np.abs(outs[0] - outs[1]).max())
    status = "PASS" if diff < tol else "FAIL"
    print(f"  batch={batch:<3} max_abs_diff={diff:.3e}  {status}")
    assert diff < tol, f"folded model diverges at batch={batch}: {diff}"
    return diff


def main():
    src = onnx.load(SRC)
    before = count_ops(src.graph)
    print(f"before: nodes={len(src.graph.node)}  BN={before['BatchNormalization']}  "
          f"Conv={before['Conv']}")

    folded = fold(src)
    onnx.checker.check_model(folded)
    onnx.save(folded, DST)

    after = count_ops(folded.graph)
    print(f"after:  nodes={len(folded.graph.node)}  BN={after['BatchNormalization']}  "
          f"Conv={after['Conv']}")
    assert after["BatchNormalization"] == 0, "BN nodes remain after folding"
    assert after["Conv"] == before["Conv"], "Conv count changed"

    print("-- numerical equivalence: nofold vs hand-folded --")
    diffs = {str(b): equivalence(SRC, DST, b) for b in (1, 4, 8)}

    # Compare against the graph torch folded for us.
    ref = onnx.load(REF)
    ref_ops = count_ops(ref.graph)
    print(f"\nhand-folded={len(folded.graph.node)}  torch-folded={len(ref.graph.node)}  "
          f"delta={len(folded.graph.node) - len(ref.graph.node)}")
    extra = after - ref_ops
    missing = ref_ops - after
    if extra or missing:
        print(f"  ops we still have that torch removed: {dict(extra)}")
        print(f"  ops torch has that we do not:         {dict(missing)}")
    else:
        print("  op-type histograms identical to torch's folded graph")

    with open(os.path.join(ART, "fold_report.json"), "w") as f:
        json.dump({
            "bn_before": before["BatchNormalization"],
            "bn_after": after["BatchNormalization"],
            "nodes_before": len(src.graph.node),
            "nodes_hand_folded": len(folded.graph.node),
            "nodes_torch_folded": len(ref.graph.node),
            "residual_ops_vs_torch": dict(extra),
            "tol": TOL,
            "max_abs_diff": diffs,
        }, f, indent=2)
    print("fold verified")


if __name__ == "__main__":
    main()
