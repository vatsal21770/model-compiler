---
name: compile-quantize-resnet
description: >
  Compiles a trained CNN classifier (ResNet-family) into an optimized,
  hardware-ready ONNX/engine artifact. Takes a model checkpoint, exports to
  ONNX with a dynamic batch dimension, folds Conv+BatchNorm, chooses a
  precision, quantizes, and verifies accuracy against a gate before it emits
  anything. Use when someone hands you a trained image classifier and wants a
  smaller/faster deployable artifact with a documented accuracy trade-off.
  Do NOT use for models that still need training, for non-classification heads
  without adapting the accuracy metric, or when no labelled validation data is
  available (the verification gate cannot run).
---

# Compile & Quantize (ResNet-family)

## What this agent does

Given a trained classifier, it produces a deployable artifact plus an honest
accuracy report. It never ships an artifact that hasn't passed the verification
gate. If it cannot pass the gate at any precision, it reports failure rather
than shipping a degraded model silently.

**Input:** a path to a trained checkpoint (`.pth`) or an existing `.onnx`, plus
a path to a small labelled validation set and a disjoint calibration set.

**Output:** an optimized ONNX model (`resnet_optimized.onnx`) and a JSON report:

```json
{
  "chosen_precision": "int8" | "fp16" | "fp32",
  "fp32_acc": 0.694,
  "int8_acc": 0.682,
  "passed": true,
  "notes": "..."
}
```

## Preconditions the agent checks first

1. A labelled validation set exists and is disjoint from the calibration set.
   If they overlap or validation labels are missing, STOP — the gate is
   meaningless without it. Report `passed: false`, note the reason.
2. The calibration set is representative (covers a spread of classes, not one).
3. The model's class ordering matches the validation labels' ordering. Run the
   class-order cross-check before trusting any accuracy number; a mismatch makes
   every downstream number wrong.

## Tools / commands it calls

| Stage | Command / API | Success signal |
|-------|---------------|----------------|
| FP32 baseline | `eval_fp32_pt.py` (torch eval) | top-1 lands near published accuracy; if far below, preprocessing or label mapping is wrong — stop |
| Export | `export.py` → `torch.onnx.export` (legacy exporter, `dynamo=False`) | parity vs PyTorch `max_abs_diff < 1e-4` at multiple batch sizes; opset and BN-count asserts pass |
| Graph surgery | `fold_conv_bn.py` (hand fold on ONNX graph) | `BN == 0` after fold; equivalence vs unfolded `< 1e-4` |
| Quantize | `quantize.py` → `onnxruntime.quantization.quantize_static`, QDQ, per-channel weights | quantized model builds; accuracy measured on disjoint val |
| Kernel (optional) | `nvcc -O3 -arch=sm_XX kernel.cu` then run | kernel output vs PyTorch reference `max_abs_err < 1e-3` |

The agent always runs the FP32 baseline first. Every later accuracy number is
compared against this baseline, not against a published figure, because the
validation subset introduces sampling noise.

## Decision loop — choosing precision

The agent does not default to the most aggressive precision. It climbs down only
as far as the gate allows.

```
1. Establish fp32 baseline accuracy on the disjoint val set.  -> fp32_acc
2. Compute the noise floor for this val set:
     noise ≈ 1.96 * sqrt(p(1-p)/n)     # ~±1.5% at n=1000, p≈0.7
   Any accuracy drop smaller than this is NOT considered real.
3. Try INT8 (post-training, per-channel weights, per-tensor activations,
   calibrated on the calib set).       -> int8_acc
     - If (fp32_acc - int8_acc) <= max_drop  -> choose INT8. Done.
     - Else enter the repair loop (below) for INT8.
4. If INT8 cannot pass after repair, fall back to FP16.
     - FP16 needs no calibration (it stays floating point), so its accuracy
       loss is almost always within noise. It should pass unless the model is
       pathologically sensitive.
5. If even FP16 fails the gate (rare), ship FP32 and report that no reduced
   precision met the bar.
```

Precision facts the loop relies on:

- **INT8 needs calibration; FP16 does not.** INT8 maps activations to 256
  integer levels, and activation ranges are input-dependent — they must be
  observed on real data first. FP16 keeps the float format, so there is no range
  to observe.
- **Per-channel weights, per-tensor activations.** Conv filter magnitudes vary
  widely, so a per-filter scale is worth it. Activation channel axes move
  through the network, so per-channel activation scales aren't practical on
  hardware.

## Which layers to skip during quantization

When INT8 regresses past the gate, the agent does not give up on INT8 wholesale.
It selectively excludes the layers most sensitive to quantization, in this
priority order, re-measuring after each:

1. The **first convolution** — it sees raw input pixels with a wide, un-normalized
   dynamic range.
2. The **final classifier (fc)** — it produces the logits directly, so its
   quantization error lands straight on the decision.
3. Any layer flagged by a per-layer sensitivity sweep (quantize one layer at a
   time, measure drop) if the two above aren't enough.

Excluding a layer keeps it in FP16/FP32 while the rest stays INT8 — a partial
quantization that trades a little size back for accuracy. This is done via
`nodes_to_exclude` in `quantize_static`.

## Verification gate — must pass before shipping

The gate is a hard threshold on accuracy drop, evaluated on the disjoint
validation set:

```
GATE:  (fp32_acc - chosen_acc) <= max_drop
       AND chosen_acc measured on val disjoint from calibration
DEFAULT max_drop = 1.0%   (tunable; set per deployment tolerance)
```

Two honesty rules baked into the gate:

- **Noise awareness.** If the measured drop is smaller than the val set's noise
  floor, the agent records that the drop is within noise and not resolvable at
  this sample size — it does not claim a precise improvement it cannot measure.
- **No shipping on an unverified graph.** After Conv+BN folding, the agent
  asserts `BN == 0` and equivalence to the unfolded graph. A graph that fails
  these never reaches quantization. (This exists because BN parameters can be
  stored in several forms in the exported graph, and a naive fold silently skips
  the ones it can't read — the assert is what catches it.)

If the gate cannot be met at any precision, `passed: false` and the agent ships
FP32 with a note, rather than shipping a model that quietly lost accuracy.

## Repair loop

Two failure classes, two responses:

**A. Unsupported / unfoldable layer during export or quantization.**
1. Identify the offending node from the error (op type, name).
2. If it's a Conv+BN fold that skipped: check whether the BN params are stored
   as initializers, Constant nodes, or Identity passthroughs, and resolve all
   forms before folding. Re-assert `BN == 0`.
3. If it's an op the quantizer/exporter can't handle: exclude that node from
   quantization (keep it FP16/FP32) and continue, OR — if targeting a real
   engine — supply it as a custom kernel / plugin. Note the exclusion in the
   report.
4. Re-run the gate.

**B. Accuracy regressed past the gate.**
1. Confirm the drop is real (larger than the noise floor). If it's within noise,
   it's not a regression — accept and record as within-noise.
2. If real, exclude the sensitive layers in the priority order above, one at a
   time, re-measuring after each.
3. If INT8 still fails after excluding first-conv and final-fc, fall back to
   FP16.
4. If FP16 fails, ship FP32 and report `passed: false` with the observed drops.

The loop is bounded: at most one sensitivity sweep, then fall back a precision
level. It does not loop indefinitely trying to rescue INT8.

## Report it emits

```json
{
  "chosen_precision": "int8",
  "fp32_acc": 0.694,
  "int8_acc": 0.682,
  "passed": true,
  "notes": "top-1 drop 1.2pp, within the ~1.5pp noise floor at n=1000; per-channel weights; calibrated on 300 disjoint images; no layers excluded. Drop not resolvable to better than a couple of points at this sample size."
}
```

`notes` always records: the measured drop, whether it's within noise, the
calibration set size, any excluded layers, and any op that needed a custom
kernel or exclusion. The report is the deliverable — a reviewer should be able
to judge the trade-off from it without rerunning anything.

## When to hand off to a human

- No labelled validation data → the gate can't run. Stop and ask.
- Even FP16 fails the gate → the model is unusually sensitive; flag for QAT
  (quantization-aware training), which is outside this agent's scope.
- An unsupported op needs a real TensorRT plugin for the target engine → surface
  it; writing and validating a plugin is a human-in-the-loop decision, worth it
  only when the op is on the critical path and profiling confirms it's a
  bottleneck.
