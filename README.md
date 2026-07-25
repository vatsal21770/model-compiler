# model-compiler

A small compile-and-quantize pipeline for ResNet-18, plus a hand-written CUDA
kernel. The pipeline takes a trained model, exports it to ONNX with a dynamic
batch dimension, folds BatchNorm into the preceding convolutions by hand,
quantizes to INT8 with post-training calibration, and reports FP32-vs-INT8
accuracy. The CUDA part is a fused bias+GELU kernel that runs on a free Colab
T4 and is checked against PyTorch.

Part 1 run on a plain CPU. Part 2 (CUDA) runs on Colab.

## What's here

| Part | Script | What it does |
|------|--------|--------------|
| Data | `prepare_data.py`  | Builds a disjoint calibration/validation subset from ImageNet-1k val |
| 1.0  | `eval_fp32_pt.py`  | FP32 PyTorch baseline (the accuracy anchor) |
| 1.1  | `export.py`        | ONNX export with dynamic batch + parity check vs PyTorch |
| 1.2  | `fold_conv_bn.py`  | Conv+BN folding done by hand on the ONNX graph |
| 1.3  | `quantize.py`      | INT8 PTQ + FP32-vs-INT8 accuracy table |
| 2    | `cuda/kernel.cu`, `cuda/kernel.ipynb` | Fused bias+GELU CUDA kernel, verified against PyTorch |
| 3    | `skill.md`         | Agent spec for the optimization pipeline |


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned torch/torchvision are CPU builds on purpose. Part 1 is a CPU
deliverable and CPU ONNX Runtime is deterministic, so the numbers below
reproduce without a GPU.


## Data

The accuracy work needs real ImageNet validation images with correct labels.
`prepare_data.py` streams the `ILSVRC/imagenet-1k` validation split from
Hugging Face (which is gated — you have to accept the terms on the dataset page
and `hf auth login` first) and writes two splits:

```bash
hf auth login
python prepare_data.py
```

```
data/
  calib/            300 images  -> INT8 calibration only
  val/             1000 images  -> accuracy measurement only
  calib_labels.json
  val_labels.json
  classes.json     1000 class names in canonical ILSVRC order
```

The two splits are disjoint by construction (the sampled records are sliced, not
re-drawn), so calibration never sees an image it will later be scored on.
The shuffle is seeded so the split is reproducible. Images are written as their
original JPEG bytes — no re-encode — so no lossy pass gets stacked underneath
the quantization measurements.

Result on my run:

```
calib: 300 images, 266 distinct classes
val:  1000 images, 648 distinct classes
```

`data/` and `artifacts/` are gitignored — regenerate `data/` with the script,
and the ONNX files with `export.py`.

## Part 1.0 — FP32 baseline

```bash
python eval_fp32_pt.py
```

This is the anchor. Every later number is compared against this, not against
the published 69.76%.

Two things happen here. First a class-order cross-check: ResNet-18 emits 1000
logits in canonical ILSVRC order, and the labels have to use the same order or
accuracy collapses to ~0.1%. The script confirms torchvision's class list and
the dataset's class list agree before it evaluates anything.

```
class-order cross-check: 997/1000 exact first-alias match (99.7%)
  idx 134  tv='crane bird'        hf='crane'
  idx 517  tv='crane'             hf='crane2'
  idx 639  tv='maillot tank suit' hf='maillot, tank suit'
```

The three mismatches are naming, not ordering — indices 134/517 are the
well-known ImageNet "crane" collision (the bird and the machine), which
torchvision and HF just spell differently. The ordering is identical.

The preprocessing is taken from `ResNet18_Weights.transforms()` rather than
hand-rolled, so it's exactly what the checkpoint was validated under (resize
256, center-crop 224, ImageNet mean/std).

Result:

| Model | n | top-1 | top-5 |
|-------|---|-------|-------|
| FP32 PyTorch (this subset)         | 1000  | 69.40% | 90.60% |
| torchvision published (full 50k)   | 50000 | 69.76% | 89.08% |

At n=1000 the standard error on top-1 is about ±1.5%, so the 0.36-point gap
from the published number is well inside noise. This matters later: **the subset
can detect a large INT8 regression but can't resolve a sub-1% one.** A measured
drop of 0.5% here is indistinguishable from noise; a drop of several points is
real. This is a deliberate tradeoff against runtime, not an oversight.

## Part 1.1 — ONNX export

```bash
python export.py
```

Exports two graphs and checks both against PyTorch:

| File | folding | opset | nodes | BN | Conv |
|------|---------|-------|-------|----|----|
| `artifacts/resnet18.onnx`         | on  | 13 | 49 | 0  | 20 |
| `artifacts/resnet18_nofold.onnx`  | off | 13 | 72 | 20 | 20 |

Parity vs PyTorch on random input (tolerance 1e-4):

| Graph | batch 1 | batch 4 | batch 8 |
|-------|---------|---------|---------|
| `resnet18.onnx`         | 3.9e-06 | 4.1e-06 | 5.7e-06 |
| `resnet18_nofold.onnx`  | 4.1e-06 | 3.8e-06 | 4.8e-06 |

Two graphs because the folded one (BN already fused by the exporter) is the
deployment artifact, and the unfolded one keeps its 20 BatchNorm nodes so Part
1.3 has something to fold by hand.

Parity is checked at three batch sizes, not one — that proves the dynamic batch
axis actually took effect rather than just being requested. Random inputs are
used because parity is about graph correctness, not accuracy, and random data
exercises the full range.

One thing worth flagging: this uses the legacy TorchScript exporter
(`dynamo=False`). The newer torch.export-based exporter runs a decomposition
pass that shreds BatchNorm and re-fuses it into Conv regardless of the
folding flag, and it silently fell back to opset 18 when opset-13 conversion
failed. Both would have broken the two-graph setup. The script asserts on opset
and BN counts so that kind of silent divergence fails loudly.

## Part 1.2 — Conv+BN folding (by hand)

```bash
python fold_conv_bn.py
```

Folds each BatchNorm into the convolution before it, directly on the ONNX graph.
No library that does folding for you. The math, per output channel:

```
s        = gamma / sqrt(running_var + eps)
W_folded = s * W
b_folded = s * (b - running_mean) + beta
```

Conv is linear and BN at inference is affine, so the two collapse into a single
Conv exactly — this is why the folded model is numerically equivalent, not just
close.

Result:

```
before: nodes=72  BN=20  Conv=20
found 20 Conv->BN pairs to fold
after:  nodes=49  BN=0  Conv=20
  batch=1 max_abs_diff=3.278e-06  PASS
  batch=4 max_abs_diff=3.338e-06  PASS
  batch=8 max_abs_diff=3.576e-06  PASS
hand-folded=49  torch-folded=49  delta=0
  op-type histograms identical to torch's folded graph
```

The equivalence is checked against the unfolded graph (the required assert), and
the node count is also compared against the graph torch folded on its own — they
come out identical, which is independent confirmation the hand fold is correct.

The fiddly part was that the exporter stores the same BN parameters three
different ways — as graph initializers, as Constant nodes, and behind Identity
passthroughs (the downsample branches). A naive fold silently skips whichever
form it can't read; the `BN == 0` assert is what caught the three that were
being skipped. The folder resolves all three forms and then sweeps the orphaned
nodes and initializers to a fixpoint so the saved graph loads clean.

## Part 1.3 — INT8 quantization

```bash
python quantize.py
```

Post-training INT8 quantization of the hand-folded graph. Calibration reads the
300-image calib split so ONNX Runtime can observe activation ranges; accuracy is
then measured on the disjoint 1000-image val split. Weights are per-channel,
activations per-tensor, static (QDQ format).

| model | top-1 | top-5 | size (MB) |
|-------|-------|-------|-----------|
| FP32  | 69.40% | 90.80% | 46.8 |
| INT8  | 68.20% | 88.40% | 11.8 |
| delta | -1.20pp | -2.40pp | 0.25x |

The FP32 number reproduces the anchor exactly, which confirms the whole path
(hand fold -> ONNX -> hand-written preprocessing) agrees with the original
torchvision model.

On accuracy: the top-1 drop of 1.2 points is inside the ±1.5-point noise floor
for n=1000, so I can't honestly claim it's distinguishable from zero — only that
it's clearly not a large regression. The top-5 drop of 2.4 points is more likely
real, and its shape (top-1 roughly flat, top-5 down more) is the fingerprint of
quantization jittering the ordering of close-together logits in the tail rather
than structural damage.

There's no accuracy gate in the assignment and the drop is within noise, so I
didn't chase it. If I needed to recover it: first widen the eval to the full 50k
val to see whether the drop is even real; if real, exclude the sensitive layers
(first conv, final fc) via `nodes_to_exclude`; beyond that, QAT. The calibration
set is only 300 images and uses min/max — a production run would use more images
and an entropy/percentile calibrator to be robust to outlier activations.

## Part 2 — CUDA kernel

A fused bias + GELU elementwise kernel (`cuda/kernel.cu`), built and run on a
free Colab T4. Open `cuda/kernel.ipynb` in Colab, set Runtime -> Change runtime
type -> T4 GPU, and run the cells top to bottom. It compiles with

```
nvcc -O3 -arch=sm_75 kernel.cu -o kernel
```

(`sm_75` is the T4's compute capability.) The kernel computes
`y[i] = GELU(x[i] + bias[i % C])` using the tanh approximation. It writes its
inputs and output to disk, and a Python cell reloads them and compares against
`torch.nn.functional.gelu(x + bias, approximate="tanh")`:

```
max_abs_err (GPU kernel vs PyTorch F.gelu) = 2.384e-07  ->  PASS
```

That's against a `1e-3` tolerance, so four orders of margin. It's not exactly
zero because the GPU computes in float and the reference in double — the same
floating-point-ordering effect as the parity checks in Part 1.

The point of the kernel is the fusion. The `x + bias` sum stays in a register
and is consumed by GELU in the same kernel, so it never round-trips through
global memory. Doing it as two kernels (add, then GELU) would write and re-read
the whole tensor for nothing. This is the same idea as TensorRT's
Conv+Bias+activation fusion, and it matters because an elementwise op is
memory-bound — the memory traffic is the runtime.

Notes on the kernel (also in the comments):

- **Coalescing:** thread `i` reads `x[i]` and writes `y[i]`, so a warp's 32
  threads hit 32 contiguous floats = one 128-byte transaction. Fully coalesced
  on the load of `x` and store of `y`. The `bias[i % C]` access isn't coalesced
  the same way, but bias is tiny and reused, so it caches immediately and isn't
  the bottleneck.
- **Grid/block sizing:** 256 threads per block (8 warps — a multiple of the
  warp size, and a conventional sweet spot), with `ceil(n / 256)` blocks. The
  `if (i < n)` guard handles the leftover threads in the final partial block.
- **Occupancy:** the kernel uses few registers and no shared memory, so
  occupancy is high — but for a memory-bound elementwise kernel occupancy isn't
  the limiter anyway, bandwidth is.
- **Next optimization:** vectorized `float4` loads, so each thread moves 16
  bytes instead of 4 and issues 4x fewer memory instructions. This is the
  highest-leverage change precisely because the kernel is bandwidth-bound.

## Part 3 — agent skill

`skill.md` — the instruction file that would let an agent run this pipeline on
its own: what it does, the tools it calls, how it picks precision, the
verification gate it has to pass before shipping, and what it does when a layer
is unsupported or accuracy regresses.

## Environment

- Ubuntu 22.04 (WSL2), Python 3.10
- torch 2.13.0+cpu, torchvision 0.28.0+cpu
- onnx 1.22.0, onnxruntime 1.23.2, ONNX opset 13
- CUDA: compiled on Colab with nvcc 12.8, target sm_75 (Tesla T4)