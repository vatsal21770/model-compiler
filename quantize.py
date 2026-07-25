"""INT8 post-training quantization (PTQ) of the folded ResNet-18 ONNX graph.

Calibration reads the disjoint calib/ split; accuracy is measured on val/.
Per-channel weights, static activation quantization (QDQ).
"""
import json
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

ART = "artifacts"
DATA = "data"
FP32 = os.path.join(ART, "resnet18_folded.onnx")     # our hand-folded graph
PREP = os.path.join(ART, "resnet18_folded_prep.onnx")
INT8 = os.path.join(ART, "resnet18_int8.onnx")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path):
    """Match ResNet18_Weights transform: resize 256 -> center crop 224 -> normalize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)                          # HWC -> CHW
    return x[np.newaxis, :]                           # add batch


def load_split(split):
    with open(os.path.join(DATA, f"{split}_labels.json")) as f:
        labels = json.load(f)
    return sorted(labels.items())


class CalibReader(CalibrationDataReader):
    """Feeds calib images one at a time so ORT can observe activation ranges."""
    def __init__(self, items, input_name):
        self.items = items
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        fname, _ = self.items[self.i]
        self.i += 1
        x = preprocess(os.path.join(DATA, "calib", fname))
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def evaluate(model_path, items, split_dir):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    top1 = top5 = 0
    for fname, label in items:
        x = preprocess(os.path.join(split_dir, fname))
        logits = sess.run(None, {name: x})[0][0]
        top5_idx = logits.argsort()[-5:][::-1]
        if top5_idx[0] == label:
            top1 += 1
        if label in top5_idx:
            top5 += 1
    n = len(items)
    return 100.0 * top1 / n, 100.0 * top5 / n, n


def main():
    # 1. Preprocess graph for quantization (shape inference + cleanup).
    quant_pre_process(FP32, PREP, skip_symbolic_shape=False)

    # 2. Determine input name.
    sess = ort.InferenceSession(PREP, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 3. Static INT8 quantization on the disjoint calib split.
    calib_items = load_split("calib")
    reader = CalibReader(calib_items, input_name)
    quantize_static(
        PREP,
        INT8,
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,                 # per-channel conv weights
        weight_type=QuantType.QInt8,      # signed INT8 weights
        activation_type=QuantType.QUInt8, # unsigned activations (>=0 after ReLU)
    )
    print(f"quantized -> {INT8}")
    print(f"calibrated on {len(calib_items)} images (disjoint from val)")

    # 4. Accuracy: FP32 vs INT8 on the val split.
    val_items = load_split("val")
    val_dir = os.path.join(DATA, "val")

    fp32_t1, fp32_t5, n = evaluate(FP32, val_items, val_dir)
    int8_t1, int8_t5, _ = evaluate(INT8, val_items, val_dir)

    size_fp32 = os.path.getsize(FP32) / 1e6
    size_int8 = os.path.getsize(INT8) / 1e6

    print("\n" + "=" * 52)
    print(f"{'model':<10}{'top-1':>10}{'top-5':>10}{'size MB':>12}")
    print("-" * 52)
    print(f"{'FP32':<10}{fp32_t1:>9.2f}%{fp32_t5:>9.2f}%{size_fp32:>11.1f}")
    print(f"{'INT8':<10}{int8_t1:>9.2f}%{int8_t5:>9.2f}%{size_int8:>11.1f}")
    print("-" * 52)
    print(f"{'delta':<10}{int8_t1 - fp32_t1:>9.2f}pp{int8_t5 - fp32_t5:>8.2f}pp"
          f"{size_int8 / size_fp32:>10.2f}x")
    print("=" * 52)

    with open(os.path.join(ART, "quant_report.json"), "w") as f:
        json.dump({
            "n_val": n, "n_calib": len(calib_items),
            "fp32": {"top1": fp32_t1, "top5": fp32_t5, "size_mb": size_fp32},
            "int8": {"top1": int8_t1, "top5": int8_t5, "size_mb": size_int8},
            "top1_drop_pp": fp32_t1 - int8_t1,
            "per_channel": True, "quant_format": "QDQ",
        }, f, indent=2)


if __name__ == "__main__":
    main()
