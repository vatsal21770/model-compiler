#FP32 PyTorch baseline on ImageNet val subset.

import json
import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

DATA = "data"
OUT = "artifacts/fp32_baseline.json"
BATCH = 32


class ImageNetSubset(Dataset):
    def __init__(self, split, transform):
        self.dir = os.path.join(DATA, split)
        with open(os.path.join(DATA, f"{split}_labels.json")) as f:
            labels = json.load(f)
        self.items = sorted(labels.items())      # deterministic order
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, label = self.items[i]
        img = Image.open(os.path.join(self.dir, fname)).convert("RGB")
        return self.transform(img), label


def check_class_order(weights):
    """Confirm HF label indices mean the same as torchvision output indices."""
    tv = weights.meta["categories"]
    with open(os.path.join(DATA, "classes.json")) as f:
        hf = json.load(f)
    assert len(tv) == len(hf) == 1000

    hits = sum(1 for t, h in zip(tv, hf) if t.strip().lower() == h.split(",")[0].strip().lower())
    pct = 100.0 * hits / 1000
    print(f"class-order cross-check: {hits}/1000 exact first-alias match ({pct:.1f}%)")
    assert pct > 90, "class ordering mismatch between HF labels and torchvision outputs"

    mism = [(i, tv[i], hf[i]) for i in range(1000)
            if tv[i].strip().lower() != hf[i].split(",")[0].strip().lower()][:5]
    for i, t, h in mism:
        print(f"  idx {i:3d}  tv='{t}'  hf='{h}'")


@torch.no_grad()
def evaluate(model, loader):
    top1 = top5 = n = 0
    for x, y in tqdm(loader, desc="fp32 eval"):
        logits = model(x)
        _, pred = logits.topk(5, dim=1)
        correct = pred.eq(y.view(-1, 1))
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += y.numel()
    return 100.0 * top1 / n, 100.0 * top5 / n, n


def main():
    torch.manual_seed(0)
    weights = ResNet18_Weights.IMAGENET1K_V1
    check_class_order(weights)

    model = torchvision.models.resnet18(weights=weights)
    model.eval()

    preprocess = weights.transforms()          # resize 256 -> crop 224 -> normalize
    print(f"transform: {preprocess}")

    ds = ImageNetSubset("val", preprocess)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4)

    t1, t5, n = evaluate(model, loader)
    print(f"\nFP32 PyTorch  |  n={n}  top-1={t1:.2f}%  top-5={t5:.2f}%")
    print("reference: 69.76% top-1 / 89.08% top-5 on full 50k val")

    os.makedirs("artifacts", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"n": n, "top1": t1, "top5": t5, "batch": BATCH}, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
