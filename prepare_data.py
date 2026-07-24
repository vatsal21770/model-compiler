"""Build disjoint calib/val splits from ImageNet-1k validation. Seeded, reproducible."""
import gc
import json
import os

from datasets import Image as HFImage
from datasets import load_dataset

REPO = "ILSVRC/imagenet-1k"
SEED = 42
N_CALIB = 300
N_VAL = 1000
OUT = "data"


def dump(split_name, records, names):
    d = os.path.join(OUT, split_name)
    os.makedirs(d, exist_ok=True)
    labels = {}
    for i, rec in enumerate(records):
        fname = f"{i:05d}.jpg"
        with open(os.path.join(d, fname), "wb") as f:
            f.write(rec["image"]["bytes"])       # raw source bytes, no re-encode
        labels[fname] = int(rec["label"])
    with open(os.path.join(OUT, f"{split_name}_labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print(f"{split_name}: {len(labels)} images, {len(set(labels.values()))} distinct classes")
    return labels


def main():
    ds = load_dataset(REPO, split="validation", streaming=True)
    names = ds.features["label"].names               # canonical ILSVRC order
    assert len(names) == 1000, f"expected 1000 classes, got {len(names)}"

    ds = ds.cast_column("image", HFImage(decode=False))
    ds = ds.shuffle(seed=SEED, buffer_size=2000)
    records = list(ds.take(N_CALIB + N_VAL))
    assert len(records) == N_CALIB + N_VAL, f"got {len(records)}"

    dump("calib", records[:N_CALIB], names)
    val_labels = dump("val", records[N_CALIB:], names)

    with open(os.path.join(OUT, "classes.json"), "w") as f:
        json.dump(names, f, indent=2)

    print("\n-- sanity decode: open these jpgs, check the name matches --")
    for fname in list(val_labels)[:5]:
        idx = val_labels[fname]
        print(f"  data/val/{fname}  ->  {idx:3d}  {names[idx]}")

    del records, ds
    gc.collect()


if __name__ == "__main__":
    main()
    os._exit(0)    # skip interpreter finalization; HF streaming threads crash on GIL teardown
