from datasets import load_dataset
from pathlib import Path
import yaml

ROOT = Path("data/stanford_yolo").resolve()

for split in ["train", "val"]:
    (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

ds = load_dataset("nyris/stanford-cars-image-retrieval-v1")

def prepare(source_split, target_split):
    valid, skipped = 0, 0

    for i, item in enumerate(ds[source_split]):
        image = item["image"].convert("RGB")
        w, h = image.size
        x1, y1, x2, y2 = map(float, item["bbox"])

        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            skipped += 1
            continue

        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        name = f"{target_split}_{i:06d}"

        image.save(ROOT / "images" / target_split / f"{name}.jpg", quality=95)

        with open(ROOT / "labels" / target_split / f"{name}.txt", "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        valid += 1

    print(f"{target_split}: {valid} images prepared; {skipped} skipped")

prepare("train", "train")
prepare("test", "val")

config = {
    "path": str(ROOT),
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["vehicle"]
}

with open(ROOT / "data.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print("YOLO dataset ready:", ROOT)
