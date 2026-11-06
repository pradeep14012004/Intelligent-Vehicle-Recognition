import json
import time
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.deep_learning.classifier import VehicleClassifier
from src.fusion.hybrid import HybridVehiclePredictor


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TEST_LIMIT = None  # None = all 8,041 test images
TOP_K = 5

RESULT_DIR = Path("results/evaluation")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Evaluation device:", DEVICE)


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

print("\nLoading Stanford Cars test split...")

dataset = load_dataset(
    "bapei/stanford_cars",
    split="test",
)

if TEST_LIMIT is not None:
    dataset = dataset.select(
        range(min(TEST_LIMIT, len(dataset)))
    )

print("Test images:", len(dataset))


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

print("\nLoading EfficientNet backbone...")

base_classifier = VehicleClassifier(
    weights="models/pretrained/efficientnetv2_best_model.pth",
    mapping="data/metadata/class_mapping.json",
    device=DEVICE,
)

print("\nLoading trained hybrid fusion model...")

hybrid = HybridVehiclePredictor(
    backbone=base_classifier.model,
    transform=base_classifier.transform,
    classes=base_classifier.classes,
    weights="models/trained/hybrid_classifier.pth",
    device=DEVICE,
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_true = []
y_pred = []
top5_correct = 0

start_time = time.time()

print("\nStarting hybrid test evaluation...\n")

for i, sample in enumerate(dataset):

    image = sample["image"].convert("RGB")

    # RGB to BGR
    image_bgr = np.asarray(image)[:, :, ::-1].copy()

    true_label = int(sample["label"])

    predictions = hybrid.predict(
        image_bgr,
        top_k=TOP_K,
    )

    predicted_label = predictions[0]["class_id"]

    predicted_top5 = [
        item["class_id"] for item in predictions
    ]

    y_true.append(true_label)
    y_pred.append(predicted_label)

    if true_label in predicted_top5:
        top5_correct += 1

    if (i + 1) % 100 == 0:
        print(f"Evaluated {i + 1}/{len(dataset)}")


# --------------------------------------------------
# METRICS
# --------------------------------------------------

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

top1 = accuracy_score(y_true, y_pred)
top5 = top5_correct / len(y_true)

precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

elapsed = time.time() - start_time

print("\n========== HYBRID TEST RESULTS ==========")

print(f"Test images: {len(y_true)}")
print(f"Top-1 accuracy: {top1:.4%}")
print(f"Top-5 accuracy: {top5:.4%}")
print(f"Macro precision: {precision:.4%}")
print(f"Macro recall: {recall:.4%}")
print(f"Macro F1-score: {f1:.4%}")
print(f"Total inference time: {elapsed:.2f} seconds")
print(f"Average time/image: {elapsed / len(y_true):.4f} seconds")


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

metrics = {
    "model": "Hybrid EfficientNetV2 + Classical CV",
    "dataset": "Stanford Cars test split",
    "test_images": len(y_true),
    "top1_accuracy": float(top1),
    "top5_accuracy": float(top5),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "total_inference_seconds": elapsed,
    "average_inference_seconds_per_image": elapsed / len(y_true),
}

with open(RESULT_DIR / "hybrid_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

class_names = [
    base_classifier.classes.get(str(i), f"Class {i}")
    for i in range(196)
]

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(196)),
    target_names=class_names,
    zero_division=0,
)

with open(RESULT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(196)),
)

np.savetxt(
    RESULT_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)

print("\nResults saved in:", RESULT_DIR)
print("Evaluation completed.")
