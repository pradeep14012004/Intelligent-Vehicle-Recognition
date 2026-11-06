import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score

from src.deep_learning.classifier import VehicleClassifier
from src.fusion.hybrid import HybridVehiclePredictor


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LIMIT = 500  # Quick optimization benchmark

print("Device:", DEVICE)

dataset = load_dataset(
    "bapei/stanford_cars",
    split="test",
).select(range(LIMIT))

classifier = VehicleClassifier(
    weights="models/pretrained/efficientnetv2_best_model.pth",
    mapping="data/metadata/class_mapping.json",
    device=DEVICE,
)

hybrid = HybridVehiclePredictor(
    backbone=classifier.model,
    transform=classifier.transform,
    classes=classifier.classes,
    weights="models/trained/hybrid_classifier.pth",
    device=DEVICE,
)


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


results = []

for sample in dataset:
    image = sample["image"].convert("RGB")
    image_bgr = np.asarray(image)[:, :, ::-1].copy()
    true_label = int(sample["label"])

    sync()
    start = time.perf_counter()
    predictions = hybrid.predict(image_bgr, top_k=5)
    sync()
    latency_ms = (time.perf_counter() - start) * 1000.0

    top_pred = predictions[0]["class_id"]
    results.append({
        "true_label": true_label,
        "predicted_label": top_pred,
        "latency_ms": latency_ms,
    })

labels = np.asarray([row["true_label"] for row in results])
preds = np.asarray([row["predicted_label"] for row in results])
latencies = np.asarray([row["latency_ms"] for row in results])

accuracy = accuracy_score(labels, preds)
print(f"Benchmark samples: {len(results)}")
print(f"Top-1 accuracy: {accuracy:.4%}")
print(f"Average latency: {latencies.mean():.2f} ms")
print(f"Median latency: {np.median(latencies):.2f} ms")
print(f"P95 latency: {np.percentile(latencies, 95):.2f} ms")
