import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.deep_learning.classifier import VehicleClassifier
from src.fusion.hybrid import HybridVehiclePredictor

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
CNN_PATH = ROOT / "models/pretrained/efficientnetv2_best_model.pth"
HYBRID_PATH = ROOT / "models/trained/hybrid_classifier.pth"
MAPPING_PATH = ROOT / "data/metadata/class_mapping.json"
OUTPUT_DIR = ROOT / "results/final_analysis"
FAILURE_DIR = OUTPUT_DIR / "failure_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILURE_DIR.mkdir(parents=True, exist_ok=True)

TEST_LIMIT = None  # None = all 8041 images
TOP_K = 5

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def synchronize():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def main():
    print("Loading Stanford Cars dataset...")

    dataset = load_dataset(
        "bapei/stanford_cars",
        split="test",
    )

    if TEST_LIMIT is not None:
        dataset = dataset.select(range(min(TEST_LIMIT, len(dataset))))

    print("Dataset size:", len(dataset))

    # --------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------

    base_classifier = VehicleClassifier(
        weights=str(CNN_PATH),
        mapping=str(MAPPING_PATH),
        device=DEVICE,
    )

    hybrid = HybridVehiclePredictor(
        backbone=base_classifier.model,
        transform=base_classifier.transform,
        classes=base_classifier.classes,
        weights=str(HYBRID_PATH),
        device=DEVICE,
    )

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    y_true = []
    y_pred = []
    y_pred_top5 = []
    latencies = []
    failures = []

    start_time = time.time()

    for idx, sample in enumerate(dataset):
        image = sample["image"].convert("RGB")
        image_bgr = np.asarray(image)[:, :, ::-1].copy()
        true_label = int(sample["label"])

        synchronize()
        pred_start = time.perf_counter()
        predictions = hybrid.predict(image_bgr, top_k=TOP_K)
        synchronize()
        elapsed_ms = (time.perf_counter() - pred_start) * 1000.0
        latencies.append(elapsed_ms)

        predicted_label = int(predictions[0]["class_id"])
        top5_labels = [int(item["class_id"]) for item in predictions]

        y_true.append(true_label)
        y_pred.append(predicted_label)
        y_pred_top5.append(top5_labels)

        if true_label not in top5_labels:
            failures.append({
                "sample_index": idx,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_top5": top5_labels,
                "true_label_name": base_classifier.classes.get(str(true_label), str(true_label)),
                "predicted_label_name": base_classifier.classes.get(str(predicted_label), str(predicted_label)),
            })

        if (idx + 1) % 100 == 0:
            print(f"Evaluated {idx + 1}/{len(dataset)}")

    elapsed = time.time() - start_time

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    top1_accuracy = accuracy_score(y_true, y_pred)
    top5_correct = np.asarray([
        true in pred_list
        for true, pred_list in zip(y_true, y_pred_top5)
    ], dtype=np.int64)
    top5_accuracy = float(top5_correct.mean())

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    average_latency_ms = float(np.mean(latencies))
    median_latency_ms = float(np.median(latencies))

    model_size_bytes = 0
    for parameter in hybrid.model.parameters():
        model_size_bytes += parameter.numel() * parameter.element_size()
    model_size_mb = model_size_bytes / (1024 ** 2)

    metrics = {
        "dataset": "Stanford Cars test split",
        "device": DEVICE,
        "samples": int(len(y_true)),
        "top1_accuracy": float(top1_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "total_inference_seconds": float(elapsed),
        "average_latency_ms": average_latency_ms,
        "median_latency_ms": median_latency_ms,
        "model_size_mb": float(model_size_mb),
    }

    with open(OUTPUT_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "final_metrics.csv",
        index=False,
    )

    if failures:
        pd.DataFrame(failures[:3]).to_csv(
            FAILURE_DIR / "top_failure_cases.csv",
            index=False,
        )

    print("\n=== FINAL HYBRID ANALYSIS ===")
    print(f"Test images: {len(y_true)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4%}")
    print(f"Top-5 accuracy: {top5_accuracy:.4%}")
    print(f"Macro precision: {precision:.4%}")
    print(f"Macro recall: {recall:.4%}")
    print(f"Macro F1: {f1:.4%}")
    print(f"Average latency: {average_latency_ms:.2f} ms")
    print(f"Median latency: {median_latency_ms:.2f} ms")
    print(f"Model size: {model_size_mb:.2f} MB")
    print("Failure cases saved to:", FAILURE_DIR)
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
