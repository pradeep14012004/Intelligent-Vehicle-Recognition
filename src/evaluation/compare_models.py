import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from src.classical_cv.features import extract_classical_features
from src.deep_learning.classifier import VehicleClassifier
from src.fusion.hybrid import HybridVehiclePredictor


SEED = 42
RESULTS = Path("results/comparison")
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

TRAIN_LIMIT = None
TEST_LIMIT = None
TRAIN_SAMPLES_PER_CLASS = 20

np.random.seed(SEED)


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def fit_classical_svm(train_ds, test_ds):
    print("\n[1/3] Training classical CV SVM baseline...")

    print("\nExtracting classical training features...")
    selected_indices = []
    for label in range(196):
        indices = [
            i for i, value in enumerate(train_ds["label"])
            if value == label
        ]
        selected_indices.extend(indices[:TRAIN_SAMPLES_PER_CLASS])

    train_subset = train_ds.select(selected_indices)
    X_train_cv = []
    y_train = []

    for i, sample in enumerate(train_subset):
        image = sample["image"].convert("RGB")
        bgr = np.asarray(image)[:, :, ::-1].copy()
        X_train_cv.append(extract_classical_features(bgr))
        y_train.append(sample["label"])
        if (i + 1) % 100 == 0:
            print(f"Extracted {i + 1}/{len(train_subset)}")

    X_train_cv = np.asarray(X_train_cv, dtype=np.float32)
    y_train = np.asarray(y_train)

    print("Training faster classical CV classifier...")
    svm = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="hinge",
            alpha=0.0001,
            max_iter=30,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,
        ),
    )

    start = time.time()
    svm.fit(X_train_cv, y_train)
    fit_time = time.time() - start
    print("Classical CV model training completed.")

    test_X = []
    test_y = []

    for idx, sample in enumerate(test_ds):
        if TEST_LIMIT is not None and idx >= TEST_LIMIT:
            break

        image = np.asarray(sample["image"].convert("RGB"))[:, :, ::-1].copy()
        test_X.append(extract_classical_features(image))
        test_y.append(int(sample["label"]))

    test_X = np.asarray(test_X, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.int64)

    start = time.time()
    test_pred = svm.predict(test_X)
    inference_time = time.time() - start

    metrics = evaluate_predictions(test_y, test_pred)
    metrics["fit_seconds"] = float(fit_time)
    metrics["inference_seconds"] = float(inference_time)
    metrics["samples"] = int(len(test_y))

    print("Classical CV baseline metrics:", metrics)
    return metrics, test_y, test_pred


def evaluate_deep_model(test_ds):
    print("\n[2/3] Evaluating pretrained EfficientNetV2 classifier...")

    deep = VehicleClassifier(
        weights="models/pretrained/efficientnetv2_best_model.pth",
        mapping="data/metadata/class_mapping.json",
        device=DEVICE,
    )

    y_true = []
    y_pred = []
    start = time.time()

    for idx, sample in enumerate(test_ds):
        if TEST_LIMIT is not None and idx >= TEST_LIMIT:
            break

        image = np.asarray(sample["image"].convert("RGB"))[:, :, ::-1].copy()
        predictions = deep.predict(image, top_k=1)
        y_true.append(int(sample["label"]))
        y_pred.append(int(predictions[0]["class_id"]))

    inference_time = time.time() - start
    metrics = evaluate_predictions(np.asarray(y_true), np.asarray(y_pred))
    metrics["inference_seconds"] = float(inference_time)
    metrics["samples"] = int(len(y_true))

    print("Deep learning metrics:", metrics)
    return metrics, np.asarray(y_true), np.asarray(y_pred)


def evaluate_hybrid_model(test_ds):
    print("\n[3/3] Evaluating hybrid EfficientNetV2 + classical CV fusion...")

    deep = VehicleClassifier(
        weights="models/pretrained/efficientnetv2_best_model.pth",
        mapping="data/metadata/class_mapping.json",
        device=DEVICE,
    )

    hybrid = HybridVehiclePredictor(
        backbone=deep.model,
        transform=deep.transform,
        classes=deep.classes,
        weights="models/trained/hybrid_classifier.pth",
        device=DEVICE,
    )

    y_true = []
    y_pred = []
    start = time.time()

    for idx, sample in enumerate(test_ds):
        if TEST_LIMIT is not None and idx >= TEST_LIMIT:
            break

        image = np.asarray(sample["image"].convert("RGB"))[:, :, ::-1].copy()
        predictions = hybrid.predict(image, top_k=1)
        y_true.append(int(sample["label"]))
        y_pred.append(int(predictions[0]["class_id"]))

    inference_time = time.time() - start
    metrics = evaluate_predictions(np.asarray(y_true), np.asarray(y_pred))
    metrics["inference_seconds"] = float(inference_time)
    metrics["samples"] = int(len(y_true))

    print("Hybrid metrics:", metrics)
    return metrics, np.asarray(y_true), np.asarray(y_pred)


def main():
    print("Device:", DEVICE)
    print("Loading dataset...")

    train_ds = load_dataset("bapei/stanford_cars", split="train")
    test_ds = load_dataset("bapei/stanford_cars", split="test")

    if TRAIN_LIMIT is not None:
        train_ds = train_ds.select(range(min(TRAIN_LIMIT, len(train_ds))))
    if TEST_LIMIT is not None:
        test_ds = test_ds.select(range(min(TEST_LIMIT, len(test_ds))))

    print("Training samples:", len(train_ds))
    print("Testing samples:", len(test_ds))

    classical_metrics, _, _ = fit_classical_svm(train_ds, test_ds)
    deep_metrics, _, _ = evaluate_deep_model(test_ds)
    hybrid_metrics, _, _ = evaluate_hybrid_model(test_ds)

    comparison = {
        "device": DEVICE,
        "dataset": "Stanford Cars",
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "classical_cv": classical_metrics,
        "deep_learning": deep_metrics,
        "hybrid": hybrid_metrics,
    }

    output_path = RESULTS / "model_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\nComparison saved to:", output_path)
    print(json.dumps({
        "classical_cv": classical_metrics,
        "deep_learning": deep_metrics,
        "hybrid": hybrid_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
