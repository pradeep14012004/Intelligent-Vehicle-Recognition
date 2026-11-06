import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from datasets import load_dataset
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_model_data_config

from src.classical_cv.features import extract_classical_features
from src.deep_learning.classifier import VehicleClassifier
from src.fusion.hybrid import HybridVehiclePredictor


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
MAX_TEST_IMAGES = 300
TRAIN_SAMPLES_PER_CLASS = 20

RESULT_DIR = Path("results/three_model_comparison")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

CLASSIFIER_WEIGHTS = "models/pretrained/efficientnetv2_best_model.pth"
HYBRID_WEIGHTS = "models/trained/hybrid_classifier.pth"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# HELPERS
# ==========================================

def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.unique(y_true),
                average="macro",
                zero_division=0,
            )
        ),
    }


# ==========================================
# LOAD DATA
# ==========================================

print("Loading Stanford Cars train/test splits...")
train_ds = load_dataset("bapei/stanford_cars", split="train")
test_ds = load_dataset("bapei/stanford_cars", split="test")

if MAX_TEST_IMAGES is not None:
    selected = np.random.default_rng(SEED).choice(
        len(test_ds),
        size=min(MAX_TEST_IMAGES, len(test_ds)),
        replace=False,
    )
    selected = np.sort(selected)
    test_ds = test_ds.select(selected)

print("Train samples:", len(train_ds))
print("Test samples:", len(test_ds))


# ==========================================
# CLASSICAL CV BASELINE
# ==========================================

print("\nTraining classical CV baseline...")
selected_indices = []
for label in range(196):
    label_indices = [
        idx for idx, value in enumerate(train_ds["label"]) if value == label
    ]
    selected_indices.extend(label_indices[:TRAIN_SAMPLES_PER_CLASS])

train_subset = train_ds.select(selected_indices)
X_train = []
y_train = []

for i, sample in enumerate(train_subset):
    image = np.asarray(sample["image"].convert("RGB"))[:, :, ::-1].copy()
    X_train.append(extract_classical_features(image))
    y_train.append(int(sample["label"]))
    if (i + 1) % 100 == 0:
        print(f"Extracted {i + 1}/{len(train_subset)} classical training samples")

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)

classical_model = make_pipeline(
    StandardScaler(),
    SGDClassifier(
        loss="hinge",
        alpha=0.0001,
        max_iter=30,
        tol=1e-3,
        random_state=SEED,
        n_jobs=-1,
    ),
)
classical_model.fit(X_train, y_train)

print("Classical CV model training completed.")

# Evaluate classical model
X_test = []
y_true_cv = []
for sample in test_ds:
    image = np.asarray(sample["image"].convert("RGB"))[:, :, ::-1].copy()
    X_test.append(extract_classical_features(image))
    y_true_cv.append(int(sample["label"]))

X_test = np.asarray(X_test, dtype=np.float32)
y_true_cv = np.asarray(y_true_cv, dtype=np.int64)
start = time.time()
y_pred_cv = classical_model.predict(X_test)
cv_inference = time.time() - start
cv_metrics = evaluate_predictions(y_true_cv, y_pred_cv)
cv_metrics["inference_seconds"] = float(cv_inference)
cv_metrics["samples"] = int(len(y_true_cv))
print("Classical CV metrics:", cv_metrics)


# ==========================================
# DEEP LEARNING MODEL
# ==========================================

print("\nLoading EfficientNetV2...")
deep = VehicleClassifier(
    weights=CLASSIFIER_WEIGHTS,
    mapping="data/metadata/class_mapping.json",
    device=DEVICE,
)

y_true_deep = []
y_pred_deep = []
start = time.time()
for sample in test_ds:
    image = np.asarray(sample["image"].convert("RGB"))[:, :, ::-1].copy()
    predictions = deep.predict(image, top_k=1)
    y_true_deep.append(int(sample["label"]))
    y_pred_deep.append(int(predictions[0]["class_id"]))

deep_inference = time.time() - start
deep_metrics = evaluate_predictions(np.asarray(y_true_deep), np.asarray(y_pred_deep))
deep_metrics["inference_seconds"] = float(deep_inference)
deep_metrics["samples"] = int(len(y_true_deep))
print("Deep learning metrics:", deep_metrics)


# ==========================================
# HYBRID MODEL
# ==========================================

print("\nLoading hybrid model...")
hybrid = HybridVehiclePredictor(
    backbone=deep.model,
    transform=deep.transform,
    classes=deep.classes,
    weights=HYBRID_WEIGHTS,
    device=DEVICE,
)

y_true_hybrid = []
y_pred_hybrid = []
start = time.time()
for sample in test_ds:
    image = np.asarray(sample["image"].convert("RGB"))[:, :, ::-1].copy()
    predictions = hybrid.predict(image, top_k=1)
    y_true_hybrid.append(int(sample["label"]))
    y_pred_hybrid.append(int(predictions[0]["class_id"]))

hybrid_inference = time.time() - start
hybrid_metrics = evaluate_predictions(np.asarray(y_true_hybrid), np.asarray(y_pred_hybrid))
hybrid_metrics["inference_seconds"] = float(hybrid_inference)
hybrid_metrics["samples"] = int(len(y_true_hybrid))
print("Hybrid metrics:", hybrid_metrics)


# ==========================================
# SAVE RESULTS
# ==========================================

comparison = {
    "device": DEVICE,
    "dataset": "Stanford Cars",
    "max_test_images": int(len(test_ds)),
    "train_samples_per_class": TRAIN_SAMPLES_PER_CLASS,
    "classical_cv": cv_metrics,
    "deep_learning": deep_metrics,
    "hybrid": hybrid_metrics,
}

with open(RESULT_DIR / "three_model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

summary_df = pd.DataFrame(
    [
        {"Model": "Classical CV", **cv_metrics},
        {"Model": "Deep Learning", **deep_metrics},
        {"Model": "Hybrid", **hybrid_metrics},
    ]
)
summary_df.to_csv(RESULT_DIR / "three_model_comparison.csv", index=False)

print("\nSaved comparison results to:", RESULT_DIR)
print(summary_df.to_string(index=False))
