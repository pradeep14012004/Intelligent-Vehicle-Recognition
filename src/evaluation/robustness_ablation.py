
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from timm.data import resolve_model_data_config, create_transform

from src.classical_cv.features import extract_classical_features


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
MAX_SAMPLES = 300

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CLASSIFIER_WEIGHTS = (
    "models/pretrained/efficientnetv2_best_model.pth"
)

HYBRID_WEIGHTS = (
    "models/trained/hybrid_classifier.pth"
)

OUTPUT_DIR = Path("results/robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Device:", DEVICE)


# ==========================================
# HYBRID MODEL
# ==========================================

class HybridClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(1792 + 8623, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3)
        )

        self.classifier = torch.nn.Linear(512, 196)

    def forward(self, x):
        return self.classifier(self.fusion(x))


# ==========================================
# LOAD MODELS
# ==========================================

print("\nLoading EfficientNetV2...")

backbone = timm.create_model(
    "efficientnetv2_rw_s",
    pretrained=False,
    num_classes=196
)

state = torch.load(
    CLASSIFIER_WEIGHTS,
    map_location="cpu",
    weights_only=True
)

backbone.load_state_dict(state, strict=True)
backbone.to(DEVICE).eval()

for parameter in backbone.parameters():
    parameter.requires_grad = False

transform = create_transform(
    **resolve_model_data_config(backbone),
    is_training=False
)

print("Loading hybrid checkpoint...")

checkpoint = torch.load(
    HYBRID_WEIGHTS,
    map_location="cpu",
    weights_only=False
)

hybrid = HybridClassifier()

hybrid.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

hybrid.to(DEVICE).eval()

cnn_mean = torch.as_tensor(
    checkpoint["cnn_mean"],
    dtype=torch.float32,
    device=DEVICE
)

cnn_std = torch.as_tensor(
    checkpoint["cnn_std"],
    dtype=torch.float32,
    device=DEVICE
)

cv_mean = torch.as_tensor(
    checkpoint["cv_mean"],
    dtype=torch.float32,
    device=DEVICE
)

cv_std = torch.as_tensor(
    checkpoint["cv_std"],
    dtype=torch.float32,
    device=DEVICE
)


# ==========================================
# ROBUSTNESS TRANSFORMATIONS
# ==========================================

def corrupt_image(image, method, index):

    image = image.copy()

    if method == "original":
        return image

    if method == "gaussian_blur":
        return cv2.GaussianBlur(
            image, (5, 5), 1.2
        )

    if method == "gaussian_noise":
        rng = np.random.default_rng(SEED + index)

        noise = rng.normal(
            0, 20, image.shape
        )

        return np.clip(
            image.astype(np.float32) + noise,
            0, 255
        ).astype(np.uint8)

    if method == "low_light":
        return np.clip(
            image.astype(np.float32) * 0.50,
            0, 255
        ).astype(np.uint8)

    if method == "jpeg":
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 35]
        )

        if success:
            decoded = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR
            )

            if decoded is not None:
                return decoded

        return image

    raise ValueError(method)


# ==========================================
# FEATURE EXTRACTION
# ==========================================

@torch.inference_mode()
def extract_features(rgb_image):

    pil_image = Image.fromarray(rgb_image)

    tensor = transform(pil_image).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    cnn = backbone.forward_features(tensor)
    cnn = backbone.global_pool(cnn).flatten(1)

    bgr = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR
    )

    cv_features = extract_classical_features(bgr)

    cnn = (cnn - cnn_mean) / cnn_std

    cv_tensor = torch.as_tensor(
        cv_features,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    cv_tensor = (cv_tensor - cv_mean) / cv_std

    return cnn, cv_tensor


# ==========================================
# PREDICTION AND METRICS
# ==========================================

def get_predictions(logits):

    probabilities = torch.softmax(
        logits, dim=1
    )

    top5 = torch.topk(
        probabilities,
        k=5,
        dim=1
    ).indices[0].cpu().numpy()

    top1 = int(top5[0])

    return top1, top5


def calculate_metrics(records):

    y_true = [
        row["true_label"] for row in records
    ]

    y_pred = [
        row["predicted_label"] for row in records
    ]

    top5_correct = [
        row["top5_correct"] for row in records
    ]

    return {
        "samples": len(records),
        "top1_accuracy": accuracy_score(
            y_true, y_pred
        ),
        "top5_accuracy": float(
            np.mean(top5_correct)
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=np.unique(y_true),
            average="macro",
            zero_division=0
        ),
        "average_inference_ms": float(
            np.mean([
                row["inference_ms"]
                for row in records
            ])
        )
    }


# ==========================================
# LOAD OFFICIAL TEST DATA
# ==========================================

print("\nLoading Stanford Cars test split...")

dataset = load_dataset(
    "bapei/stanford_cars",
    split="test"
)

rng = np.random.default_rng(SEED)

sample_count = min(
    MAX_SAMPLES,
    len(dataset)
)

selected_indices = np.sort(
    rng.choice(
        len(dataset),
        size=sample_count,
        replace=False
    )
)

print("Selected test images:", sample_count)


# ==========================================
# ROBUSTNESS EXPERIMENT
# ==========================================

conditions = [
    "original",
    "gaussian_blur",
    "gaussian_noise",
    "low_light",
    "jpeg"
]

robustness_results = {}

all_robustness_rows = []

print("\n========== ROBUSTNESS TEST ==========")

for condition in conditions:

    print(f"\nTesting: {condition}")

    rows = []

    for count, index in enumerate(
        selected_indices, start=1
    ):

        sample = dataset[int(index)]

        rgb = np.asarray(
            sample["image"].convert("RGB")
        )

        transformed = corrupt_image(
            rgb,
            condition,
            int(index)
        )

        true_label = int(sample["label"])

        start = time.perf_counter()

        cnn, cv_features = extract_features(
            transformed
        )

        combined = torch.cat(
            [cnn, cv_features],
            dim=1
        )

        with torch.inference_mode():

            logits = hybrid(combined)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        pred, top5 = get_predictions(logits)

        row = {
            "condition": condition,
            "true_label": true_label,
            "predicted_label": pred,
            "top5_correct": int(
                true_label in top5
            ),
            "inference_ms": elapsed
        }

        rows.append(row)
        all_robustness_rows.append(row)

        if count % 50 == 0:
            print(
                f"Processed {count}/{sample_count}"
            )

    metrics = calculate_metrics(rows)

    robustness_results[condition] = metrics

    print(
        f"Top-1: {metrics['top1_accuracy']:.2%} | "
        f"Top-5: {metrics['top5_accuracy']:.2%} | "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )


# ==========================================
# ABLATION EXPERIMENT
# ==========================================

print("\n========== HYBRID ABLATION TEST ==========")

ablation_rows = []

for count, index in enumerate(
    selected_indices, start=1
):

    sample = dataset[int(index)]

    rgb = np.asarray(
        sample["image"].convert("RGB")
    )

    true_label = int(sample["label"])

    cnn, cv_features = extract_features(rgb)

    # Full hybrid
    full_features = torch.cat(
        [cnn, cv_features], dim=1
    )

    # CNN contribution masked
    cnn_masked = torch.cat(
        [
            torch.zeros_like(cnn),
            cv_features
        ],
        dim=1
    )

    # Classical CV contribution masked
    cv_masked = torch.cat(
        [
            cnn,
            torch.zeros_like(cv_features)
        ],
        dim=1
    )

    with torch.inference_mode():

        full_logits = hybrid(full_features)
        cnn_masked_logits = hybrid(cnn_masked)
        cv_masked_logits = hybrid(cv_masked)

    variants = {
        "full_hybrid": full_logits,
        "without_cnn_features": cv_masked_logits,
        "without_classical_features": cnn_masked_logits
    }

    for name, logits in variants.items():

        pred, top5 = get_predictions(logits)

        ablation_rows.append({
            "variant": name,
            "true_label": true_label,
            "predicted_label": pred,
            "top5_correct": int(
                true_label in top5
            )
        })

    if count % 50 == 0:
        print(
            f"Ablation processed {count}/{sample_count}"
        )


ablation_results = {}

for variant in [
    "full_hybrid",
    "without_cnn_features",
    "without_classical_features"
]:

    rows = [
        row for row in ablation_rows
        if row["variant"] == variant
    ]

    y_true = [
        row["true_label"] for row in rows
    ]

    y_pred = [
        row["predicted_label"] for row in rows
    ]

    ablation_results[variant] = {
        "top1_accuracy": accuracy_score(
            y_true, y_pred
        ),
        "top5_accuracy": float(
            np.mean([
                row["top5_correct"]
                for row in rows
            ])
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=np.unique(y_true),
            average="macro",
            zero_division=0
        )
    }


# ==========================================
# SAVE RESULTS
# ==========================================

robustness_df = pd.DataFrame(
    [
        {
            "Condition": name,
            "Top-1 Accuracy": value["top1_accuracy"],
            "Top-5 Accuracy": value["top5_accuracy"],
            "Macro F1": value["macro_f1"],
            "Inference ms/image": value[
                "average_inference_ms"
            ]
        }
        for name, value in robustness_results.items()
    ]
)

ablation_df = pd.DataFrame(
    [
        {
            "Variant": name,
            **metrics
        }
        for name, metrics in ablation_results.items()
    ]
)

robustness_df.to_csv(
    OUTPUT_DIR / "robustness_metrics.csv",
    index=False
)

ablation_df.to_csv(
    OUTPUT_DIR / "ablation_metrics.csv",
    index=False
)

with open(
    OUTPUT_DIR / "experiment_results.json",
    "w"
) as f:

    json.dump(
        {
            "device": DEVICE,
            "test_samples": sample_count,
            "robustness": robustness_results,
            "ablation": ablation_results,
            "ablation_method": (
                "Post-hoc feature masking in the trained "
                "hybrid model; not independently trained branches."
            )
        },
        f,
        indent=4
    )


# ==========================================
# FINAL REPORT
# ==========================================

print("\n========== ROBUSTNESS RESULTS ==========")

print(
    robustness_df.to_string(index=False)
)

print("\n========== ABLATION RESULTS ==========")

print(
    ablation_df.to_string(index=False)
)

print("\nResults saved to:", OUTPUT_DIR)
print("EXPERIMENTS COMPLETED")
