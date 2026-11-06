
import time
import random
from pathlib import Path

import numpy as np
import torch
import timm

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from timm.data import create_transform, resolve_model_data_config

from src.classical_cv.features import extract_classical_features


# ==================================================
# CONFIGURATION
# ==================================================

START = time.time()

SEED = 42
EPOCHS = 8
BATCH_SIZE = 128
VALIDATION_SIZE = 0.10
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = CACHE_DIR / "stanford_cars_features_8144.npz"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading Stanford Cars training split...")

ds = load_dataset(
    "bapei/stanford_cars",
    split="train"
)

labels = np.asarray(ds["label"], dtype=np.int64)
indices = np.arange(len(ds))

print("Training images:", len(ds))
print("Classes:", len(np.unique(labels)))


# Stratified split: validation data comes only from train split
train_idx, val_idx = train_test_split(
    indices,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=labels
)

print("Training samples:", len(train_idx))
print("Validation samples:", len(val_idx))


# ==================================================
# FEATURE EXTRACTION / CACHE
# ==================================================

if CACHE_FILE.exists():

    print("\nLoading cached features...")

    cached = np.load(CACHE_FILE)

    cnn = cached["cnn"]
    cv = cached["cv"]
    y = cached["labels"]

    print("Cached features loaded:", len(y))

else:

    print("\nLoading pretrained EfficientNetV2...")

    backbone = timm.create_model(
        "efficientnetv2_rw_s",
        pretrained=False,
        num_classes=196
    )

    state = torch.load(
        "models/pretrained/efficientnetv2_best_model.pth",
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

    cnn_features = []
    cv_features = []

    print("\nExtracting CNN and classical features...")

    for start in range(0, len(ds), 16):

        end = min(start + 16, len(ds))

        samples = [
            ds[i] for i in range(start, end)
        ]

        images = [
            sample["image"].convert("RGB")
            for sample in samples
        ]

        batch = torch.stack(
            [transform(image) for image in images]
        ).to(DEVICE)

        with torch.inference_mode():

            features = backbone.forward_features(batch)

            features = backbone.global_pool(
                features
            ).flatten(1)

        cnn_features.extend(
            features.cpu().numpy()
        )

        for image in images:

            bgr = np.asarray(image)[:, :, ::-1].copy()

            feature = extract_classical_features(bgr)

            cv_features.append(feature)

        if end % 128 == 0 or end == len(ds):
            print(f"Extracted {end}/{len(ds)}")

    cnn = np.asarray(cnn_features, dtype=np.float32)
    cv = np.asarray(cv_features, dtype=np.float32)
    y = labels

    if cnn.shape != (len(ds), 1792):
        raise ValueError(f"Unexpected CNN feature shape: {cnn.shape}")

    if cv.shape != (len(ds), 8623):
        raise ValueError(f"Unexpected CV feature shape: {cv.shape}")

    print("\nSaving feature cache...")

    np.savez(
        CACHE_FILE,
        cnn=cnn,
        cv=cv,
        labels=y
    )

    del backbone

    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ==================================================
# NORMALIZATION
# ==================================================

print("\nNormalizing features...")

cnn_mean = cnn[train_idx].mean(axis=0)
cnn_std = cnn[train_idx].std(axis=0) + 1e-6

cv_mean = cv[train_idx].mean(axis=0)
cv_std = cv[train_idx].std(axis=0) + 1e-6

cnn = (cnn - cnn_mean) / cnn_std
cv = (cv - cv_mean) / cv_std

X = torch.from_numpy(
    np.concatenate([cnn, cv], axis=1).astype(np.float32)
)

y_tensor = torch.from_numpy(y.astype(np.int64))

print("Combined feature shape:", X.shape)


# ==================================================
# DATA LOADERS
# ==================================================

train_dataset = TensorDataset(
    X[train_idx],
    y_tensor[train_idx]
)

val_dataset = TensorDataset(
    X[val_idx],
    y_tensor[val_idx]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# HYBRID FUSION MODEL
# ==================================================

class HybridClassifier(nn.Module):

    def __init__(self, num_classes=196):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(1792 + 8623, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        fused = self.fusion(x)

        return self.classifier(fused)


model = HybridClassifier().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==================================================
# TRAINING
# ==================================================

best_accuracy = 0.0

CHECKPOINT = Path(
    "models/trained/hybrid_classifier.pth"
)

CHECKPOINT.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nStarting full-dataset hybrid training...")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(yb)

    # Validation
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            predictions = model(xb).argmax(dim=1)

            correct += (
                predictions == yb
            ).sum().item()

            total += len(yb)

    accuracy = correct / total

    average_loss = total_loss / len(train_idx)

    scheduler.step(accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Validation accuracy: {accuracy:.2%}"
    )

    # Save best checkpoint
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cnn_mean": cnn_mean,
                "cnn_std": cnn_std,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "num_classes": 196,
                "best_validation_accuracy": best_accuracy,
                "training_samples": len(train_idx),
                "validation_samples": len(val_idx),
                "epochs": EPOCHS
            },
            CHECKPOINT
        )

        print("Best checkpoint saved.")


# ==================================================
# COMPLETE
# ==================================================

print("\n========== TRAINING COMPLETE ==========")

print("Best validation accuracy:", f"{best_accuracy:.2%}")
print("Training samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Checkpoint:", CHECKPOINT)
print("Feature cache:", CACHE_FILE)
print("Elapsed seconds:", round(time.time() - START, 1))
