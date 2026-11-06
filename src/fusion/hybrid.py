
import numpy as np
import torch
from torch import nn
from PIL import Image

from src.classical_cv.features import extract_classical_features


class HybridClassifier(nn.Module):
    def __init__(self, num_classes=196):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(1792 + 8623, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.classifier(self.fusion(x))


class HybridVehiclePredictor:
    def __init__(
        self,
        backbone,
        transform,
        classes,
        weights="models/trained/hybrid_classifier.pth",
        device="auto"
    ):
        if device == "auto":
            device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.backbone = backbone
        self.transform = transform
        self.classes = classes

        # Load the trained fusion checkpoint
        checkpoint = torch.load(
            weights,
            map_location="cpu",
            weights_only=False
        )

        if checkpoint["num_classes"] != 196:
            raise ValueError("Expected a 196-class checkpoint")

        self.model = HybridClassifier(
            num_classes=checkpoint["num_classes"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        self.model.to(self.device).eval()

        # Exact normalization statistics used during training
        self.cnn_mean = torch.as_tensor(
            checkpoint["cnn_mean"], dtype=torch.float32
        ).to(self.device)

        self.cnn_std = torch.as_tensor(
            checkpoint["cnn_std"], dtype=torch.float32
        ).to(self.device)

        self.cv_mean = np.asarray(
            checkpoint["cv_mean"], dtype=np.float32
        )

        self.cv_std = np.asarray(
            checkpoint["cv_std"], dtype=np.float32
        )

        self.validation_accuracy = checkpoint.get(
            "best_validation_accuracy"
        )

        print("Hybrid fusion checkpoint loaded successfully")
        print("Preliminary validation accuracy:",
              self.validation_accuracy)

    @torch.inference_mode()
    def predict(self, roi_bgr, top_k=5):
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("Empty ROI supplied")

        # 1. CNN feature extraction
        rgb = roi_bgr[:, :, ::-1].copy()
        pil_image = Image.fromarray(rgb)

        image_tensor = self.transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        features = self.backbone.forward_features(image_tensor)
        cnn_features = self.backbone.global_pool(
            features
        ).flatten(1)

        cnn_features = (
            cnn_features - self.cnn_mean
        ) / self.cnn_std

        # 2. Classical CV feature extraction
        cv_features = np.asarray(
            extract_classical_features(roi_bgr),
            dtype=np.float32
        ).reshape(1, -1)

        if cv_features.shape[1] != 8623:
            raise ValueError(
                f"Expected 8623 classical features, "
                f"received {cv_features.shape[1]}"
            )

        cv_features = (
            cv_features - self.cv_mean
        ) / self.cv_std

        cv_tensor = torch.from_numpy(cv_features).to(self.device)

        # 3. Hybrid feature fusion
        combined = torch.cat(
            [cnn_features, cv_tensor],
            dim=1
        )

        # 4. Final 196-class prediction
        logits = self.model(combined)
        probabilities = torch.softmax(logits, dim=1)[0]

        scores, indices = torch.topk(
            probabilities,
            k=min(top_k, probabilities.numel())
        )

        predictions = []

        for score, index in zip(
            scores.cpu().tolist(),
            indices.cpu().tolist()
        ):
            predictions.append({
                "class_id": index,
                "vehicle_model": self.classes.get(
                    str(index), f"Unknown class {index}"
                ),
                "confidence": float(score)
            })

        print("HYBRID FUSION INFERENCE ACTIVE")
        print("Combined feature shape:", combined.shape)
        print("Hybrid top prediction:", predictions[0])

        return predictions
