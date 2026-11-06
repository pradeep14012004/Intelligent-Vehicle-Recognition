import json

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from timm.data import resolve_model_data_config, create_transform


class VehicleClassifier:
    def __init__(
        self,
        weights="models/pretrained/efficientnetv2_best_model.pth",
        mapping="data/metadata/class_mapping.json",
        device="auto",
    ):
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = timm.create_model(
            "efficientnetv2_rw_s",
            pretrained=False,
            num_classes=196,
        )

        state = torch.load(
            weights,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        with open(mapping, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        self.transform = create_transform(
            **resolve_model_data_config(self.model),
            is_training=False,
        )

    def _prepare(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, bgr_image, top_k=5):
        tensor = self._prepare(bgr_image)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        scores, indices = torch.topk(
            probabilities,
            k=min(top_k, len(self.classes)),
        )

        return [
            {
                "class_id": index,
                "vehicle_model": self.classes.get(
                    str(index), f"Unknown class {index}"
                ),
                "confidence": float(score),
            }
            for score, index in zip(scores.tolist(), indices.tolist())
        ]

    def explain(self, bgr_image, class_id=None):
        """
        Grad-CAM using EfficientNetV2 conv_head activations.
        Returns the selected class, heatmap and overlay.
        """
        activations = {}
        gradients = {}

        def forward_hook(module, inputs, output):
            activations["value"] = output
            output.register_hook(
                lambda grad: gradients.update({"value": grad})
            )

        target_layer = self.model.conv_head
        handle = target_layer.register_forward_hook(forward_hook)

        try:
            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                tensor = self._prepare(bgr_image)
                logits = self.model(tensor)

                if class_id is None:
                    class_id = int(logits.argmax(dim=1).item())

                score = logits[0, class_id]
                score.backward()

            activation = activations["value"].detach()
            gradient = gradients["value"].detach()

            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1))[0]

            cam = cam.cpu().numpy()
            cam -= cam.min()
            cam /= cam.max() + 1e-8

            height, width = bgr_image.shape[:2]
            heatmap = cv2.resize(cam, (width, height))
            heatmap_uint8 = np.uint8(255 * heatmap)

            colored = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET,
            )

            overlay = cv2.addWeighted(
                bgr_image, 0.55, colored, 0.45, 0
            )

            return {
                "class_id": class_id,
                "vehicle_model": self.classes.get(
                    str(class_id), "Unknown"
                ),
                "heatmap": heatmap_uint8,
                "overlay": overlay,
            }

        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)
