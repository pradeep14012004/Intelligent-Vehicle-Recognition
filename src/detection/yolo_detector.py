from pathlib import Path
from typing import Any

class VehicleDetector:
    """Lazy YOLO wrapper. COCO class 2 is car; extend class policy for other vehicles."""
    def __init__(self, weights: str, confidence: float = 0.25, iou: float = 0.45):
        self.weights, self.confidence, self.iou = weights, confidence, iou
        self.model = None

    def load(self):
        from ultralytics import YOLO
        if not Path(self.weights).exists():
            raise FileNotFoundError(f"Detector weights not found: {self.weights}. Download pretrained YOLO weights first.")
        self.model = YOLO(self.weights)
        return self

    def predict(self, image) -> list[dict[str, Any]]:
        if self.model is None: self.load()
        result = self.model.predict(image, conf=self.confidence, iou=self.iou, verbose=False)[0]
        detections = []
        if result.boxes is None: return detections
        for box in result.boxes:
            cls = int(box.cls.item())
            name = self.model.names.get(cls, str(cls))
            if name not in {"car", "truck", "bus", "motorcycle"}: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf.item()), "class_name": name})
        return detections
