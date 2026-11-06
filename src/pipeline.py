
import cv2

from src.config import load_config
from src.preprocessing.image import enhance_image, preprocess_image
from src.detection.yolo_detector import VehicleDetector
from src.roi.extract import extract_roi
from src.deep_learning.classifier import VehicleClassifier
from src.classical_cv.features import extract_classical_features
from src.fusion.hybrid import HybridVehiclePredictor


class VehicleRecognitionPipeline:

    def __init__(self, config=None):
        self.cfg = config or load_config()

        model_cfg = self.cfg["model"]
        inf = self.cfg["inference"]

        self.detector = VehicleDetector(
            model_cfg["detector_weights"],
            inf["confidence_threshold"],
            inf["iou_threshold"]
        )

        self.classifier = VehicleClassifier(
            weights=model_cfg["classifier_weights"],
            mapping=self.cfg["class_names_file"],
            device=inf["device"]
        )

        self.hybrid_predictor = HybridVehiclePredictor(
            backbone=self.classifier.model,
            transform=self.classifier.transform,
            classes=self.classifier.classes,
            weights=model_cfg.get(
                "hybrid_weights",
                "models/trained/hybrid_classifier.pth"
            ),
            device=inf["device"]
        )

    def detect_and_crop(self, image, enhancement="none"):

        image = enhance_image(image, enhancement)
        detections = self.detector.predict(image)

        results = []

        for det in detections:

            roi = extract_roi(
                image,
                det["bbox"],
                self.cfg["inference"]["roi_padding"]
            )

            classical_features = extract_classical_features(roi)

            predictions = []
            explanation = None

            if det["class_name"] == "car":

                predictions = self.hybrid_predictor.predict(
                    roi,
                    top_k=5
                )

                if predictions:
                    explanation = self.classifier.explain(
                        roi,
                        class_id=predictions[0]["class_id"]
                    )

                status = "Hybrid car-model classification active"

            else:
                status = (
                    "Vehicle detected; Stanford Cars classifier "
                    "supports car-model recognition only"
                )

            results.append({
                **det,
                "roi": roi,
                "predictions": predictions,
                "explanation": explanation,
                "classification_status": status,
                "classical_feature_dimension": int(
                    classical_features.shape[0]
                )
            })

        return image, results

    def compare_preprocessing(
        self,
        image,
        methods=("original", "gaussian", "bilateral", "median")
    ):
        """
        Compare preprocessing methods using:
        - One detection pass on the original image.
        - The same bounding box for every preprocessing variant.
        - The trained hybrid classifier for each ROI.

        Input image is expected in OpenCV BGR format.
        """

        # Detect once on the original image.
        detections = self.detector.predict(image)

        comparison_results = []

        for det_index, det in enumerate(detections):

            if det["class_name"] != "car":
                continue

            method_results = {}

            for method in methods:

                # Convert BGR to RGB for preprocessing.
                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                # Apply selected preprocessing.
                processed_rgb = preprocess_image(
                    rgb,
                    method
                )

                # Convert back to BGR for ROI extraction
                # and consistency with the existing pipeline.
                processed_bgr = cv2.cvtColor(
                    processed_rgb,
                    cv2.COLOR_RGB2BGR
                )

                # Use exactly the same detector bounding box.
                roi = extract_roi(
                    processed_bgr,
                    det["bbox"],
                    self.cfg["inference"]["roi_padding"]
                )

                # Hybrid inference.
                predictions = self.hybrid_predictor.predict(
                    roi,
                    top_k=5
                )

                if predictions:
                    top1 = predictions[0]

                    top2_confidence = (
                        float(predictions[1]["confidence"])
                        if len(predictions) > 1
                        else 0.0
                    )

                    top1_confidence = float(
                        top1["confidence"]
                    )

                    method_results[method] = {
                        "vehicle_model": top1["vehicle_model"],
                        "confidence": top1_confidence,
                        "top2_model": (
                            predictions[1]["vehicle_model"]
                            if len(predictions) > 1
                            else "N/A"
                        ),
                        "top2_confidence": top2_confidence,
                        "confidence_gap": (
                            top1_confidence - top2_confidence
                        ),
                        "predictions": predictions,
                        "roi": roi
                    }

            comparison_results.append({
                "detection_index": det_index + 1,
                "bbox": det["bbox"],
                "detection_confidence": float(
                    det["confidence"]
                ),
                "methods": method_results
            })

        return comparison_results
