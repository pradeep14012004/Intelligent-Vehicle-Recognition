import sys
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import VehicleRecognitionPipeline

app = FastAPI(
    title="Intelligent Vehicle Recognition API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = VehicleRecognitionPipeline()
    return pipeline


def encode_image(image):
    if image is None:
        return None

    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Intelligent Vehicle Recognition API"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/recognize")
async def recognize_vehicle(
    file: UploadFile = File(...),
    preprocessing: str = Form("none")
):
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Upload a JPG or PNG image.")

    try:
        contents = await file.read()
        image = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(400, "Invalid image.")

        model_pipeline = get_pipeline()

        processed, detections = model_pipeline.detect_and_crop(
            image.copy(),
            preprocessing
        )

        output = []

        for item in detections:
            predictions = item.get("predictions", [])
            explanation = item.get("explanation")

            output.append({
                "bbox": item.get("bbox"),
                "class_name": item.get("class_name"),
                "detection_confidence": float(item.get("confidence", 0)),
                "roi": encode_image(item.get("roi")),
                "predictions": [
                    {
                        "vehicle_model": p["vehicle_model"],
                        "confidence": float(p["confidence"]),
                        "class_id": int(p["class_id"])
                    }
                    for p in predictions
                ],
                "classification_status": item.get(
                    "classification_status", ""
                ),
                "classical_feature_dimension": item.get(
                    "classical_feature_dimension"
                ),
                "gradcam": encode_image(
                    explanation.get("overlay")
                ) if explanation else None
            })

        return {
            "success": True,
            "detection_count": len(output),
            "annotated_image": encode_image(processed),
            "detections": output
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/compare")
async def compare_preprocessing(
    file: UploadFile = File(...)
):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(400, "Upload a JPG or PNG image.")

    contents = await file.read()
    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(400, "Invalid image.")

    try:
        model_pipeline = get_pipeline()
        comparisons = model_pipeline.compare_preprocessing(image)

        output = []

        for result in comparisons:
            methods = {}

            for name, prediction in result["methods"].items():
                methods[name] = {
                    "vehicle_model": prediction["vehicle_model"],
                    "confidence": float(prediction["confidence"]),
                    "top2_model": prediction["top2_model"],
                    "top2_confidence": float(prediction["top2_confidence"]),
                    "confidence_gap": float(prediction["confidence_gap"]),
                    "roi": encode_image(prediction["roi"])
                }

            output.append({
                "detection_index": result["detection_index"],
                "detection_confidence": result["detection_confidence"],
                "methods": methods
            })

        return {"success": True, "comparisons": output}

    except Exception as exc:
        raise HTTPException(500, str(exc))
