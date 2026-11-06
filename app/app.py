
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import cv2
import numpy as np
import streamlit as st

from src.pipeline import VehicleRecognitionPipeline
from src.preprocessing.image import get_preprocessing_variants


def get_confidence_status(confidence):

    if confidence >= 0.75:
        return "High", "success"

    elif confidence >= 0.50:
        return "Medium", "warning"

    return "Low", "error"


def show_confidence(confidence):

    status, level = get_confidence_status(confidence)

    st.metric(
        "Model confidence",
        f"{confidence * 100:.2f}%"
    )

    if level == "success":
        st.success("High confidence prediction.")

    elif level == "warning":
        st.warning(
            "Moderate confidence. Similar models may be confused."
        )

    else:
        st.error(
            "Low confidence. Treat this prediction as uncertain."
        )


st.set_page_config(
    page_title="P5 Vehicle Recognition",
    layout="wide"
)

st.title("P5 — Intelligent Vehicle Recognition")

st.caption(
    "Preprocessing → Detection → ROI extraction → "
    "Hybrid vehicle recognition"
)

upload = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

mode = st.selectbox(
    "Preprocessing for main detection",
    ["none", "gaussian", "bilateral", "clahe", "gamma"]
)


if upload:

    data = upload.getvalue()

    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error("Unable to read this image.")
        st.stop()

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        rgb_image,
        caption="Input image",
        use_container_width=True
    )

    # ==========================================
    # IMAGE PREPROCESSING VISUAL COMPARISON
    # ==========================================

    st.divider()
    st.header("1. Image Preprocessing Comparison")

    variants = get_preprocessing_variants(rgb_image)

    titles = {
        "original": "Original",
        "gaussian": "Gaussian",
        "bilateral": "Bilateral",
        "median": "Median"
    }

    columns = st.columns(4)

    for column, (method, processed) in zip(
        columns,
        variants.items()
    ):

        with column:

            st.subheader(titles[method])

            st.image(
                processed,
                use_container_width=True
            )

    st.caption(
        "Visual comparison of noise reduction and edge preservation."
    )

    # ==========================================
    # RUN COMPLETE PIPELINE
    # ==========================================

    if st.button(
        "Run vehicle detection and comparison",
        type="primary"
    ):

        try:

            with st.spinner(
                "Loading models and running vehicle recognition..."
            ):

                pipeline = VehicleRecognitionPipeline()

                processed, results = pipeline.detect_and_crop(
                    image.copy(),
                    mode
                )

                comparison_results = (
                    pipeline.compare_preprocessing(
                        image.copy()
                    )
                )

            # ==================================
            # MAIN DETECTION RESULTS
            # ==================================

            st.divider()
            st.header("2. Vehicle Detection and Recognition")

            if not results:

                st.warning(
                    "No supported vehicle detected. "
                    "Try another image or adjust detector settings."
                )

            for i, item in enumerate(results, 1):

                c1, c2 = st.columns(2)

                with c1:

                    st.image(
                        cv2.cvtColor(
                            item["roi"],
                            cv2.COLOR_BGR2RGB
                        ),
                        caption=(
                            f"ROI {i}: {item['class_name']} "
                            f"({item['confidence']:.3f})"
                        ),
                        use_container_width=True
                    )

                with c2:

                    st.metric(
                        "Detection confidence",
                        f"{item['confidence']:.1%}"
                    )

                    st.write(
                        "**Detected type:**",
                        item["class_name"]
                    )

                    st.write("**Top vehicle model predictions**")

                    predictions = item["predictions"]

                    if predictions:

                        for prediction in predictions:

                            st.write(
                                f"{prediction['vehicle_model']} — "
                                f"{prediction['confidence']:.2%}"
                            )

                        top_prediction = predictions[0]

                        st.subheader("Prediction confidence")

                        show_confidence(
                            float(top_prediction["confidence"])
                        )

                        if len(predictions) >= 2:

                            first = float(
                                predictions[0]["confidence"]
                            )

                            second = float(
                                predictions[1]["confidence"]
                            )

                            margin = first - second

                            st.write(
                                "Top-1 vs Top-2 confidence gap: "
                                f"{margin * 100:.2f}%"
                            )

                            if margin < 0.10:

                                st.warning(
                                    "The leading predictions are close. "
                                    "The exact vehicle model may be ambiguous."
                                )

                    else:

                        st.info(
                            item["classification_status"]
                        )

                    st.caption(
                        "Classical feature vector: "
                        f"{item['classical_feature_dimension']} dimensions"
                    )

                    if item["explanation"] is not None:

                        st.write("**Grad-CAM explainability**")

                        st.image(
                            cv2.cvtColor(
                                item["explanation"]["overlay"],
                                cv2.COLOR_BGR2RGB
                            ),
                            caption=(
                                "Grad-CAM overlay — EfficientNet branch "
                                "explanation"
                            ),
                            use_container_width=True
                        )

            # ==================================
            # MODEL PREPROCESSING COMPARISON
            # ==================================

            st.divider()

            st.header(
                "3. Hybrid Model Preprocessing Comparison"
            )

            st.info(
                "The detector bounding box is obtained from the "
                "original image and reused across all four methods. "
                "Each processed ROI is independently classified "
                "by the same trained hybrid model."
            )

            if not comparison_results:

                st.warning(
                    "No car detected for preprocessing comparison."
                )

            for result in comparison_results:

                st.subheader(
                    f"Vehicle {result['detection_index']}"
                )

                st.write(
                    f"YOLO detection confidence: "
                    f"{result['detection_confidence']:.2%}"
                )

                methods = result["methods"]

                rows = []

                for method in [
                    "original",
                    "gaussian",
                    "bilateral",
                    "median"
                ]:

                    if method not in methods:
                        continue

                    prediction = methods[method]

                    rows.append({
                        "Preprocessing": method.title(),
                        "Predicted model": prediction["vehicle_model"],
                        "Top-1 confidence": (
                            f"{prediction['confidence']:.2%}"
                        ),
                        "Top-2 model": prediction["top2_model"],
                        "Top-2 confidence": (
                            f"{prediction['top2_confidence']:.2%}"
                        ),
                        "Confidence gap": (
                            f"{prediction['confidence_gap']:.2%}"
                        )
                    })

                if rows:

                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Show detailed predictions per method.
                    st.subheader("Prediction details")

                    method_columns = st.columns(4)

                    for column, method in zip(
                        method_columns,
                        [
                            "original",
                            "gaussian",
                            "bilateral",
                            "median"
                        ]
                    ):

                        with column:

                            if method not in methods:
                                continue

                            prediction = methods[method]

                            st.markdown(
                                f"**{method.title()}**"
                            )

                            st.image(
                                cv2.cvtColor(
                                    prediction["roi"],
                                    cv2.COLOR_BGR2RGB
                                ),
                                use_container_width=True
                            )

                            st.write(
                                prediction["vehicle_model"]
                            )

                            st.metric(
                                "Confidence",
                                f"{prediction['confidence']:.2%}"
                            )

            st.success(
                "Vehicle recognition and preprocessing comparison completed."
            )

        except FileNotFoundError as e:

            st.error(str(e))

        except Exception as e:

            st.exception(e)


st.divider()

st.caption(
    "System: YOLOv8 detection | ROI extraction | "
    "EfficientNetV2 features | Classical CV features | "
    "Hybrid fusion | Grad-CAM | Preprocessing comparison"
)
