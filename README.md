# Intelligent Fine-Grained Vehicle Recognition and Analysis System

**Hybrid Computer Vision · Deep Learning · Fine-Grained Classification · Object Detection**

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-purple)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Under%20Development-yellow)]()

## 1. Overview

This project develops a vehicle recognition system that combines classical computer vision and deep learning to identify fine-grained vehicle classes from images. It uses the Stanford Cars dataset, which contains 16,185 images across 196 car classes (8,144 training and 8,041 test images).

The implemented classification system extracts deep visual representations using EfficientNetV2, extracts handcrafted visual descriptors using classical computer vision, fuses both feature sets, and predicts one of 196 vehicle classes. A YOLOv8n detector pretrained on COCO is also integrated for generic vehicle detection and ROI extraction.

**Important distinction:** the reported accuracy below is from the fine-grained classification evaluation. The current YOLO weights are COCO-pretrained, not fine-tuned on Stanford Cars. Stanford Cars-specific detection training and detection mAP evaluation are still pending.

## 2. Architecture and workflow

```text
                   INPUT IMAGE
                       |
                       v
              IMAGE PREPROCESSING
                       |
             +---------+---------+
             |                   |
             v                   v
       YOLOv8n DETECTOR     CLASSICAL CV
       (COCO pretrained)   FEATURE EXTRACTION
             |             HOG / LBP / HSV /
             v             edge descriptors
        VEHICLE ROI              |
        EXTRACTION               |
             |                   |
             v                   |
       EfficientNetV2             |
       feature extraction         |
             |                   |
             v                   v
        CNN FEATURES        CLASSICAL FEATURES
             |                   |
             +---------+---------+
                       |
                       v
                 FEATURE FUSION
                       |
                       v
               HYBRID CLASSIFIER
                       |
                       v
              196-CLASS PREDICTION
                       |
                       v
             TOP-K + CONFIDENCE
                       |
                       v
                STREAMLIT UI
```

### Classification pipeline

1. Load and preprocess the input image.
2. Extract EfficientNetV2 deep features.
3. Extract handcrafted classical-CV features from the image.
4. Normalize both feature groups using training-set statistics.
5. Concatenate the feature vectors.
6. Pass the fused representation through the trained fusion classifier.
7. Return the top-k class predictions and confidence scores.

### Detection pipeline

The project includes a YOLOv8n wrapper using pretrained COCO weights. It can detect supported COCO vehicle categories (including car, motorcycle, bus and truck) and provide candidate regions for ROI extraction. This is a generic pretrained detector; it has not yet been fine-tuned or evaluated on Stanford Cars bounding-box ground truth.

## 3. Implemented components

### Dataset and class mapping
- Hugging Face integration for `bapei/stanford_cars`.
- 8,144 training images and 8,041 test images.
- 196 fine-grained class labels.
- Class mapping stored in `data/metadata/class_mapping.json`.

The Hugging Face classification dataset exposes image and class label fields. The inspected `segmentationAPIs/standford_cars_masks` dataset exposed a `bbox` field, but every inspected box was `[0, 0, 0, 0]`; those entries are **not valid detection annotations** and are not used as ground truth.

### Deep learning feature extractor
- EfficientNetV2-RW-S backbone.
- Transfer learning and feature extraction.
- 1,792-dimensional pooled CNN feature representation.
- Trained checkpoint: `models/pretrained/efficientnetv2_best_model.pth`.

### Classical computer vision
The feature extraction module combines handcrafted descriptors, including:
- HOG for gradient and shape information.
- LBP for local texture patterns.
- HSV colour histograms.
- Edge and other image descriptors.

The resulting classical feature vector has 8,623 dimensions.

### Hybrid feature fusion
- CNN features: 1,792 dimensions.
- Classical-CV features: 8,623 dimensions.
- Combined representation: 10,415 dimensions.
- Fusion classifier: fully connected layers with batch normalization, GELU activation and dropout.
- Output: 196 vehicle classes.
- Trained checkpoint: `models/trained/hybrid_classifier.pth`.

### Detection and ROI
- YOLOv8n COCO-pretrained detector integration.
- Vehicle class filtering and ROI extraction.
- Detection-to-classification integration exists in the pipeline, but full end-to-end accuracy on detector-generated ROIs has not yet been established.

### Streamlit application
The app includes image upload, detection/ROI workflow, classification output, top-5 predictions, confidence display, preprocessing/feature information and Grad-CAM visualization for the EfficientNetV2 branch. Grad-CAM currently explains the CNN branch, not the full fused decision.

## 4. Evaluation results

### Full Stanford Cars test-set classification

The hybrid classifier was evaluated on all 8,041 test images.

| Metric | Result |
|---|---:|
| Top-1 accuracy | **88.32%** |
| Top-5 accuracy | **97.94%** |
| Macro precision | 88.53% |
| Macro recall | 88.20% |
| Macro F1-score | 88.18% |
| Average inference latency | 42.08 ms/image |
| Median inference latency | 38.74 ms/image |

These are **classification metrics**, not detection metrics. The reported latency is from the classification evaluation setup and should not be interpreted as full detector-plus-classifier end-to-end latency.

### Initial model comparison

An initial comparison was run on a 300-image subset:

| Model | Top-1 accuracy |
|---|---:|
| Classical CV | 3.00% |
| Deep Learning | 87.33% |
| Hybrid | 86.67% |

In this experiment the hybrid model did not outperform the deep-learning baseline. The classical baseline and fusion strategy need further investigation; this result is not presented as evidence of hybrid superiority.

### Initial robustness evaluation

A 300-image subset was evaluated under image degradations:

| Condition | Top-1 accuracy |
|---|---:|
| Original | 86.67% |
| Gaussian blur | 87.00% |
| Gaussian noise | 82.33% |
| Low light | 86.67% |
| JPEG compression | 84.67% |

These are initial subset results, not full-test-set robustness results.

## 5. Technology stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | PyTorch, TorchVision |
| CNN backbone | EfficientNetV2 |
| Object detection | Ultralytics YOLOv8n |
| Computer vision | OpenCV |
| Handcrafted features | HOG, LBP, HSV, edge descriptors |
| Dataset | Stanford Cars |
| Dataset loading | Hugging Face Datasets |
| Application | Streamlit |
| Numerical processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Hardware acceleration | Apple Silicon MPS / CPU |

## 6. Repository layout

```text
Vehicle_Recognition/
├── app/
│   └── app.py
├── data/
│   └── metadata/
│       └── class_mapping.json
├── models/
│   ├── pretrained/
│   │   └── efficientnetv2_best_model.pth
│   ├── trained/
│   │   └── hybrid_classifier.pth
│   └── CHECKPOINTS.md
├── results/
│   ├── evaluation/
│   └── final_analysis/
├── src/
│   ├── classical_cv/
│   ├── deep_learning/
│   ├── detection/
│   ├── evaluation/
│   ├── fusion/
│   ├── preprocessing/
│   ├── roi/
│   ├── training/
│   ├── config.py
│   └── pipeline.py
├── config.yaml
├── requirements.txt
└── README.md
```

## 7. Setup and execution

### Clone

```bash
git clone https://github.com/pradeep14012004/Intelligent-Vehicle-Recognition.git
cd Intelligent-Vehicle-Recognition
```

### Create environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version supported by the installed PyTorch, TorchVision and Ultralytics releases. The development environment used for the current experiments was macOS on Apple Silicon with MPS acceleration.

### Launch Streamlit

```bash
streamlit run app/app.py
```

The application expects the model checkpoints and class mapping at the configured paths. Check `config.yaml` and `models/CHECKPOINTS.md` if local paths differ.

### Load classification data

```python
from datasets import load_dataset

train_data = load_dataset("bapei/stanford_cars", split="train")
test_data = load_dataset("bapei/stanford_cars", split="test")
```

The dataset itself is not committed to this repository. Download it through Hugging Face or follow the dataset provider's terms.

## 8. Remaining work

### A. Stanford Cars object detection
- [ ] Obtain genuine Stanford Cars bounding-box annotations and matching image archives.
- [ ] Validate annotation files and coordinate conventions.
- [ ] Convert valid annotations into YOLO format.
- [ ] Fine-tune YOLOv8n on Stanford Cars.
- [ ] Evaluate on held-out detection annotations using mAP50 and mAP50-95.
- [ ] Report precision, recall and representative detection examples.

### B. End-to-end detection and recognition
- [ ] Integrate the Stanford Cars-trained detector with the hybrid classifier.
- [ ] Validate detector-generated ROI preprocessing against classifier training preprocessing.
- [ ] Evaluate classification on detected/cropped ROIs.
- [ ] Test multiple vehicles in one image.
- [ ] Report detection and classification confidence separately.
- [ ] Add a low-confidence / uncertain-prediction handling strategy.

### C. Optimization and analysis
- [ ] Complete and verify FP16 versus FP32 benchmarking.
- [ ] Compare CPU and Apple Silicon MPS inference.
- [ ] Measure full pipeline latency, peak memory and total deployment size.
- [ ] Conduct controlled feature ablation experiments.
- [ ] Improve and fairly evaluate the classical-CV baseline.
- [ ] Evaluate occlusion, viewing angle, illumination and unseen images.
- [ ] Document at least three representative failure cases.

### D. Application and documentation
- [ ] Validate the complete Streamlit workflow with final detector weights.
- [ ] Display bounding boxes with make/model/year predictions.
- [ ] Add final comparison charts and detection visualizations.
- [ ] Finalize reproducibility instructions and project report.

## 9. Current status

| Component | Status |
|---|---|
| Stanford Cars classification data integration | Implemented |
| EfficientNetV2 feature extractor | Implemented |
| Classical-CV feature extraction | Implemented |
| Hybrid feature fusion and classifier | Trained |
| Full test-set classification evaluation | Completed |
| Initial robustness experiments | Completed on subset |
| COCO-pretrained YOLO integration | Implemented |
| Stanford Cars detector fine-tuning | Pending |
| Valid Stanford Cars detection annotations | Pending |
| Detection mAP evaluation | Pending |
| End-to-end detector + classifier evaluation | Pending |
| Streamlit application | Implemented; final end-to-end validation pending |
| FP16 optimization benchmark | Pending verification |

## 10. Model and data notes

- The repository includes the EfficientNetV2 and hybrid classifier checkpoints currently used by the project.
- Stanford Cars image archives and feature caches are excluded from Git.
- The YOLO COCO checkpoint is not a Stanford Cars fine-tuned detector.
- Do not use the zero-valued bounding boxes from the inspected Hugging Face masks dataset as detection ground truth.
- Keep class-index mapping consistent with the checkpoint and dataset metadata.
- Check upstream dataset and pretrained-weight licenses before redistribution.

## 11. Project goal

The intended final system combines vehicle localization and 196-class fine-grained recognition in one application, returning bounding boxes, vehicle class predictions, confidence scores and visual explanations. The detection training, full end-to-end validation and optimization work listed above remain part of the development roadmap.

**Repository:** https://github.com/pradeep14012004/Intelligent-Vehicle-Recognition

**Project status:** Under active development.
