# Vehicle Recognition — P5 integrated starter

This project keeps the P5 master architecture: preprocessing → YOLO detection → ROI extraction → parallel classical-CV and CNN branches → hybrid fusion → multi-task classification → calibration/OOD/explainability → evaluation and mobile optimization.

## Important checkpoint rules
- Never load a checkpoint into a different architecture. EfficientNetV2-RW-S, EfficientNet-B3 and EfficientNet-B0 are distinct.
- Put downloaded checkpoints in `models/pretrained/` and record their source/license in `models/CHECKPOINTS.md`.
- The Stanford Cars 196 class ordering must match the checkpoint's metadata exactly. Do not guess class indices.
- YOLO COCO weights detect the COCO `car` class; they are not a Stanford Cars fine-tuned detector.
- No checkpoint is bundled in this starter. Download from the original repository/model host and verify its hash before use.

## Environment (Mac Apple Silicon)
Use Python 3.11 or 3.12 for the broadest PyTorch/Ultralytics compatibility; Python 3.14 may not have compatible wheels for every dependency.

```bash
cd Vehicle_Recognition
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Install PyTorch using the current official selector if the default pip install does not provide a suitable Apple Silicon build: https://pytorch.org/get-started/locally/

## Dataset
Expected layout:
```
data/stanford_cars/
  car_ims/
  devkit/
  cars_train/   # optional organized image split
  cars_test/    # optional organized image split
```
Keep original Stanford Cars annotations. Do not create random labels for make/model/year. Those heads require verified metadata mapped from the dataset's class names.

## Run
```bash
python -m src.tools.check_environment
streamlit run app/app.py
```
The app currently runs an architecture-ready pipeline. Configure actual checkpoint paths in `config.yaml` before enabling model inference.

## Training order
1. Verify dataset and class mapping.
2. Load pretrained YOLO and verify detection output.
3. Load a matching Stanford Cars classifier checkpoint (or ImageNet backbone for baseline).
4. Fit classical feature vocabulary/SVM on training data only.
5. Train fusion and auxiliary heads; selectively fine-tune CNN.
6. Calibrate on validation only; evaluate on untouched test data.
7. Run ablations, robustness, latency and quantization tests.

## Repository reuse
See `docs/REUSE_MAP.md`. This project contains clean integration interfaces, not copied third-party repository source. Review each upstream license and download checkpoints from their canonical host before reuse.
