# Upstream reuse map (architecture remains P5)

| Upstream | Reusable contribution | P5 destination | Checkpoint caution |
|---|---|---|---|
| https://github.com/kiko1992-creator/TwinCar | EfficientNetV2 training recipe, augmentation, sampling, Mixup/CutMix, Grad-CAM++ concepts | deep learning / training / explainability | Confirm exact variant and class map; RW-S weights do not load into B0/B3 |
| https://github.com/ZedekiahS/Car-Model-Recognition | YOLO car detection + ROI crop flow | detection / roi | COCO detector is generic car detection, not fine-grained car recognition |
| https://github.com/matkalinowski/EfficientNet-on-Stanford-Cars-Dataset | MATLAB annotation parsing and B0 baseline | data / baseline | Fine-tune on your split; do not claim upstream metrics as yours |
| https://github.com/NugrohoRahmanto/bow-hog-linear-svm-feature-fusion | BoVW + HOG + SVM pattern | classical_cv | Fit vocabulary and SVM on training split only |
| https://github.com/ahmedwalidahmad-debug/car-type-classification | Streamlit UX and Grad-CAM presentation ideas | app | Reimplement against this project's pipeline interfaces |

A separate TwinCar repository/checkpoint may use EfficientNet-B3. Treat it as a distinct model. Before loading any `.pth`, inspect its model config, state-dict keys, number of output classes, class-index file, license and checksum. Never silently substitute one backbone for another.
