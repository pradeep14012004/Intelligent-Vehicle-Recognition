import cv2
import numpy as np


def enhance_image(image: np.ndarray, mode: str = "none") -> np.ndarray:
    """Input/output are BGR uint8 images. Enhancement is optional and configurable."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if mode == "none":
        return image
    if mode == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)
    if mode == "bilateral":
        return cv2.bilateralFilter(image, 7, 50, 50)
    if mode == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if mode == "gamma":
        gamma = 1.15
        lut = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        ).astype("uint8")
        return cv2.LUT(image, lut)
    raise ValueError(f"Unknown enhancement mode: {mode}")


def preprocess_image(image, method="original"):
    """
    Apply an image preprocessing technique.
    Args:
        image: Input image as an RGB NumPy array.
        method: original, gaussian, bilateral, median, clahe.
    Returns:
        Processed RGB image.
    """
    if image is None:
        raise ValueError("Input image cannot be None")

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if method == "original":
        return image.copy()
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(
            image, d=9, sigmaColor=75, sigmaSpace=75
        )
    if method == "median":
        return cv2.medianBlur(image, 5)
    if method == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    raise ValueError(f"Unsupported preprocessing method: {method}")


def get_preprocessing_variants(image):
    """Return the common preprocessing variants for visual comparison."""
    return {
        "original": preprocess_image(image, "original"),
        "gaussian": preprocess_image(image, "gaussian"),
        "bilateral": preprocess_image(image, "bilateral"),
        "median": preprocess_image(image, "median"),
    }

