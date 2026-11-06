import numpy as np

def extract_roi(image: np.ndarray, bbox, padding: float = 0.05):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = x2-x1, y2-y1
    if bw <= 1 or bh <= 1: raise ValueError("Invalid detection box")
    x1=max(0, int(x1-bw*padding)); y1=max(0, int(y1-bh*padding))
    x2=min(w, int(x2+bw*padding)); y2=min(h, int(y2+bh*padding))
    roi=image[y1:y2, x1:x2]
    if roi.size == 0: raise ValueError("ROI is empty after clipping")
    return roi
