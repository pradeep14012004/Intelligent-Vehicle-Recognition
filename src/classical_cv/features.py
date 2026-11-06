import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_classical_features(bgr: np.ndarray) -> np.ndarray:
    """Compact HOG + LBP histogram + HSV histogram + edge density descriptor."""
    image=cv2.resize(bgr, (128,128))
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h=hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    lbp=local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist,_=np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0,10), density=True)
    hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_hist=cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256]).flatten()
    hsv_hist /= (hsv_hist.sum()+1e-8)
    edges=cv2.Canny(gray, 80, 160)
    edge_density=np.array([edges.mean()/255.0], dtype=np.float32)
    return np.concatenate([h.astype(np.float32), lbp_hist.astype(np.float32), hsv_hist.astype(np.float32), edge_density])
