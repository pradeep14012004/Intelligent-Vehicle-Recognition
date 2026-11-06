import numpy as np
from src.roi.extract import extract_roi

def test_roi_clips_to_image():
    image=np.zeros((100,100,3),dtype=np.uint8)
    roi=extract_roi(image,[-10,-10,50,50],0.0)
    assert roi.shape[:2] == (50,50)
