#!/usr/bin/env python3

import numpy as np
import cv2
from paddleocr import PaddleOCR

def test_paddleocr():
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Create a simple test image
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(img, 'ABC123', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    print("Running OCR test...")
    result = ocr.ocr(img, cls=True)
    
    if result and result[0]:
        print("OCR successful!")
        for line in result[0]:
            if len(line) >= 2 and len(line[1]) >= 2:
                text = line[1][0]
                confidence = line[1][1]
                print(f"Detected text: '{text}' with confidence: {confidence:.3f}")
    else:
        print("No text detected")
    
    print("PaddleOCR test completed successfully!")
    return True

if __name__ == "__main__":
    test_paddleocr()
