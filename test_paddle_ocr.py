#!/usr/bin/env python3
"""
Simple test script to verify TrOCR integration
"""

import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trocr():
    """Test TrOCR with a simple synthetic license plate image"""
    
    # Initialize TrOCR
    logger.info("Initializing TrOCR...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
    
    # Create a simple synthetic license plate image for testing
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
    
    # Add some text using OpenCV
    cv2.putText(img, 'ABC123', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Convert OpenCV image to PIL Image
    logger.info("Converting image format for TrOCR")
    plate_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(plate_rgb)
    
    # Test OCR
    logger.info("Running OCR test with TrOCR...")
    try:
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if generated_text:
            logger.info("OCR successful!")
            # Filter characters to only alphanumeric (license plate characters)
            filtered_text = ''.join(char for char in generated_text if char.isalnum())
            logger.info(f"Detected text: '{filtered_text}'")
        else:
            logger.warning("No text detected")
            
    except Exception as e:
        logger.error(f"OCR test failed: {e}")
        return False
    
    logger.info("TrOCR test completed successfully!")
    return True

if __name__ == "__main__":
    test_trocr()
