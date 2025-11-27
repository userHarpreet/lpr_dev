"""
Plate Editor Process Module

Enhances and preprocesses license plate images for OCR processing.
Applies various image processing techniques to improve recognition accuracy.
"""

import cv2
import logging
import time
from multiprocessing import Process, Event, Queue
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Load configuration from environment
RESIZE_FACTOR = float(os.getenv("RESIZE_FACTOR", "1.0"))
PLATE_ENHANCE_METHOD = os.getenv("PLATE_ENHANCE_METHOD", "prepare")  # "prepare" or "enhance"
PLATE_CROP_PERCENTAGE = float(os.getenv("PLATE_CROP_PERCENTAGE", "0.11"))  # 11% crop from left by default


@dataclass
class EditedPlate:
    """Data structure for an enhanced/edited plate image."""
    plate_id: int
    frame_id: int
    timestamp: datetime
    original_image: 'cv2.Mat'
    edited_image: 'cv2.Mat'
    enhancement_method: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    def __repr__(self):
        return (f"EditedPlate(id={self.plate_id}, frame={self.frame_id}, "
                f"method={self.enhancement_method}, size={self.width}x{self.height})")


@dataclass
class FrameWithEditedPlates:
    """Frame data with edited/enhanced license plates."""
    frame_id: int
    timestamp: datetime
    source_name: str
    vehicle_count: int = 0
    plate_count: int = 0
    edited_plates: List[EditedPlate] = field(default_factory=list)
    
    def __repr__(self):
        return (f"FrameWithEditedPlates(id={self.frame_id}, vehicles={self.vehicle_count}, "
                f"edited_plates={len(self.edited_plates)})")


class PlateEditorProcess(Process):
    """
    Process that enhances and preprocesses license plate images for OCR.
    
    Reads frames with detected plates from input queue, applies image enhancement
    techniques (resize, grayscale, histogram equalization, filtering, thresholding),
    and passes enhanced plates to output queue.
    
    Configuration is read from environment variables:
    - RESIZE_FACTOR: Multiplier for plate resizing (default: 1.0)
    - PLATE_ENHANCE_METHOD: Enhancement method 'prepare' or 'enhance' (default: prepare)
    
    Args:
        in_q: Input queue with FrameWithPlates objects from PlateDetectorProcess
        out_q: Output queue to pass FrameWithEditedPlates objects
        stop_event: Event to signal process termination
        enhance_method: Optional override for enhancement method
        resize_factor: Optional override for resize factor
        name: Process name (optional)
    """
    
    def __init__(
        self,
        in_q: Optional[Queue] = None,
        out_q: Optional[Queue] = None,
        stop_event: Optional[Event] = None,
        enhance_method: Optional[str] = None,
        resize_factor: Optional[float] = None,
        crop_percentage: Optional[float] = None,
        name: str = "PlateEditor",
    ):
        super().__init__(name=name)
        self.daemon = False
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        
        # Use environment variables or provided values
        self.enhance_method = enhance_method or PLATE_ENHANCE_METHOD
        self.resize_factor = resize_factor if resize_factor is not None else RESIZE_FACTOR
        self.crop_percentage = crop_percentage if crop_percentage is not None else PLATE_CROP_PERCENTAGE
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Statistics
        self.frames_processed = 0
        self.plates_enhanced = 0
        self.frames_dropped = 0
        self.enhancement_errors = 0
        self.start_time = None
    
    def run(self):
        """Main process loop - enhance plate images."""
        try:
            self.logger.info(f"[{self.name}] Starting plate editor process")
            self.logger.info(
                f"[{self.name}] Enhancement method: {self.enhance_method}, "
                f"Resize factor: {self.resize_factor}, "
                f"Crop percentage: {self.crop_percentage*100:.1f}%"
            )
            self._edit_plates()
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error in plate editor: {e}", exc_info=True)
        finally:
            self.logger.info(
                f"[{self.name}] Plate editor stopped. Stats - "
                f"Processed: {self.frames_processed}, "
                f"Enhanced: {self.plates_enhanced}, "
                f"Errors: {self.enhancement_errors}, "
                f"Dropped: {self.frames_dropped}"
            )
    
    def _edit_plates(self):
        """Process frames and enhance plate images."""
        self.start_time = time.time()
        plate_id_counter = 0
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Get frame with plates from input queue
                frame_obj = self.in_q.get(timeout=2.0)
                
                edited_plates = []
                
                # Enhance each detected plate
                for plate_detection in frame_obj.plate_detections:
                    try:
                        # Crop plate region from frame
                        plate_image = plate_detection.crop_from_image(frame_obj.image)
                        
                        # Crop from the left side (remove leading portion)
                        plate_image = self._crop_plate(plate_image)
                        
                        # Enhance/edit the plate image
                        enhanced_image = self._enhance_plate(plate_image)
                        
                        # Create edited plate object
                        edited_plate = EditedPlate(
                            plate_id=plate_id_counter,
                            frame_id=frame_obj.frame_id,
                            timestamp=frame_obj.timestamp,
                            original_image=plate_image,
                            edited_image=enhanced_image,
                            enhancement_method=self.enhance_method,
                            x1=plate_detection.x1,
                            y1=plate_detection.y1,
                            x2=plate_detection.x2,
                            y2=plate_detection.y2,
                            confidence=plate_detection.confidence
                        )
                        
                        edited_plates.append(edited_plate)
                        self.plates_enhanced += 1
                        plate_id_counter += 1
                        
                    except Exception as e:
                        self.logger.warning(
                            f"[{self.name}] Failed to enhance plate in frame {frame_obj.frame_id}: {e}"
                        )
                        self.enhancement_errors += 1
                
                # Create output frame with edited plates
                output_frame = FrameWithEditedPlates(
                    frame_id=frame_obj.frame_id,
                    timestamp=frame_obj.timestamp,
                    source_name=frame_obj.source_name,
                    vehicle_count=frame_obj.vehicle_count,
                    plate_count=len(frame_obj.plate_detections),
                    edited_plates=edited_plates
                )
                
                # Put in output queue
                try:
                    if self.out_q.full():
                        self.frames_dropped += 1
                        self.logger.debug(
                            f"[{self.name}] Output queue full, dropped frame {frame_obj.frame_id}"
                        )
                    else:
                        self.out_q.put(output_frame, timeout=1.0)
                        self.frames_processed += 1
                except Exception as e:
                    self.logger.warning(f"[{self.name}] Failed to put edited plates in queue: {e}")
                    self.frames_dropped += 1
                    
            except Exception as e:
                # Queue timeout or other error
                if "Empty" not in str(e):
                    self.logger.debug(f"[{self.name}] Queue timeout or error: {e}")
    
    def _crop_plate(self, image: 'cv2.Mat') -> 'cv2.Mat':
        """
        Crop plate image from the left side to remove leading portion.
        
        Useful for removing country/state markers or excess space on license plates.
        
        Args:
            image: Plate image to crop
            
        Returns:
            Cropped plate image
        """
        try:
            h, w = image.shape[:2]
            
            # Calculate crop amount from left
            crop_width = int(w * self.crop_percentage)
            
            # Crop from left: (crop_width, 0) to (w, h)
            cropped = image[:, crop_width:w]
            
            self.logger.debug(
                f"[{self.name}] Cropped plate: {w}x{h} -> {cropped.shape[1]}x{cropped.shape[0]} "
                f"(removed {crop_width}px from left)"
            )
            
            return cropped
            
        except Exception as e:
            self.logger.warning(f"[{self.name}] Error cropping plate, returning original: {e}")
            return image
    
    def _enhance_plate(self, image: 'cv2.Mat') -> 'cv2.Mat':
        """
        Enhance plate image for OCR using selected method.
        
        Args:
            image: Plate image to enhance
            
        Returns:
            Enhanced plate image
        """
        try:
            if self.enhance_method == "prepare":
                return self._prepare_plate_for_ocr(image)
            elif self.enhance_method == "enhance":
                return self._enhance_plate_classic(image)
            else:
                self.logger.warning(
                    f"[{self.name}] Unknown enhancement method: {self.enhance_method}, using prepare"
                )
                return self._prepare_plate_for_ocr(image)
        except Exception as e:
            self.logger.error(f"[{self.name}] Error enhancing plate: {e}", exc_info=True)
            # Fallback: ensure 3-channel image
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
    
    def _prepare_plate_for_ocr(self, image: 'cv2.Mat') -> 'cv2.Mat':
        """
        Prepare plate image for OCR: resize, grayscale, CLAHE, bilateral filter,
        and adaptive thresholding. Returns 3-channel BGR image.
        
        This method is more conservative and preserves detail well.
        """
        try:
            # Resize to reasonable working size keeping aspect
            h, w = image.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)
            
            # Bilateral filter: reduce noise while preserving edges
            filtered = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Adaptive thresholding for better character separation
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to BGR for downstream consumers
            ocr_ready = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            return ocr_ready
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error in _prepare_plate_for_ocr: {e}")
            raise
    
    def _enhance_plate_classic(self, image: 'cv2.Mat') -> 'cv2.Mat':
        """
        Classic enhancement method: resize, grayscale, blur, threshold,
        morphological operations. Returns 3-channel BGR image.
        
        This method is more aggressive and creates high-contrast binary images.
        """
        try:
            # Resize with aspect ratio preservation
            resized_img = self._resize_plate(image, self.resize_factor)
            
            # Convert to grayscale
            gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            # Gaussian blur
            blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
            
            # Binary threshold (Otsu's method)
            _, binary_img = cv2.threshold(
                blurred_img, 200, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Morphological operations
            eroded_img = cv2.erode(binary_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dilated_img = cv2.dilate(eroded_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            
            # Convert back to BGR (3-channel)
            bgr_img = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
            return bgr_img
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error in _enhance_plate_classic: {e}")
            raise
    
    @staticmethod
    def _resize_plate(image: 'cv2.Mat', multiplier: float) -> 'cv2.Mat':
        """
        Resize plate image with aspect ratio preservation.
        
        Args:
            image: Input image
            multiplier: Resize multiplier
            
        Returns:
            Resized image
        """
        if multiplier == 1.0:
            return image
        
        if multiplier <= 0:
            return image
        
        h, w = image.shape[:2]
        aspect_ratio = w / h
        new_height = int(h * multiplier)
        new_width = int(new_height * aspect_ratio)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


class PlateEnhancementConfig:
    """Configuration utility for plate enhancement settings."""
    
    # Enhancement method presets
    PRESETS = {
        "light": {
            "method": "prepare",
            "resize_factor": 1.0,
            "description": "Light enhancement for clear images"
        },
        "medium": {
            "method": "prepare",
            "resize_factor": 1.5,
            "description": "Medium enhancement with slight upscaling"
        },
        "heavy": {
            "method": "enhance",
            "resize_factor": 2.0,
            "description": "Heavy enhancement with upscaling and morphological ops"
        },
        "ocr_optimized": {
            "method": "prepare",
            "resize_factor": 1.2,
            "description": "Optimized for OCR accuracy"
        },
    }
    
    @staticmethod
    def get_preset(preset_name: str) -> dict:
        """Get preset configuration."""
        return PlateEnhancementConfig.PRESETS.get(
            preset_name,
            PlateEnhancementConfig.PRESETS["medium"]
        )


class PlateEnhancementVisualizer:
    """Utility for visualizing enhancement results."""
    
    @staticmethod
    def compare_enhancements(original: 'cv2.Mat', edited: 'cv2.Mat') -> 'cv2.Mat':
        """
        Create a side-by-side comparison of original and enhanced plate.
        
        Args:
            original: Original plate image
            edited: Enhanced plate image
            
        Returns:
            Combined comparison image
        """
        try:
            # Ensure same height
            h1, w1 = original.shape[:2]
            h2, w2 = edited.shape[:2]
            
            target_h = max(h1, h2)
            
            if h1 != target_h:
                original = cv2.resize(original, (int(w1 * target_h / h1), target_h))
            if h2 != target_h:
                edited = cv2.resize(edited, (int(w2 * target_h / h2), target_h))
            
            # Concatenate horizontally
            comparison = cv2.hconcat([original, edited])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (10, 25), font, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, "Enhanced", (comparison.shape[1]//2 + 10, 25), font, 0.7, (0, 255, 0), 2)
            
            return comparison
        except Exception as e:
            return original
    
    @staticmethod
    def draw_enhancement_info(image: 'cv2.Mat', method: str, confidence: float) -> 'cv2.Mat':
        """Add enhancement method and confidence info to image."""
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        info_text = f"Method: {method} | Conf: {confidence:.2f}"
        cv2.putText(result, info_text, (10, 30), font, 0.6, (0, 255, 0), 2)
        
        return result


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
    )
    
    # Create test queue and event
    test_in_q = Queue(maxsize=10)
    test_out_q = Queue(maxsize=10)
    stop_event = Event()
    
    # Create editor process
    editor = PlateEditorProcess(
        in_q=test_in_q,
        out_q=test_out_q,
        stop_event=stop_event,
    )
    
    editor.start()
    
    # Test with a sample image
    test_image = cv2.imread("sample_image.jpg")
    
    if test_image is not None:
        from plate_detector import FrameWithPlates, PlateDetection
        
        frame = FrameWithPlates(
            frame_id=0,
            timestamp=datetime.now(),
            image=test_image,
            source_name="test",
            vehicle_count=1,
            plate_detections=[
                PlateDetection(
                    class_id=0,
                    class_name="plate",
                    confidence=0.95,
                    x1=100,
                    y1=100,
                    x2=300,
                    y2=150
                )
            ]
        )
        
        test_in_q.put(frame)
        
        try:
            result_frame = test_out_q.get(timeout=5.0)
            print(f"Result: {result_frame}")
            print(f"Edited plates: {result_frame.edited_plates}")
        except:
            print("Timeout waiting for results")
    
    stop_event.set()
    editor.join(timeout=5)
    print("Plate editor stopped")
