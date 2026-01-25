"""
OCR Process Module

Performs Optical Character Recognition on license plate images using PaddleOCR.
"""

import cv2
import logging
import time
from multiprocessing import Process, Event, Queue
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from paddleocr import PaddleOCR
import os
from dotenv import load_dotenv
import psutil

load_dotenv()

# Load configuration from environment
OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "True") == "True"
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "en")
OCR_DEVICE = os.getenv("OCR_DEVICE", "cpu")


@dataclass
class OCRResult:
    """Data structure for OCR results."""
    plate_id: int
    frame_id: int
    timestamp: datetime
    raw_text: Optional[str]
    confidence: Optional[float]
    valid: bool = False
    corrected_text: Optional[str] = None
    detection_method: str = "primary"  # primary, upscaled, rotated, original
    
    def __repr__(self):
        return (f"OCRResult(plate={self.plate_id}, raw='{self.raw_text}', "
                f"conf={self.confidence:.2f if self.confidence else 'None'}, "
                f"corrected='{self.corrected_text}')")


@dataclass
class FrameWithOCR:
    """Frame data with OCR results for each plate."""
    frame_id: int
    timestamp: datetime
    source_name: str
    vehicle_count: int = 0
    plate_count: int = 0
    ocr_results: List[OCRResult] = field(default_factory=list)
    
    def __repr__(self):
        return (f"FrameWithOCR(id={self.frame_id}, vehicles={self.vehicle_count}, "
                f"plates={self.plate_count}, ocr_results={len(self.ocr_results)})")


class OCRProcess(Process):
    """
    Process that performs OCR on license plate images using PaddleOCR.
    
    Reads frames with edited plates from input queue, runs OCR recognition,
    and passes frames with OCR results to output queue.
    
    Configuration is read from environment variables:
    - OCR_USE_ANGLE_CLS: Use angle classification (default: True)
    - OCR_LANGUAGE: OCR language (default: en)
    - OCR_DEVICE: Device to use 'cpu' or 'cuda' (default: cpu)
    
    Args:
        in_q: Input queue with FrameWithEditedPlates objects from PlateEditorProcess
        out_q: Output queue to pass FrameWithOCR objects
        stop_event: Event to signal process termination
        use_angle_cls: Optional override for angle classification
        language: Optional override for OCR language
        device: Optional override for device
        name: Process name (optional)
    """
    
    def __init__(
        self,
        in_q: Optional[Queue] = None,
        out_q: Optional[Queue] = None,
        stop_event: Optional[Event] = None,
        use_angle_cls: Optional[bool] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        name: str = "OCR",
    ):
        super().__init__(name=name)
        self.daemon = False
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        
        # Use environment variables or provided values
        self.use_angle_cls = use_angle_cls if use_angle_cls is not None else OCR_USE_ANGLE_CLS
        self.language = language or OCR_LANGUAGE
        self.device = device or OCR_DEVICE
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Statistics
        self.frames_processed = 0
        self.plates_recognized = 0
        self.recognition_errors = 0
        self.frames_dropped = 0
        self.start_time = None
        
        # Model
        self.ocr_model = None
    
    def run(self):
        """Main process loop - perform OCR on plates."""
        try:
            # Configure logging for this process
            import sys
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(name)s/%(levelname)s: %(message)s",
                stream=sys.stdout,
                force=True
            )
            self.logger.info(f"[{self.name}] Starting OCR process")
            self._load_model()
            self._recognize_plates()
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error in OCR: {e}", exc_info=True)
        finally:
            self.logger.info(
                f"[{self.name}] OCR process stopped. Stats - "
                f"Processed: {self.frames_processed}, "
                f"Recognized: {self.plates_recognized}, "
                f"Errors: {self.recognition_errors}, "
                f"Dropped: {self.frames_dropped}"
            )
    
    def _load_model(self):
        """Load PaddleOCR model."""
        try:
            self.logger.info(
                f"[{self.name}] Loading PaddleOCR model. "
                f"Language: {self.language}, Angle classification: {self.use_angle_cls}, "
                f"Device: {self.device}"
            )
            
            # Load model
            self.ocr_model = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.language
            )
            
            self.logger.info(f"[{self.name}] PaddleOCR model loaded successfully")
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to load OCR model: {e}", exc_info=True)
            raise
    
    def _recognize_plates(self):
        """Process frames and recognize plate text."""
        self.start_time = time.time()
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Get frame with edited plates from input queue
                frame_obj = self.in_q.get(timeout=2.0)
                
                ocr_results = []
                
                # Recognize each edited plate
                for idx, edited_plate in enumerate(frame_obj.edited_plates):
                    try:
                        # Run OCR on plate image
                        raw_text, confidence, method = self._run_ocr(edited_plate.edited_image)
                        
                        # Create OCR result object
                        ocr_result = OCRResult(
                            plate_id=edited_plate.plate_id,
                            frame_id=frame_obj.frame_id,
                            timestamp=frame_obj.timestamp,
                            raw_text=raw_text,
                            confidence=confidence,
                            detection_method=method
                        )
                        
                        ocr_results.append(ocr_result)
                        self.plates_recognized += 1
                        
                        if raw_text:
                            self.logger.debug(
                                f"[{self.name}] Plate {edited_plate.plate_id}: "
                                f"Text='{raw_text}', Conf={confidence:.2f}, Method={method}"
                            )
                        
                    except Exception as e:
                        self.logger.warning(
                            f"[{self.name}] Failed to recognize plate {edited_plate.plate_id}: {e}"
                        )
                        self.recognition_errors += 1
                
                # Create output frame with OCR results
                output_frame = FrameWithOCR(
                    frame_id=frame_obj.frame_id,
                    timestamp=frame_obj.timestamp,
                    source_name=frame_obj.source_name,
                    vehicle_count=frame_obj.vehicle_count,
                    plate_count=frame_obj.plate_count,
                    ocr_results=ocr_results
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
                    self.logger.warning(f"[{self.name}] Failed to put OCR results in queue: {e}")
                    self.frames_dropped += 1
                    
            except Exception as e:
                # Queue timeout or other error
                if "Empty" not in str(e):
                    self.logger.debug(f"[{self.name}] Queue timeout or error: {e}")
    
    def _run_ocr(self, image: 'cv2.Mat') -> Tuple[Optional[str], Optional[float], str]:
        """
        Run OCR on plate image with fallback strategies.
        
        Args:
            image: Plate image to recognize
            
        Returns:
            Tuple of (text, confidence, detection_method)
        """
        try:
            # Ensure 3-channel image
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            start_time = time.time()
            mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            # Primary OCR attempt
            result = self.ocr_model.ocr(image, cls=self.use_angle_cls)
            
            elapsed = time.time() - start_time
            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            self.logger.debug(
                f"[{self.name}] OCR completed in {elapsed:.2f}s, "
                f"Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB"
            )
            
            # Parse OCR result
            text, confidence = self._parse_ocr_result(result)
            
            if text:
                return text, confidence, "primary"
            
            # Fallback 1: Try upscaled image
            try:
                text, confidence = self._try_upscaled_ocr(image)
                if text:
                    return text, confidence, "upscaled"
            except Exception as e:
                self.logger.debug(f"[{self.name}] Upscaled OCR failed: {e}")
            
            # Fallback 2: Try rotated image if available in result
            try:
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    rot_img = result[0].get('rot_img')
                    if rot_img is not None:
                        result_rot = self.ocr_model.ocr(rot_img, cls=self.use_angle_cls)
                        text, confidence = self._parse_ocr_result(result_rot)
                        if text:
                            return text, confidence, "rotated"
            except Exception as e:
                self.logger.debug(f"[{self.name}] Rotated OCR failed: {e}")
            
            # If all attempts failed, return empty result
            return None, None, "failed"
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during OCR: {e}", exc_info=True)
            return None, None, "error"
    
    def _parse_ocr_result(self, result) -> Tuple[Optional[str], Optional[float]]:
        """
        Parse PaddleOCR result and extract text and confidence.
        
        Handles multiple output formats from PaddleOCR.
        
        Args:
            result: PaddleOCR result
            
        Returns:
            Tuple of (text, confidence) or (None, None) if parsing fails
        """
        try:
            if not result or not isinstance(result, list) or len(result) == 0:
                return None, None
            
            lines = result[0] if isinstance(result[0], list) else [result[0]]
            
            texts = []
            confs = []
            
            for line in lines:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    # Format: [box, [text, conf]]
                    text_data = line[1]
                    if isinstance(text_data, (list, tuple)) and len(text_data) >= 1:
                        text = str(text_data[0]) if text_data[0] else ""
                        conf = float(text_data[1]) if len(text_data) > 1 and text_data[1] is not None else 0.0
                        
                        # Filter: keep only alphanumeric
                        filtered_text = ''.join(c for c in text if c.isalnum())
                        
                        if filtered_text:
                            texts.append(filtered_text)
                            confs.append(conf)
            
            if texts:
                combined_text = ''.join(texts)
                avg_confidence = sum(confs) / len(confs) if confs else 0.0
                return combined_text, avg_confidence
            
            return None, None
            
        except Exception as e:
            self.logger.debug(f"[{self.name}] Error parsing OCR result: {e}")
            return None, None
    
    def _try_upscaled_ocr(self, image: 'cv2.Mat') -> Tuple[Optional[str], Optional[float]]:
        """
        Try OCR on upscaled image for small plates.
        
        Args:
            image: Plate image
            
        Returns:
            Tuple of (text, confidence) or (None, None)
        """
        h, w = image.shape[:2]
        min_h = 128
        min_w = 320
        
        if h >= min_h and w >= min_w:
            return None, None
        
        scale = max(min_h / max(1, h), min_w / max(1, w))
        upscaled = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        self.logger.debug(f"[{self.name}] Attempting OCR on upscaled image (scale={scale:.2f})")
        
        result = self.ocr_model.ocr(upscaled, cls=self.use_angle_cls)
        return self._parse_ocr_result(result)


class PlateValidator:
    """Utility class for validating and correcting OCR results."""
    
    @staticmethod
    def validate_plate(text: str) -> Tuple[bool, str]:
        """
        Validate if text looks like a license plate.
        
        Args:
            text: Recognized text
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not text:
            return False, "Empty text"
        
        # Remove spaces and convert to uppercase
        text = text.replace(" ", "").upper()
        
        # Basic check: should have alphanumeric characters
        if not any(c.isalnum() for c in text):
            return False, "No alphanumeric characters"
        
        # Length check: typical license plates are 6-10 characters
        if len(text) < 4:
            return False, "Too short (< 4 chars)"
        
        if len(text) > 12:
            return False, "Too long (> 12 chars)"
        
        return True, "Valid"
    
    @staticmethod
    def correct_common_ocr_errors(text: str) -> str:
        """
        Correct common OCR mistakes (e.g., 0 vs O, 1 vs I, 8 vs B).
        
        Args:
            text: Raw OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Common substitutions - customize based on your plate format
        corrections = {
            '0': 'O',  # Zero to letter O (if more likely)
            # Add more corrections as needed
        }
        
        # Keep original for now - can be enhanced based on plate format
        return text


class OCRResultVisualizer:
    """Utility class for visualizing OCR results."""
    
    @staticmethod
    def draw_ocr_result(
        image: 'cv2.Mat',
        text: Optional[str],
        confidence: Optional[float],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> 'cv2.Mat':
        """
        Draw OCR result on plate image.
        
        Args:
            image: Plate image
            text: Recognized text
            confidence: Confidence score
            color: Text color (BGR)
            
        Returns:
            Image with OCR result overlay
        """
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw text
        text_to_draw = f"{text}" if text else "No text detected"
        if confidence is not None:
            text_to_draw += f" ({confidence:.2f})"
        
        cv2.putText(result, text_to_draw, (10, 30), font, 0.8, color, 2)
        
        return result
    
    @staticmethod
    def create_ocr_report(ocr_results: List[OCRResult]) -> str:
        """
        Create a formatted report of OCR results.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            Formatted report string
        """
        report = "OCR Results:\n"
        report += "=" * 60 + "\n"
        
        for result in ocr_results:
            report += f"Plate ID: {result.plate_id}\n"
            report += f"  Text: {result.raw_text or 'Not detected'}\n"
            report += f"  Confidence: {result.confidence:.2f if result.confidence else 'N/A'}\n"
            report += f"  Method: {result.detection_method}\n"
            report += "-" * 60 + "\n"
        
        return report


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
    
    # Create OCR process
    ocr_proc = OCRProcess(
        in_q=test_in_q,
        out_q=test_out_q,
        stop_event=stop_event,
    )
    
    ocr_proc.start()
    
    # Test with a sample image
    test_image = cv2.imread("sample_plate.jpg")
    
    if test_image is not None:
        from plate_editor import FrameWithEditedPlates, EditedPlate
        
        frame = FrameWithEditedPlates(
            frame_id=0,
            timestamp=datetime.now(),
            source_name="test",
            vehicle_count=1,
            plate_count=1,
            edited_plates=[
                EditedPlate(
                    plate_id=0,
                    frame_id=0,
                    timestamp=datetime.now(),
                    original_image=test_image,
                    edited_image=test_image,
                    enhancement_method="prepare",
                    x1=0,
                    y1=0,
                    x2=300,
                    y2=150,
                    confidence=0.95
                )
            ]
        )
        
        test_in_q.put(frame)
        
        try:
            result_frame = test_out_q.get(timeout=10.0)
            print(f"Result: {result_frame}")
            print(f"OCR Results: {result_frame.ocr_results}")
        except Exception as e:
            print(f"Timeout waiting for results: {e}")
    
    stop_event.set()
    ocr_proc.join(timeout=5)
    print("OCR process stopped")
