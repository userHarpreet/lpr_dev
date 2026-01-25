"""
Plate Detector Process Module

Detects license plates in vehicle frames using YOLOv8 and passes detected
plates to downstream processing.
"""

import cv2
import logging
import time
from multiprocessing import Process, Event, Queue
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()

# Load configuration from environment
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "requirements/best28.pt")
PLATE_CONF_MIN = float(os.getenv("PLATE_CONF_MIN", "0.45"))
PLATE_CLASS_ID = int(os.getenv("PLATE_CLASS_ID", "0"))
DETECTION_DEVICE = os.getenv("DETECTION_DEVICE", "cpu")


@dataclass
class PlateDetection:
    """Data structure for a detected license plate."""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def crop_from_image(self, image: 'cv2.Mat') -> 'cv2.Mat':
        """Extract plate region from image."""
        return image[self.y1:self.y2, self.x1:self.x2]
    
    def __repr__(self):
        return (f"PlateDetection(class={self.class_name}, conf={self.confidence:.2f}, "
                f"box=({self.x1},{self.y1})-({self.x2},{self.y2}))")


@dataclass
class FrameWithPlates:
    """Frame data with detected license plates."""
    frame_id: int
    timestamp: datetime
    image: 'cv2.Mat'
    source_name: str
    vehicle_count: int = 0
    plate_detections: List[PlateDetection] = field(default_factory=list)
    
    def __repr__(self):
        return (f"FrameWithPlates(id={self.frame_id}, vehicles={self.vehicle_count}, "
                f"plates={len(self.plate_detections)}, timestamp={self.timestamp})")


class PlateDetectorProcess(Process):
    """
    Process that detects license plates in vehicle frames using YOLOv8.
    
    Reads frames with vehicles from input queue, runs plate detection, and passes
    frames with detected plates to output queue.
    
    Configuration is read from environment variables:
    - PLATE_MODEL_PATH: Path to plate detection model (default: requirements/best28.pt)
    - PLATE_CONF_MIN: Confidence threshold for detections 0.0-1.0 (default: 0.45)
    - PLATE_CLASS_ID: Class ID for license plates (default: 0)
    - DETECTION_DEVICE: Device to use 'cpu' or 'cuda' (default: cpu)
    
    Args:
        in_q: Input queue with FrameWithVehicles objects from VehicleDetectorProcess
        out_q: Output queue to pass FrameWithPlates objects
        stop_event: Event to signal process termination
        model_path: Optional override for model path (uses PLATE_MODEL_PATH if None)
        conf_threshold: Optional override for confidence threshold (uses PLATE_CONF_MIN if None)
        plate_class_id: Optional override for plate class ID (uses PLATE_CLASS_ID if None)
        device: Optional override for device (uses DETECTION_DEVICE if None)
        name: Process name (optional)
    """
    
    def __init__(
        self,
        in_q: Optional[Queue] = None,
        out_q: Optional[Queue] = None,
        stop_event: Optional[Event] = None,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        plate_class_id: Optional[int] = None,
        device: Optional[str] = None,
        name: str = "PlateDetector",
    ):
        super().__init__(name=name)
        self.daemon = False
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        
        # Use environment variables or provided values
        self.model_path = model_path or PLATE_MODEL_PATH
        self.conf_threshold = conf_threshold if conf_threshold is not None else PLATE_CONF_MIN
        self.plate_class_id = plate_class_id if plate_class_id is not None else PLATE_CLASS_ID
        self.device = device or DETECTION_DEVICE
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Statistics
        self.frames_processed = 0
        self.plates_detected = 0
        self.frames_dropped = 0
        self.start_time = None
        
        # Model
        self.model = None
    
    def run(self):
        """Main process loop - detect plates in frames."""
        try:
            # Configure logging for this process
            import sys
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(name)s/%(levelname)s: %(message)s",
                stream=sys.stdout,
                force=True
            )
            self.logger.info(f"[{self.name}] Starting plate detector process")
            self._load_model()
            self._detect_plates()
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error in plate detector: {e}", exc_info=True)
        finally:
            self.logger.info(
                f"[{self.name}] Plate detector stopped. Stats - "
                f"Processed: {self.frames_processed}, "
                f"Plates: {self.plates_detected}, "
                f"Dropped: {self.frames_dropped}"
            )
    
    def _load_model(self):
        """Load plate detection model."""
        try:
            self.logger.info(f"[{self.name}] Loading plate model from: {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"[{self.name}] Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            self.logger.info(
                f"[{self.name}] Plate model loaded successfully. "
                f"Device: {self.device}, Plate Class ID: {self.plate_class_id}"
            )
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to load model: {e}", exc_info=True)
            raise
    
    def _detect_plates(self):
        """Process frames and detect plates."""
        self.start_time = time.time()
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Get frame with vehicles from input queue
                frame_obj = self.in_q.get(timeout=2.0)
                
                # Run plate detection on the frame
                plate_detections = self._run_inference(frame_obj.image)
                
                # Create output frame with plates
                output_frame = FrameWithPlates(
                    frame_id=frame_obj.frame_id,
                    timestamp=frame_obj.timestamp,
                    image=frame_obj.image,
                    source_name=frame_obj.source_name,
                    vehicle_count=len(frame_obj.detections),
                    plate_detections=plate_detections
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
                    self.logger.warning(f"[{self.name}] Failed to put detection in queue: {e}")
                    self.frames_dropped += 1
                    
            except Exception as e:
                # Queue timeout or other error
                if "Empty" not in str(e):
                    self.logger.debug(f"[{self.name}] Queue timeout or error: {e}")
    
    def _run_inference(self, image: 'cv2.Mat') -> List[PlateDetection]:
        """
        Run plate detection inference on frame.
        
        Args:
            image: Input image
            
        Returns:
            List of PlateDetection objects
        """
        try:
            # Run inference
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            plate_detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # Extract detections
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Filter by plate class ID
                        if class_id == self.plate_class_id:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Get class name
                            class_name = self.model.names.get(class_id, "plate")
                            
                            plate_detection = PlateDetection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=confidence,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2
                            )
                            
                            plate_detections.append(plate_detection)
                            self.plates_detected += 1
            
            if len(plate_detections) > 0:
                self.logger.debug(
                    f"[{self.name}] Frame detection: Found {len(plate_detections)} plate(s)"
                )
            
            return plate_detections
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during inference: {e}", exc_info=True)
            return []


class PlateFilter:
    """Utility class for filtering plate detections based on various criteria."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def filter_by_confidence(
        self, detections: List[PlateDetection], min_conf: float = 0.5
    ) -> List[PlateDetection]:
        """Filter detections by confidence threshold."""
        return [d for d in detections if d.confidence >= min_conf]
    
    def filter_by_aspect_ratio(
        self, detections: List[PlateDetection], min_ratio: float = 2.0, max_ratio: float = 5.0
    ) -> List[PlateDetection]:
        """
        Filter detections by aspect ratio (width/height).
        License plates are typically wider than they are tall.
        """
        filtered = []
        for d in detections:
            if d.height > 0:
                ratio = d.width / d.height
                if min_ratio <= ratio <= max_ratio:
                    filtered.append(d)
        return filtered
    
    def filter_by_area(
        self, detections: List[PlateDetection], min_area: int = 500, max_area: int = None
    ) -> List[PlateDetection]:
        """Filter detections by bounding box area."""
        filtered = [d for d in detections if d.area >= min_area]
        if max_area:
            filtered = [d for d in filtered if d.area <= max_area]
        return filtered
    
    def filter_overlapping(
        self, detections: List[PlateDetection], iou_threshold: float = 0.3
    ) -> List[PlateDetection]:
        """Remove overlapping detections, keeping highest confidence ones."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        filtered = []
        for det in sorted_dets:
            keep = True
            for kept_det in filtered:
                iou = self._calculate_iou(det, kept_det)
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                filtered.append(det)
        
        return filtered
    
    @staticmethod
    def _calculate_iou(det1: PlateDetection, det2: PlateDetection) -> float:
        """Calculate Intersection over Union between two detections."""
        # Intersection area
        inter_x1 = max(det1.x1, det2.x1)
        inter_y1 = max(det1.y1, det2.y1)
        inter_x2 = min(det1.x2, det2.x2)
        inter_y2 = min(det1.y2, det2.y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Union area
        area1 = det1.area
        area2 = det2.area
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class PlateVisualizer:
    """Utility class for visualizing plate detections on frames."""
    
    @staticmethod
    def draw_plates(
        image: 'cv2.Mat',
        detections: List[PlateDetection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> 'cv2.Mat':
        """
        Draw bounding boxes and labels on image for detected plates.
        
        Args:
            image: Input image
            detections: List of PlateDetection objects
            color: Box color (BGR)
            thickness: Line thickness
            font_scale: Font scale for text
            
        Returns:
            Image with drawn plate detections
        """
        result = image.copy()
        
        for det in detections:
            # Draw bounding box
            cv2.rectangle(result, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            # Background for text
            cv2.rectangle(
                result,
                (det.x1, det.y1 - label_size[1] - baseline),
                (det.x1 + label_size[0], det.y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                result,
                label,
                (det.x1, det.y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1
            )
        
        return result
    
    @staticmethod
    def extract_plate_regions(
        image: 'cv2.Mat',
        detections: List[PlateDetection],
        padding: int = 5,
    ) -> List[Tuple['cv2.Mat', PlateDetection]]:
        """
        Extract plate regions from image.
        
        Args:
            image: Input image
            detections: List of PlateDetection objects
            padding: Pixels to pad around detection box
            
        Returns:
            List of (cropped_image, detection) tuples
        """
        result = []
        h, w = image.shape[:2]
        
        for det in detections:
            # Add padding
            x1 = max(0, det.x1 - padding)
            y1 = max(0, det.y1 - padding)
            x2 = min(w, det.x2 + padding)
            y2 = min(h, det.y2 + padding)
            
            # Crop
            plate_img = image[y1:y2, x1:x2]
            result.append((plate_img, det))
        
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
    
    # Create detector process
    detector = PlateDetectorProcess(
        in_q=test_in_q,
        out_q=test_out_q,
        stop_event=stop_event,
    )
    
    detector.start()
    
    # Test with a sample image
    test_image = cv2.imread("sample_image.jpg")
    
    if test_image is not None:
        from vehicle_detector import FrameWithVehicles, Detection
        
        frame = FrameWithVehicles(
            frame_id=0,
            timestamp=datetime.now(),
            image=test_image,
            source_name="test",
            vehicle_count=1,
            detections=[
                Detection(
                    class_id=2,
                    class_name="car",
                    confidence=0.95,
                    x1=100,
                    y1=100,
                    x2=400,
                    y2=400
                )
            ]
        )
        
        test_in_q.put(frame)
        
        try:
            result_frame = test_out_q.get(timeout=5.0)
            print(f"Result: {result_frame}")
            print(f"Plates: {result_frame.plate_detections}")
        except:
            print("Timeout waiting for results")
    
    stop_event.set()
    detector.join(timeout=5)
    print("Plate detector stopped")
