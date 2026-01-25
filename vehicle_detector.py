"""
Vehicle Detector Process Module

Detects vehicles in frames using YOLOv8 and passes detected vehicles
to downstream processing.
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
VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", "requirements/yolo12n.pt")
VEHICLE_CONF_MIN = float(os.getenv("VEHICLE_CONF_MIN", "0.45"))
VEHICLE_CLASSES = [int(cls) for cls in os.getenv("VEHICLE_CLASSES", "2,5,7").split(",")]
DETECTION_DEVICE = os.getenv("DETECTION_DEVICE", "cpu")


@dataclass
class Detection:
    """Data structure for a detected object."""
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
    
    def __repr__(self):
        return (f"Detection(class={self.class_name}, conf={self.confidence:.2f}, "
                f"box=({self.x1},{self.y1})-({self.x2},{self.y2}))")


@dataclass
class FrameWithVehicles:
    """Frame data with detected vehicles."""
    frame_id: int
    timestamp: datetime
    image: 'cv2.Mat'
    source_name: str
    detections: List[Detection] = field(default_factory=list)
    
    def __repr__(self):
        return (f"FrameWithVehicles(id={self.frame_id}, vehicles={len(self.detections)}, "
                f"timestamp={self.timestamp})")


class VehicleDetectorProcess(Process):
    """
    Process that detects vehicles in frames using YOLOv8.
    
    Reads frames from input queue, runs vehicle detection, and passes
    frames with detected vehicles to output queue.
    
    Configuration is read from environment variables:
    - VEHICLE_MODEL_PATH: Path to YOLOv8 model weights (default: requirements/yolo12n.pt)
    - VEHICLE_CONF_MIN: Confidence threshold for detections 0.0-1.0 (default: 0.45)
    - VEHICLE_CLASSES: Comma-separated vehicle class IDs (default: 2,5,7 = car,bus,truck)
    - DETECTION_DEVICE: Device to use 'cpu' or 'cuda' (default: cpu)
    
    Args:
        in_q: Input queue with Frame objects from VideoReaderProcess
        out_q: Output queue to pass FrameWithVehicles objects
        stop_event: Event to signal process termination
        model_path: Optional override for model path (uses VEHICLE_MODEL_PATH if None)
        conf_threshold: Optional override for confidence threshold (uses VEHICLE_CONF_MIN if None)
        vehicle_classes: Optional override for vehicle classes (uses VEHICLE_CLASSES if None)
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
        vehicle_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
        name: str = "VehicleDetector",
    ):
        super().__init__(name=name)
        self.daemon = False
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        
        # Use environment variables or provided values
        self.model_path = model_path or VEHICLE_MODEL_PATH
        self.conf_threshold = conf_threshold if conf_threshold is not None else VEHICLE_CONF_MIN
        self.vehicle_classes = vehicle_classes or VEHICLE_CLASSES
        self.device = device or DETECTION_DEVICE
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Statistics
        self.frames_processed = 0
        self.vehicles_detected = 0
        self.frames_dropped = 0
        self.start_time = None
        
        # Model
        self.model = None
    
    def run(self):
        """Main process loop - detect vehicles in frames."""
        try:
            # Force logging configuration in child process
            import sys
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(name)s/%(levelname)s: %(message)s",
                stream=sys.stdout
            )
            self.logger.info(f"[{self.name}] Starting vehicle detector process")
            self._load_model()
            self._detect_vehicles()
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error in vehicle detector: {e}", exc_info=True)
        finally:
            self.logger.info(
                f"[{self.name}] Vehicle detector stopped. Stats - "
                f"Processed: {self.frames_processed}, "
                f"Vehicles: {self.vehicles_detected}, "
                f"Dropped: {self.frames_dropped}"
            )
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            self.logger.info(f"[{self.name}] Loading YOLO model from: {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"[{self.name}] Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            self.logger.info(
                f"[{self.name}] YOLO model loaded successfully. "
                f"Device: {self.device}, Classes: {self.vehicle_classes}"
            )
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to load model: {e}", exc_info=True)
            raise
    
    def _detect_vehicles(self):
        """Process frames and detect vehicles."""
        self.start_time = time.time()
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Get frame from input queue
                frame_obj = self.in_q.get(timeout=2.0)
                
                # Run detection
                detections = self._run_inference(frame_obj)
                
                # Create output frame with detections
                output_frame = FrameWithVehicles(
                    frame_id=frame_obj.frame_id,
                    timestamp=frame_obj.timestamp,
                    image=frame_obj.data,
                    source_name=frame_obj.source_name,
                    detections=detections
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
    
    def _run_inference(self, frame_obj) -> List[Detection]:
        """
        Run YOLO inference on frame.
        
        Args:
            frame_obj: Frame object with image data
            
        Returns:
            List of Detection objects
        """
        try:
            # Run inference with timing
            inf_start = time.time()
            results = self.model(frame_obj.data, conf=self.conf_threshold, verbose=False)
            inf_duration = time.time() - inf_start
            
            # Log slow inference (likely on emulation)
            self.logger.info(
                f"[{self.name}] Inference time: {inf_duration:.3f}s for frame {frame_obj.frame_id}"
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # Extract detections
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Filter by vehicle classes
                        if class_id in self.vehicle_classes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Get class name
                            class_name = self.model.names[class_id]
                            
                            detection = Detection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=confidence,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2
                            )
                            
                            detections.append(detection)
                            self.vehicles_detected += 1
            
            if len(detections) > 0:
                self.logger.debug(
                    f"[{self.name}] Frame {frame_obj.frame_id}: "
                    f"Detected {len(detections)} vehicle(s)"
                )
            
            return detections
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during inference: {e}", exc_info=True)
            return []


class VehicleFilter:
    """Utility class for filtering detections based on various criteria."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def filter_by_confidence(
        self, detections: List[Detection], min_conf: float = 0.5
    ) -> List[Detection]:
        """Filter detections by confidence threshold."""
        return [d for d in detections if d.confidence >= min_conf]
    
    def filter_by_area(
        self, detections: List[Detection], min_area: int = 100, max_area: int = None
    ) -> List[Detection]:
        """Filter detections by bounding box area."""
        filtered = [d for d in detections if d.area >= min_area]
        if max_area:
            filtered = [d for d in filtered if d.area <= max_area]
        return filtered
    
    def filter_by_position(
        self, detections: List[Detection], region: Tuple[int, int, int, int] = None
    ) -> List[Detection]:
        """
        Filter detections by region of interest.
        
        Args:
            detections: List of Detection objects
            region: (x1, y1, x2, y2) region bounds
            
        Returns:
            Detections within specified region
        """
        if not region:
            return detections
        
        x1, y1, x2, y2 = region
        filtered = []
        
        for d in detections:
            # Check if detection center is in region
            cx, cy = d.center
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                filtered.append(d)
        
        return filtered
    
    def filter_overlapping(
        self, detections: List[Detection], iou_threshold: float = 0.3
    ) -> List[Detection]:
        """Remove overlapping detections, keeping highest confidence ones."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        filtered = []
        for i, det in enumerate(sorted_dets):
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
    def _calculate_iou(det1: Detection, det2: Detection) -> float:
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


class DetectionVisualizer:
    """Utility class for visualizing detections on frames."""
    
    @staticmethod
    def draw_detections(
        image: 'cv2.Mat',
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> 'cv2.Mat':
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image
            detections: List of Detection objects
            color: Box color (BGR)
            thickness: Line thickness
            font_scale: Font scale for text
            
        Returns:
            Image with drawn detections
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
    def draw_region_of_interest(
        image: 'cv2.Mat',
        region: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> 'cv2.Mat':
        """Draw region of interest rectangle on image."""
        result = image.copy()
        x1, y1, x2, y2 = region
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
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
    detector = VehicleDetectorProcess(
        in_q=test_in_q,
        out_q=test_out_q,
        stop_event=stop_event,
    )
    
    detector.start()
    
    # Test with a sample image
    test_image = cv2.imread("sample_image.jpg")
    
    if test_image is not None:
        from video_reader import Frame
        
        frame = Frame(
            data=test_image,
            timestamp=datetime.now(),
            frame_id=0,
            source_name="test"
        )
        
        test_in_q.put(frame)
        
        try:
            result_frame = test_out_q.get(timeout=5.0)
            print(f"Result: {result_frame}")
            print(f"Detections: {result_frame.detections}")
        except:
            print("Timeout waiting for results")
    
    stop_event.set()
    detector.join(timeout=5)
    print("Vehicle detector stopped")
