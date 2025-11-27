"""
Video Reader Process Module

Handles reading frames from video sources (file, camera, or stream) and
passes them to downstream processing queues.
"""

import cv2
import logging
import time
import threading
from multiprocessing import Process, Event, Queue
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Frame:
    """Data structure for passing frame data through the pipeline."""
    data: 'cv2.Mat'
    timestamp: datetime
    frame_id: int
    source_name: str
    
    def __repr__(self):
        return (f"Frame(id={self.frame_id}, timestamp={self.timestamp}, "
                f"source={self.source_name}, shape={self.data.shape})")


class VideoReaderProcess(Process):
    """
    Process that reads video frames from various sources and passes them
    to a queue for downstream processing.
    
    Supports:
    - Video files (MP4, AVI, MOV, etc.)
    - USB cameras / Webcams (index 0, 1, 2, etc.)
    - IP camera streams (RTSP, HTTP, etc.)
    - Local network streams
    
    Args:
        source: Path to video file, camera index (int), or stream URL (str)
        out_q: Output queue to pass Frame objects
        stop_event: Event to signal process termination
        name: Process name (optional)
        fps_limit: Maximum FPS to read (None = unlimited)
        skip_frames: Number of frames to skip (for efficiency)
    """
    
    def __init__(
        self,
        source: Union[str, int] = 0,
        out_q: Optional[Queue] = None,
        stop_event: Optional[Event] = None,
        name: str = "VideoReader",
        fps_limit: Optional[float] = None,
        skip_frames: int = 0,
    ):
        super().__init__(name=name)
        self.daemon = False
        self.source = source
        self.out_q = out_q
        self.stop_event = stop_event
        self.fps_limit = fps_limit
        self.skip_frames = skip_frames
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Frame tracking
        self.frame_count = 0
        self.skipped_count = 0
        self.dropped_count = 0
        self.start_time = None
        
    def run(self):
        """Main process loop - read and distribute frames."""
        try:
            self.logger.info(f"[{self.name}] Starting video reader with source: {self.source}")
            self._read_video()
        except Exception as e:
            self.logger.error(f"[{self.name}] Fatal error in video reader: {e}", exc_info=True)
        finally:
            self.logger.info(
                f"[{self.name}] Video reader stopped. Stats - "
                f"Read: {self.frame_count}, Skipped: {self.skipped_count}, "
                f"Dropped: {self.dropped_count}"
            )
    
    def _read_video(self):
        """Open video source and process frames."""
        cap = self._open_source()
        if cap is None:
            self.logger.error(f"[{self.name}] Failed to open video source")
            return
        
        try:
            self.start_time = time.time()
            frame_skip_counter = 0
            
            while not (self.stop_event and self.stop_event.is_set()):
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.info(f"[{self.name}] End of video stream or failed to read frame")
                    break
                
                # Handle frame skipping for performance
                if frame_skip_counter < self.skip_frames:
                    frame_skip_counter += 1
                    self.skipped_count += 1
                    continue
                
                frame_skip_counter = 0
                
                # Create Frame object with metadata
                frame_obj = Frame(
                    data=frame,
                    timestamp=datetime.now(),
                    frame_id=self.frame_count,
                    source_name=self._get_source_name()
                )
                
                # Try to put frame in queue with timeout
                try:
                    if self.out_q.full():
                        # If queue is full, drop frame and log
                        self.dropped_count += 1
                        self.logger.debug(
                            f"[{self.name}] Queue full, dropped frame {self.frame_count}"
                        )
                    else:
                        self.out_q.put(frame_obj, timeout=1.0)
                        self.frame_count += 1
                except Exception as e:
                    self.logger.warning(f"[{self.name}] Failed to put frame in queue: {e}")
                    self.dropped_count += 1
                
                # Control frame rate if limit specified
                if self.fps_limit:
                    elapsed = time.time() - self.start_time
                    expected_time = self.frame_count / self.fps_limit
                    sleep_time = expected_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
        finally:
            cap.release()
            self.logger.info(f"[{self.name}] Video source released")
    
    def _open_source(self) -> Optional[cv2.VideoCapture]:
        """
        Open video source based on type.
        
        Returns:
            cv2.VideoCapture object or None if failed
        """
        try:
            if isinstance(self.source, int):
                # Camera index (0 = default camera, 1 = second camera, etc.)
                self.logger.info(f"[{self.name}] Opening camera device: {self.source}")
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    self.logger.error(f"[{self.name}] Failed to open camera {self.source}")
                    return None
                # Configure camera
                self._configure_camera(cap)
                
            elif isinstance(self.source, str):
                # File path or stream URL
                source_type = self._detect_source_type(self.source)
                self.logger.info(f"[{self.name}] Opening {source_type}: {self.source}")
                
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    self.logger.error(f"[{self.name}] Failed to open source: {self.source}")
                    return None
            else:
                self.logger.error(f"[{self.name}] Unsupported source type: {type(self.source)}")
                return None
            
            # Log video properties
            self._log_video_properties(cap)
            return cap
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error opening video source: {e}", exc_info=True)
            return None
    
    def _configure_camera(self, cap: cv2.VideoCapture):
        """Configure camera settings for optimal performance."""
        try:
            # Set FPS
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Set frame size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.logger.debug(f"[{self.name}] Camera configured")
        except Exception as e:
            self.logger.warning(f"[{self.name}] Could not configure camera: {e}")
    
    def _detect_source_type(self, source: str) -> str:
        """Detect if source is a file path or network stream."""
        if source.startswith(("http://", "https://", "rtsp://", "rtsps://")):
            return "Network Stream"
        else:
            return "Video File"
    
    def _get_source_name(self) -> str:
        """Get human-readable source name."""
        if isinstance(self.source, int):
            return f"Camera{self.source}"
        elif isinstance(self.source, str):
            if self.source.startswith(("http://", "https://", "rtsp://")):
                return "StreamURL"
            else:
                return Path(self.source).name
        return "Unknown"
    
    def _log_video_properties(self, cap: cv2.VideoCapture):
        """Log video properties for debugging."""
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(
                f"[{self.name}] Video properties - "
                f"Resolution: {width}x{height}, "
                f"FPS: {fps:.2f}, "
                f"Total frames: {int(frame_count)}"
            )
        except Exception as e:
            self.logger.debug(f"[{self.name}] Could not read video properties: {e}")


class FramePreprocessor:
    """
    Utility class for preprocessing frames before passing to detectors.
    Handles resizing, color conversion, and other preparations.
    """
    
    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 480,
        maintain_aspect: bool = True,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.maintain_aspect = maintain_aspect
        self.logger = logging.getLogger(__name__)
    
    def process(self, frame: Frame) -> Frame:
        """Apply preprocessing to a frame."""
        try:
            resized = self.resize_frame(frame.data)
            # Return a new Frame so the original is not mutated by preprocessing
            return Frame(
                data=resized,
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                source_name=frame.source_name,
            )
        except Exception as e:
            self.logger.error(f"[FramePreprocessor] Error preprocessing frame: {e}")
            return frame
    
    def resize_frame(self, image):
        """Resize frame while maintaining aspect ratio if enabled."""
        if not self.maintain_aspect:
            return cv2.resize(image, (self.target_width, self.target_height))
        
        h, w = image.shape[:2]
        aspect = w / h
        target_aspect = self.target_width / self.target_height
        
        if aspect > target_aspect:
            # Width is limiting factor
            new_w = self.target_width
            new_h = int(new_w / aspect)
        else:
            # Height is limiting factor
            new_h = self.target_height
            new_w = int(new_h * aspect)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


class VideoReadQueue:
    """
    Thread-safe wrapper for reading frames from the queue with
    advanced options like timeout and frame skipping.
    """
    
    def __init__(self, queue: Queue, timeout: float = 1.0):
        self.queue = queue
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def get_frame(self, skip_count: int = 0) -> Optional[Frame]:
        """
        Get next frame from queue, optionally skipping frames.
        
        Args:
            skip_count: Number of frames to skip
            
        Returns:
            Frame object or None if timeout
        """
        try:
            frame = self.queue.get(timeout=self.timeout)
            
            # Skip frames if requested
            for _ in range(skip_count):
                try:
                    self.queue.get(timeout=0.1)
                except:
                    break
            
            return frame
        except:
            return None
    
    def get_batch(self, batch_size: int = 5) -> list:
        """Get multiple frames from queue without blocking."""
        batch = []
        try:
            while len(batch) < batch_size:
                frame = self.queue.get(timeout=0.1)
                batch.append(frame)
        except:
            pass
        return batch


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
    )
    
    # Create test queue and event
    test_queue = Queue(maxsize=10)
    stop_event = Event()
    
    # Test with default camera (index 0)
    reader = VideoReaderProcess(
        source=0,
        out_q=test_queue,
        stop_event=stop_event,
        fps_limit=30,
        skip_frames=0,
    )
    
    reader.start()
    
    # Read and display a few frames
    frame_count = 0
    try:
        while frame_count < 100:
            try:
                frame_obj = test_queue.get(timeout=2.0)
                print(f"Received: {frame_obj}")
                frame_count += 1
            except:
                print("Queue timeout")
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        reader.join(timeout=5)
        print("Video reader process stopped")
