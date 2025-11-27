"""
Unit Tests for VideoReader Module

Tests cover:
- Frame dataclass functionality
- VideoReaderProcess initialization
- FramePreprocessor image transformations
- VideoReadQueue operations
- Helper methods (source detection, source naming, property logging)
"""

import unittest
import logging
import time
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from multiprocessing import Queue, Event

import cv2
import numpy as np

from video_reader import (
    Frame,
    VideoReaderProcess,
    FramePreprocessor,
    VideoReadQueue,
)


class TestFrame(unittest.TestCase):
    """Test Frame dataclass"""
    
    def setUp(self):
        """Create test data"""
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.timestamp = datetime.now()
        
    def test_frame_creation(self):
        """Test Frame object creation"""
        frame = Frame(
            data=self.test_image,
            timestamp=self.timestamp,
            frame_id=0,
            source_name="test_source"
        )
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_id, 0)
        self.assertEqual(frame.source_name, "test_source")
        self.assertTrue(np.array_equal(frame.data, self.test_image))
    
    def test_frame_repr(self):
        """Test Frame string representation"""
        frame = Frame(
            data=self.test_image,
            timestamp=self.timestamp,
            frame_id=5,
            source_name="camera0"
        )
        
        repr_str = repr(frame)
        self.assertIn("id=5", repr_str)
        self.assertIn("camera0", repr_str)
        self.assertIn("(480, 640, 3)", repr_str)
    
    def test_frame_image_shape(self):
        """Test that Frame stores image with correct shape"""
        test_shapes = [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]
        
        for shape in test_shapes:
            test_img = np.zeros(shape, dtype=np.uint8)
            frame = Frame(
                data=test_img,
                timestamp=self.timestamp,
                frame_id=0,
                source_name="test"
            )
            self.assertEqual(frame.data.shape, shape)


class TestFramePreprocessor(unittest.TestCase):
    """Test FramePreprocessor utility class"""
    
    def setUp(self):
        """Create test images"""
        self.preprocessor = FramePreprocessor(
            target_width=640,
            target_height=480,
            maintain_aspect=True
        )
        self.test_frame = Frame(
            data=np.zeros((1080, 1920, 3), dtype=np.uint8),
            timestamp=datetime.now(),
            frame_id=0,
            source_name="test"
        )
    
    def test_preprocessor_initialization(self):
        """Test FramePreprocessor initialization"""
        self.assertEqual(self.preprocessor.target_width, 640)
        self.assertEqual(self.preprocessor.target_height, 480)
        self.assertTrue(self.preprocessor.maintain_aspect)
    
    def test_resize_frame_without_aspect_ratio(self):
        """Test resize without maintaining aspect ratio"""
        preprocessor = FramePreprocessor(
            target_width=640,
            target_height=480,
            maintain_aspect=False
        )
        
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = preprocessor.resize_frame(test_image)
        
        self.assertEqual(resized.shape[:2], (480, 640))
    
    def test_resize_frame_with_aspect_ratio(self):
        """Test resize while maintaining aspect ratio"""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3 aspect
        resized = self.preprocessor.resize_frame(test_image)
        
        # Should fit within target bounds while maintaining aspect
        self.assertLessEqual(resized.shape[1], self.preprocessor.target_width)
        self.assertLessEqual(resized.shape[0], self.preprocessor.target_height)
        
        # Check aspect ratio is preserved
        original_aspect = 640 / 480
        resized_aspect = resized.shape[1] / resized.shape[0]
        self.assertAlmostEqual(original_aspect, resized_aspect, places=1)
    
    def test_process_frame(self):
        """Test processing a Frame object"""
        processed = self.preprocessor.process(self.test_frame)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed.frame_id, self.test_frame.frame_id)
        # Image should be resized
        self.assertNotEqual(processed.data.shape, self.test_frame.data.shape)


class TestVideoReadQueue(unittest.TestCase):
    """Test VideoReadQueue wrapper class"""
    
    def setUp(self):
        """Create test queue"""
        self.queue = Queue(maxsize=10)
        self.read_queue = VideoReadQueue(self.queue, timeout=1.0)
        self.timestamp = datetime.now()
    
    def test_queue_initialization(self):
        """Test VideoReadQueue initialization"""
        self.assertEqual(self.read_queue.timeout, 1.0)
        self.assertIsNotNone(self.read_queue.queue)
    
    def test_get_frame_success(self):
        """Test getting a frame from queue"""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            data=test_image,
            timestamp=self.timestamp,
            frame_id=0,
            source_name="test"
        )
        
        self.queue.put(frame)
        retrieved = self.read_queue.get_frame()
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.frame_id, 0)
    
    def test_get_frame_timeout(self):
        """Test get_frame returns None on timeout"""
        read_queue = VideoReadQueue(self.queue, timeout=0.1)
        result = read_queue.get_frame()
        
        self.assertIsNone(result)
    
    def test_get_frame_with_skip(self):
        """Test skipping frames"""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Put 5 frames in queue
        for i in range(5):
            frame = Frame(
                data=test_image,
                timestamp=self.timestamp,
                frame_id=i,
                source_name="test"
            )
            self.queue.put(frame)
        
        # Get first frame and skip 2
        retrieved = self.read_queue.get_frame(skip_count=2)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.frame_id, 0)
        # Queue should have fewer items after skipping
        self.assertLess(self.queue.qsize(), 5)
    
    def test_get_batch(self):
        """Test getting batch of frames"""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Put 5 frames in queue
        for i in range(5):
            frame = Frame(
                data=test_image,
                timestamp=self.timestamp,
                frame_id=i,
                source_name="test"
            )
            self.queue.put(frame)
        
        batch = self.read_queue.get_batch(batch_size=3)
        
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch[0].frame_id, 0)
        self.assertEqual(batch[1].frame_id, 1)
        self.assertEqual(batch[2].frame_id, 2)
    
    def test_get_batch_partial(self):
        """Test getting batch when fewer frames available"""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Put 2 frames
        for i in range(2):
            frame = Frame(
                data=test_image,
                timestamp=self.timestamp,
                frame_id=i,
                source_name="test"
            )
            self.queue.put(frame)
        
        # Request 5 but only 2 available
        batch = self.read_queue.get_batch(batch_size=5)
        
        self.assertEqual(len(batch), 2)


class TestVideoReaderProcessInit(unittest.TestCase):
    """Test VideoReaderProcess initialization"""
    
    def test_process_initialization_with_camera(self):
        """Test initializing VideoReaderProcess with camera index"""
        queue = Queue()
        stop_event = Event()
        
        process = VideoReaderProcess(
            source=0,
            out_q=queue,
            stop_event=stop_event,
            fps_limit=30,
            skip_frames=0
        )
        
        self.assertEqual(process.source, 0)
        self.assertEqual(process.fps_limit, 30)
        self.assertEqual(process.skip_frames, 0)
        self.assertFalse(process.daemon)
    
    def test_process_initialization_with_file(self):
        """Test initializing VideoReaderProcess with file path"""
        queue = Queue()
        stop_event = Event()
        
        process = VideoReaderProcess(
            source="test_video.mp4",
            out_q=queue,
            stop_event=stop_event
        )
        
        self.assertEqual(process.source, "test_video.mp4")
        self.assertIsNone(process.fps_limit)
    
    def test_process_initialization_with_stream(self):
        """Test initializing VideoReaderProcess with stream URL"""
        queue = Queue()
        stop_event = Event()
        
        process = VideoReaderProcess(
            source="rtsp://example.com/stream",
            out_q=queue,
            stop_event=stop_event
        )
        
        self.assertEqual(process.source, "rtsp://example.com/stream")
    
    def test_process_frame_tracking_initialization(self):
        """Test that frame tracking counters are initialized"""
        process = VideoReaderProcess(source=0)
        
        self.assertEqual(process.frame_count, 0)
        self.assertEqual(process.skipped_count, 0)
        self.assertEqual(process.dropped_count, 0)
        self.assertIsNone(process.start_time)


class TestVideoReaderHelperMethods(unittest.TestCase):
    """Test VideoReaderProcess helper methods"""
    
    def setUp(self):
        """Create VideoReaderProcess instance"""
        self.process = VideoReaderProcess(source=0)
    
    def test_detect_source_type_http_stream(self):
        """Test detecting HTTP stream"""
        result = self.process._detect_source_type("http://example.com/stream.m3u8")
        self.assertEqual(result, "Network Stream")
    
    def test_detect_source_type_https_stream(self):
        """Test detecting HTTPS stream"""
        result = self.process._detect_source_type("https://example.com/stream.m3u8")
        self.assertEqual(result, "Network Stream")
    
    def test_detect_source_type_rtsp_stream(self):
        """Test detecting RTSP stream"""
        result = self.process._detect_source_type("rtsp://example.com/stream")
        self.assertEqual(result, "Network Stream")
    
    def test_detect_source_type_file(self):
        """Test detecting video file"""
        result = self.process._detect_source_type("video.mp4")
        self.assertEqual(result, "Video File")
    
    def test_get_source_name_camera(self):
        """Test getting source name for camera"""
        process = VideoReaderProcess(source=0)
        name = process._get_source_name()
        self.assertEqual(name, "Camera0")
    
    def test_get_source_name_file(self):
        """Test getting source name for file"""
        process = VideoReaderProcess(source="/path/to/video.mp4")
        name = process._get_source_name()
        self.assertEqual(name, "video.mp4")
    
    def test_get_source_name_stream_url(self):
        """Test getting source name for stream URL"""
        process = VideoReaderProcess(source="rtsp://example.com/stream")
        name = process._get_source_name()
        self.assertEqual(name, "StreamURL")
    
    def test_get_source_name_http_stream(self):
        """Test getting source name for HTTP stream"""
        process = VideoReaderProcess(source="http://example.com/stream.m3u8")
        name = process._get_source_name()
        self.assertEqual(name, "StreamURL")


class TestVideoReaderWithMockedCapture(unittest.TestCase):
    """Test VideoReaderProcess with mocked cv2.VideoCapture"""
    
    @patch('video_reader.cv2.VideoCapture')
    def test_open_source_camera_success(self, mock_capture):
        """Test opening camera source successfully"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        process = VideoReaderProcess(source=0)
        result = process._open_source()
        
        self.assertIsNotNone(result)
        mock_capture.assert_called_once_with(0)
        mock_cap.isOpened.assert_called()
    
    @patch('video_reader.cv2.VideoCapture')
    def test_open_source_camera_failure(self, mock_capture):
        """Test opening camera source failure"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        process = VideoReaderProcess(source=0)
        result = process._open_source()
        
        self.assertIsNone(result)
    
    @patch('video_reader.cv2.VideoCapture')
    def test_open_source_file(self, mock_capture):
        """Test opening file source"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        process = VideoReaderProcess(source="test.mp4")
        result = process._open_source()
        
        self.assertIsNotNone(result)
        mock_capture.assert_called_once_with("test.mp4")
    
    @patch('video_reader.cv2.VideoCapture')
    def test_open_source_invalid_type(self, mock_capture):
        """Test opening invalid source type"""
        process = VideoReaderProcess(source=["invalid", "list"])
        result = process._open_source()
        
        self.assertIsNone(result)
        mock_capture.assert_not_called()
    
    @patch('video_reader.cv2.VideoCapture')
    def test_configure_camera(self, mock_capture):
        """Test camera configuration"""
        mock_cap = MagicMock()
        
        process = VideoReaderProcess(source=0)
        process._configure_camera(mock_cap)
        
        # Verify camera settings were applied
        expected_calls = [
            ((cv2.CAP_PROP_FPS, 30),),
            ((cv2.CAP_PROP_FRAME_WIDTH, 1280),),
            ((cv2.CAP_PROP_FRAME_HEIGHT, 720),),
            ((cv2.CAP_PROP_BUFFERSIZE, 1),),
        ]
        
        for call_args in expected_calls:
            mock_cap.set.assert_any_call(*call_args[0])
    
    @patch('video_reader.cv2.VideoCapture')
    def test_log_video_properties(self, mock_capture):
        """Test logging video properties"""
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 1000,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
        }.get(prop, 0)
        
        process = VideoReaderProcess(source="test.mp4")
        
        # Should not raise exception
        with patch.object(process.logger, 'info') as mock_log:
            process._log_video_properties(mock_cap)
            mock_log.assert_called()


class TestVideoReaderFrameTracking(unittest.TestCase):
    """Test frame counting and tracking in VideoReaderProcess"""
    
    def test_frame_counter_initialization(self):
        """Test that frame counters start at 0"""
        process = VideoReaderProcess(source=0)
        
        self.assertEqual(process.frame_count, 0)
        self.assertEqual(process.skipped_count, 0)
        self.assertEqual(process.dropped_count, 0)
    
    def test_frame_counter_attributes(self):
        """Test frame counter attributes exist"""
        process = VideoReaderProcess(source=0)
        
        self.assertTrue(hasattr(process, 'frame_count'))
        self.assertTrue(hasattr(process, 'skipped_count'))
        self.assertTrue(hasattr(process, 'dropped_count'))
        self.assertTrue(hasattr(process, 'start_time'))


class TestVideoReaderWithRealVideo(unittest.TestCase):
    """Test VideoReaderProcess with actual video file"""
    
    @classmethod
    def setUpClass(cls):
        """Create a test video file"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_video_path = os.path.join(cls.temp_dir, "test_video.mp4")
        
        # Create a simple test video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.test_video_path, fourcc, 30.0, (640, 480))
        
        # Write 10 frames
        for i in range(10):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 25)
            out.write(frame)
        
        out.release()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test video file"""
        if os.path.exists(cls.test_video_path):
            os.remove(cls.test_video_path)
        os.rmdir(cls.temp_dir)
    
    def test_read_video_file(self):
        """Test reading from actual video file"""
        queue = Queue(maxsize=20)
        stop_event = Event()
        
        process = VideoReaderProcess(
            source=self.test_video_path,
            out_q=queue,
            stop_event=stop_event,
            fps_limit=30
        )
        
        process.start()
        
        # Give process time to start
        time.sleep(1)
        
        # Check if frames are being read
        frames_received = 0
        try:
            while frames_received < 5:
                frame = queue.get(timeout=2.0)
                frames_received += 1
                self.assertIsNotNone(frame)
                self.assertIsInstance(frame, Frame)
                self.assertEqual(frame.data.shape, (480, 640, 3))
        except:
            pass
        
        stop_event.set()
        process.join(timeout=5)
        
        self.assertGreater(frames_received, 0, "Should have received at least 1 frame")


class TestVideoReaderEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_skip_frames_boundary(self):
        """Test skip_frames parameter with boundary values"""
        queue = Queue()
        
        for skip_count in [0, 1, 5, 100]:
            process = VideoReaderProcess(
                source=0,
                out_q=queue,
                skip_frames=skip_count
            )
            self.assertEqual(process.skip_frames, skip_count)
    
    def test_fps_limit_boundary(self):
        """Test fps_limit parameter with boundary values"""
        queue = Queue()
        
        for fps in [None, 15, 30, 60, 120]:
            process = VideoReaderProcess(
                source=0,
                out_q=queue,
                fps_limit=fps
            )
            self.assertEqual(process.fps_limit, fps)
    
    def test_frame_with_different_data_types(self):
        """Test Frame with different numpy data types"""
        timestamp = datetime.now()
        
        data_types = [np.uint8, np.uint16, np.float32]
        
        for dtype in data_types:
            test_image = np.zeros((480, 640, 3), dtype=dtype)
            frame = Frame(
                data=test_image,
                timestamp=timestamp,
                frame_id=0,
                source_name="test"
            )
            
            self.assertEqual(frame.data.dtype, dtype)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
    )
    
    unittest.main(verbosity=2)
