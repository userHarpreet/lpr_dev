"""
Unit tests for vehicle_detector module.

Covers:
- Detection dataclass properties
- VehicleFilter methods (confidence, area, position, overlapping IoU)
- DetectionVisualizer drawing
- VehicleDetectorProcess._run_inference using a mocked YOLO model

These tests avoid loading real YOLO weights by mocking the model and its outputs.
"""

import unittest
from unittest.mock import MagicMock
from datetime import datetime

import numpy as np
import cv2

from vehicle_detector import (
    Detection,
    VehicleFilter,
    DetectionVisualizer,
    VehicleDetectorProcess,
)
from video_reader import Frame


class TestDetectionDataclass(unittest.TestCase):
    def test_properties(self):
        det = Detection(class_id=2, class_name='car', confidence=0.9, x1=10, y1=20, x2=110, y2=220)
        self.assertEqual(det.width, 100)
        self.assertEqual(det.height, 200)
        self.assertEqual(det.area, 20000)
        self.assertEqual(det.center, ((10 + 110) // 2, (20 + 220) // 2))
        self.assertIn('car', repr(det))


class TestVehicleFilter(unittest.TestCase):
    def setUp(self):
        self.vf = VehicleFilter()
        self.d1 = Detection(2, 'car', 0.9, 10, 10, 110, 110)  # area 10000
        self.d2 = Detection(2, 'car', 0.4, 50, 50, 150, 150)  # lower confidence
        self.d3 = Detection(2, 'car', 0.8, 200, 200, 260, 260)  # smaller area

    def test_filter_by_confidence(self):
        filtered = self.vf.filter_by_confidence([self.d1, self.d2, self.d3], min_conf=0.5)
        self.assertIn(self.d1, filtered)
        self.assertIn(self.d3, filtered)
        self.assertNotIn(self.d2, filtered)

    def test_filter_by_area(self):
        filtered = self.vf.filter_by_area([self.d1, self.d3], min_area=5000)
        self.assertIn(self.d1, filtered)
        self.assertNotIn(self.d3, filtered)

    def test_filter_by_position(self):
        region = (0, 0, 120, 120)
        filtered = self.vf.filter_by_position([self.d1, self.d3], region=region)
        self.assertIn(self.d1, filtered)
        self.assertNotIn(self.d3, filtered)

    def test_filter_overlapping(self):
        # create overlapping detections: d_a high conf, d_b lower conf overlapping
        d_a = Detection(2, 'car', 0.95, 10, 10, 100, 100)
        d_b = Detection(2, 'car', 0.60, 20, 20, 110, 110)
        # non-overlapping
        d_c = Detection(2, 'car', 0.85, 200, 200, 260, 260)

        filtered = self.vf.filter_overlapping([d_a, d_b, d_c], iou_threshold=0.3)
        # d_a and d_c should be kept; d_b removed due to overlap with higher-conf d_a
        self.assertIn(d_a, filtered)
        self.assertIn(d_c, filtered)
        self.assertNotIn(d_b, filtered)


class TestDetectionVisualizer(unittest.TestCase):
    def test_draw_detections_changes_image(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        det = Detection(2, 'car', 0.9, 50, 50, 150, 150)
        out = DetectionVisualizer.draw_detections(img, [det])
        self.assertEqual(out.shape, img.shape)
        # Ensure at least some pixels changed (rectangle drawn)
        self.assertFalse(np.array_equal(out, img))

    def test_draw_region_of_interest(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        region = (10, 10, 100, 100)
        out = DetectionVisualizer.draw_region_of_interest(img, region)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))


class TestVehicleDetectorInferenceMock(unittest.TestCase):
    def test_run_inference_with_mocked_model(self):
        # Create a fake frame
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=test_img, timestamp=datetime.now(), frame_id=1, source_name='test')

        # Create detector instance but do not start the process
        detector = VehicleDetectorProcess(in_q=None, out_q=None, stop_event=None, vehicle_classes=[2])

        # Build a mocked model
        mock_model = MagicMock()

        # Create a fake 'box' object with needed attributes
        box = MagicMock()
        box.cls = np.array([2])
        box.conf = np.array([0.87])
        box.xyxy = np.array([[15.0, 25.0, 115.0, 125.0]])

        # The result object returned by ultralytics has .boxes
        result = MagicMock()
        result.boxes = [box]

        # Model call returns a list-like results
        mock_model.return_value = [result]
        mock_model.names = {2: 'car'}

        detector.model = mock_model

        detections = detector._run_inference(frame)

        self.assertEqual(len(detections), 1)
        d = detections[0]
        self.assertEqual(d.class_id, 2)
        self.assertEqual(d.class_name, 'car')
        self.assertAlmostEqual(d.confidence, 0.87, places=2)
        self.assertEqual(d.x1, 15)
        self.assertEqual(d.y1, 25)
        self.assertEqual(d.x2, 115)
        self.assertEqual(d.y2, 125)


if __name__ == '__main__':
    unittest.main(verbosity=2)
