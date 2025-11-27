"""
Unit tests for plate_detector module.

Covers:
- PlateDetection dataclass properties
- PlateFilter utilities (aspect ratio, confidence, overlapping)
- PlateVisualizer drawing and cropping
- PlateDetectorProcess._run_inference using mocked model
"""

import unittest
from unittest.mock import MagicMock
from datetime import datetime

import numpy as np
import cv2

from plate_detector import (
    PlateDetection,
    PlateFilter,
    PlateVisualizer,
    PlateDetectorProcess,
)
from video_reader import Frame


class TestPlateDetectionDataclass(unittest.TestCase):
    def test_properties_and_repr(self):
        pd = PlateDetection(class_id=0, class_name='plate', confidence=0.95, x1=10, y1=20, x2=210, y2=80)
        self.assertEqual(pd.width, 200)
        self.assertEqual(pd.height, 60)
        self.assertEqual(pd.area, 12000)
        self.assertIn('plate', repr(pd))


class TestPlateFilter(unittest.TestCase):
    def setUp(self):
        self.filter = PlateFilter()
        self.p1 = PlateDetection(0, 'plate', 0.9, 10, 10, 210, 70)  # wide plate
        self.p2 = PlateDetection(0, 'plate', 0.4, 50, 20, 150, 80)  # low conf
        self.p3 = PlateDetection(0, 'plate', 0.85, 300, 300, 420, 360)  # separate

    def test_filter_by_confidence(self):
        out = self.filter.filter_by_confidence([self.p1, self.p2, self.p3], min_conf=0.5)
        self.assertIn(self.p1, out)
        self.assertNotIn(self.p2, out)

    def test_filter_by_aspect_ratio(self):
        # p1 is wide (should pass), p3 might be near 2:1 depending on coords
        out = self.filter.filter_by_aspect_ratio([self.p1, self.p3], min_ratio=2.0, max_ratio=6.0)
        self.assertIn(self.p1, out)

    def test_filter_overlapping(self):
        # overlapping plates: higher-conf kept
        a = PlateDetection(0, 'plate', 0.95, 10, 10, 110, 40)
        b = PlateDetection(0, 'plate', 0.6, 20, 15, 115, 45)
        out = self.filter.filter_overlapping([a, b], iou_threshold=0.2)
        self.assertIn(a, out)
        self.assertNotIn(b, out)


class TestPlateVisualizer(unittest.TestCase):
    def test_draw_plate_boxes(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        p = PlateDetection(0, 'plate', 0.9, 50, 50, 200, 90)
        out = PlateVisualizer.draw_plates(img, [p])
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))

    def test_crop_plate_from_image(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        # draw a white rectangle where plate will be
        cv2.rectangle(img, (50, 50), (200, 90), (255, 255, 255), -1)
        p = PlateDetection(0, 'plate', 0.9, 50, 50, 200, 90)
        cropped = p.crop_from_image(img)
        self.assertEqual(cropped.shape[1], p.width)
        self.assertEqual(cropped.shape[0], p.height)


class TestPlateDetectorInferenceMock(unittest.TestCase):
    def test_run_inference_with_mocked_model(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=img, timestamp=datetime.now(), frame_id=1, source_name='test')

        detector = PlateDetectorProcess(in_q=None, out_q=None, stop_event=None, plate_class_id=0)

        # Mock model and result structure similar to ultralytics
        mock_model = MagicMock()
        box = MagicMock()
        box.cls = np.array([0])
        box.conf = np.array([0.92])
        box.xyxy = np.array([[30.0, 40.0, 180.0, 80.0]])

        result = MagicMock()
        result.boxes = [box]

        mock_model.return_value = [result]
        mock_model.names = {0: 'plate'}

        detector.model = mock_model

        detections = detector._run_inference(frame)

        self.assertEqual(len(detections), 1)
        d = detections[0]
        self.assertEqual(d.class_name, 'plate')
        self.assertAlmostEqual(d.confidence, 0.92, places=2)
        self.assertEqual(d.x1, 30)
        self.assertEqual(d.y1, 40)
        self.assertEqual(d.x2, 180)
        self.assertEqual(d.y2, 80)


if __name__ == '__main__':
    unittest.main(verbosity=2)
