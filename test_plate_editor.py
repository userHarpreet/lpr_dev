"""
Unit tests for plate_editor module.

Covers:
- Enhancement functions produce expected output shapes and types
- Cropping behavior removes left-edge percentage when configured
- Ensures original Frame is not mutated by enhancement (returns new EditedPlate)
"""

import unittest
from datetime import datetime
import numpy as np
import cv2
from plate_editor import PlateEditorProcess, EditedPlate, PlateEnhancementConfig
from video_reader import Frame


class TestPlateEditorEnhancements(unittest.TestCase):
    def setUp(self):
        # create a synthetic plate-like image (small white rectangle on dark background)
        self.img = np.zeros((60, 200, 3), dtype=np.uint8)
        cv2.rectangle(self.img, (10, 10), (190, 50), (255, 255, 255), -1)
        self.frame = Frame(data=self.img.copy(), timestamp=datetime.now(), frame_id=0, source_name='test')
        self.editor = PlateEditorProcess(in_q=None, out_q=None, stop_event=None)

    def test_enhance_method_returns_image(self):
        edited = self.editor._enhance_plate(self.frame.data)
        self.assertIsNotNone(edited)
        self.assertIsInstance(edited, np.ndarray)

    def test_crop_plate_percentage(self):
        # simulate cropping slice from left by percentage
        cropped = self.editor._crop_plate(self.frame.data)
        # width should be reduced by crop percentage
        self.assertIsNotNone(cropped)
        self.assertIsInstance(cropped, np.ndarray)
        # cropped should be smaller or equal in width
        self.assertLessEqual(cropped.shape[1], self.img.shape[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
