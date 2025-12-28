# src/detection/detect.py

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, model_path="models/blaze_face_short_range.tflite",
                 min_confidence=0.7):
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame_rgb):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        result = self.detector.detect(mp_image)
        boxes = []

        h, w, _ = frame_rgb.shape

        if not result.detections:
            return []

        for det in result.detections:
            
            b = det.bounding_box
            x1 = int(b.origin_x)
            y1 = int(b.origin_y)
            x2 = int(b.origin_x + b.width)
            y2 = int(b.origin_y + b.height)
            boxes.append((x1, y1, x2, y2))

        return boxes
