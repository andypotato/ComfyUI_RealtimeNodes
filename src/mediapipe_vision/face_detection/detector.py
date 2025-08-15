"""Face Detection implementation using MediaPipe.

This module contains the implementation of MediaPipe Face Detection functionality.
"""

from typing import Any, List

import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common import BaseDetector
from ..types import BoundingBox, FaceDetectionResult, FaceKeypoint


class FaceDetector(BaseDetector[FaceDetectionResult]):
    """Detects faces in an image using MediaPipe FaceDetector."""

    def __init__(self, model_path: str, **kwargs):
        """Initialize the detector with the model path and configuration.

        Args:
            model_path: Path to the MediaPipe FaceDetector .task file.
            **kwargs: Configuration options for the detector.
        """
        super().__init__(model_path, **kwargs)

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.FaceDetectorOptions:
        """Create FaceDetector-specific options."""
        return vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mode_enum,
            min_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
        )

    def _create_detector_instance(self, options: vision.FaceDetectorOptions) -> vision.FaceDetector:
        """Create the FaceDetector instance from options."""
        return vision.FaceDetector.create_from_options(options)

    def _process_detection_result(self, detection_result: Any) -> List[FaceDetectionResult]:
        """Process a single image's detection result."""
        current_image_detections = []
        if detection_result and detection_result.detections:
            for detection in detection_result.detections:
                bbox_mp = detection.bounding_box
                keypoints_mp = detection.keypoints
                score = detection.categories[0].score if detection.categories else None

                bbox = BoundingBox(origin_x=bbox_mp.origin_x, origin_y=bbox_mp.origin_y, width=bbox_mp.width, height=bbox_mp.height)

                keypoints = None
                if keypoints_mp:
                    keypoints = [FaceKeypoint(label=kp.label, x=kp.x, y=kp.y) for kp in keypoints_mp]

                current_image_detections.append(FaceDetectionResult(bounding_box=bbox, keypoints=keypoints, score=score))
        return current_image_detections

    def detect(self, image_batch: torch.Tensor) -> List[List[FaceDetectionResult]]:
        """Detects faces in a batch of images."""
        batch_results = []
        for i in range(image_batch.shape[0]):
            # Use the run_detection method from the base class for each image
            result = self.run_detection(image_batch[i])
            batch_results.append(result)
        return batch_results