from typing import Any, List, Optional

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider
from ..common import BaseDetector
from ..types import BoundingBox, ObjectDetectionCategory, ObjectDetectionResult


class ObjectDetector(BaseDetector[ObjectDetectionResult]):
    """Detects objects using MediaPipe ObjectDetector."""

    def __init__(self, model_path: str, **kwargs):
        """Initialize the detector with the model path.

        Args:
            model_path: Path to the MediaPipe ObjectDetector .tflite file.
        """
        if not model_path:
            raise ValueError("A valid model_path must be provided.")

        # Pass all config options to the base class to handle instance creation
        super().__init__(model_path, **kwargs)

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.ObjectDetectorOptions:
        """Create ObjectDetector-specific options with parameters:
            - score_threshold: Minimum score for object detection
            - max_results: Maximum number of detection results
        """
        return vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=mode_enum,
            score_threshold=kwargs.get('score_threshold', 0.5),
            max_results=kwargs.get('max_results', 5),
            category_allowlist=kwargs.get('category_allowlist', None),
            category_denylist=kwargs.get('category_denylist', None),
        )

    def _create_detector_instance(self, options: vision.ObjectDetectorOptions) -> vision.ObjectDetector:
        """Create ObjectDetector-specific instance."""
        return vision.ObjectDetector.create_from_options(options)

    def _process_detection_result(self, detection_result: Any) -> List[ObjectDetectionResult]:
        """Process a single image's detection result."""
        current_image_detections = []
        if detection_result and detection_result.detections:
            for detection in detection_result.detections:
                bbox_mp = detection.bounding_box
                categories_mp = detection.categories

                bbox = BoundingBox(origin_x=bbox_mp.origin_x, origin_y=bbox_mp.origin_y, width=bbox_mp.width, height=bbox_mp.height)

                categories = [
                    ObjectDetectionCategory(
                        index=cat.index,
                        score=cat.score,
                        display_name=cat.display_name,
                        category_name=cat.category_name,
                    )
                    for cat in categories_mp
                ]

                current_image_detections.append(ObjectDetectionResult(bounding_box=bbox, categories=categories))
        return current_image_detections

    def detect(self, image_batch: torch.Tensor) -> List[List[ObjectDetectionResult]]:
        """Detects objects in a batch of images."""
        batch_results = []
        for i in range(image_batch.shape[0]):
            # Use the run_detection method from the base class for each image
            result = self.run_detection(image_batch[i])
            batch_results.append(result)
        return batch_results
