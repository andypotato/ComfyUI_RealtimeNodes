from typing import Any, List

import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common import BaseDetector
from ..types import GestureCategory, GestureRecognitionResult


class GestureRecognizer(BaseDetector[GestureRecognitionResult]):
    """Recognizes hand gestures using MediaPipe GestureRecognizer."""

    def __init__(self, model_path: str, **kwargs):
        """Initialize the detector with the model path and configuration."""
        super().__init__(model_path, **kwargs)

    def _create_detector_options(self, base_options: python.BaseOptions, 
                               mode_enum: vision.RunningMode, **kwargs) -> vision.GestureRecognizerOptions:
        """Create GestureRecognizer-specific options."""
        return vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            num_hands=kwargs.get('num_hands', 2),
            min_hand_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            min_hand_presence_confidence=kwargs.get('min_presence_confidence', 0.5),
        )

    def _create_detector_instance(self, options: vision.GestureRecognizerOptions) -> vision.GestureRecognizer:
        """Create GestureRecognizer instance from options."""
        return vision.GestureRecognizer.create_from_options(options)

    def _process_detection_result(self, recognition_result: Any) -> List[GestureRecognitionResult]:
        """Process a single image's detection result."""
        current_image_gestures = []
        if recognition_result and recognition_result.gestures:
            for hand_idx, hand_gestures in enumerate(recognition_result.gestures):
                gesture_categories = [
                    GestureCategory(
                        index=cat.index,
                        score=cat.score,
                        display_name=cat.display_name,
                        category_name=cat.category_name,
                    )
                    for cat in hand_gestures
                ]
                handedness = None
                if recognition_result.handedness and hand_idx < len(recognition_result.handedness):
                    handedness_cats = recognition_result.handedness[hand_idx]
                    if handedness_cats:
                        handedness = handedness_cats[0].display_name
                current_image_gestures.append(GestureRecognitionResult(gestures=gesture_categories, handedness=handedness))
        return current_image_gestures

    def detect(self, image_batch: torch.Tensor) -> List[List[GestureRecognitionResult]]:
        """Recognizes gestures in a batch of images."""
        batch_results = []
        for i in range(image_batch.shape[0]):
            # Use the run_detection method from the base class for each image
            result = self.run_detection(image_batch[i])
            batch_results.append(result)
        return batch_results