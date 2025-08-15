from typing import Any, List

import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common import BaseDetector
from ..types import HandLandmarksResult, LandmarkPoint


class HandLandmarkDetector(BaseDetector[HandLandmarksResult]):
    """Detects hand landmarks using MediaPipe HandLandmarker."""

    def __init__(self, model_path: str, **kwargs):
        """Initialize the detector with the model path and configuration."""
        super().__init__(model_path, **kwargs)

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.HandLandmarkerOptions:
        """Create HandLandmarker-specific options."""
        return vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            num_hands=kwargs.get('num_hands', 2),
            min_hand_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
            min_hand_presence_confidence=kwargs.get('min_presence_confidence', 0.5),
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
        )

    def _create_detector_instance(self, options: vision.HandLandmarkerOptions) -> vision.HandLandmarker:
        """Create the HandLandmarker instance from options."""
        return vision.HandLandmarker.create_from_options(options)

    def _process_detection_result(self, detection_result: Any) -> List[HandLandmarksResult]:
        """Process a single image's detection result."""
        current_image_results = []
        if detection_result and detection_result.hand_landmarks:
            for hand_idx, hand_landmarks_mp in enumerate(detection_result.hand_landmarks):
                landmarks = [LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z) for lm_idx, lm in enumerate(hand_landmarks_mp)]

                world_landmarks = None
                if detection_result.hand_world_landmarks and hand_idx < len(detection_result.hand_world_landmarks):
                    world_landmarks = [
                        LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z)
                        for lm_idx, lm in enumerate(detection_result.hand_world_landmarks[hand_idx])
                    ]

                handedness = None
                if detection_result.handedness and hand_idx < len(detection_result.handedness):
                    handedness_cats = detection_result.handedness[hand_idx]
                    if handedness_cats:
                        handedness = handedness_cats[0].display_name

                current_image_results.append(
                    HandLandmarksResult(landmarks=landmarks, world_landmarks=world_landmarks, handedness=handedness)
                )
        return current_image_results

    def detect(self, image_batch: torch.Tensor) -> List[List[HandLandmarksResult]]:
        """Detects hand landmarks in a batch of images."""
        batch_results = []
        for i in range(image_batch.shape[0]):
            # Use the run_detection method from the base class for each image
            result = self.run_detection(image_batch[i])
            batch_results.append(result)
        return batch_results