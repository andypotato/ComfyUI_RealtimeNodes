from typing import Any, List, Tuple

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common import BaseDetector
from ..types import LandmarkPoint, PoseLandmarksResult

class PoseLandmarkDetector(BaseDetector[PoseLandmarksResult]):
    """Detects pose landmarks in an image using MediaPipe PoseLandmarker."""

    def __init__(self, model_path: str, **kwargs):
        """Initialize the detector with the model path and configuration."""
        self._output_segmentation_masks = kwargs.get('output_segmentation_masks', False)
        super().__init__(model_path, **kwargs)

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.PoseLandmarkerOptions:
        """Create PoseLandmarker-specific options."""
        return vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            num_poses=kwargs.get('num_poses', 1),
            min_pose_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
            min_pose_presence_confidence=kwargs.get('min_presence_confidence', 0.5),
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            output_segmentation_masks=self._output_segmentation_masks,
        )

    def _create_detector_instance(self, options: vision.PoseLandmarkerOptions) -> vision.PoseLandmarker:
        """Create the PoseLandmarker instance from options."""
        return vision.PoseLandmarker.create_from_options(options)

    def _process_detection_result(self, detection_result: Any) -> Tuple[List[PoseLandmarksResult], List[torch.Tensor]]:
        """Process a single image's detection result, extracting landmarks and masks."""
        image_landmarks = []
        image_masks = []

        if detection_result and detection_result.pose_landmarks:
            for pose_idx, pose_landmarks_mp in enumerate(detection_result.pose_landmarks):
                landmarks = [
                    LandmarkPoint(
                        index=lm_idx, x=lm.x, y=lm.y, z=lm.z,
                        visibility=getattr(lm, "visibility", None),
                        presence=getattr(lm, "presence", None),
                    )
                    for lm_idx, lm in enumerate(pose_landmarks_mp)
                ]

                world_landmarks = None
                if detection_result.pose_world_landmarks and pose_idx < len(detection_result.pose_world_landmarks):
                    world_landmarks = [
                        LandmarkPoint(
                            index=lm_idx, x=lm.x, y=lm.y, z=lm.z,
                            visibility=getattr(lm, "visibility", None),
                            presence=getattr(lm, "presence", None),
                        )
                        for lm_idx, lm in enumerate(detection_result.pose_world_landmarks[pose_idx])
                    ]
                image_landmarks.append(PoseLandmarksResult(landmarks=landmarks, world_landmarks=world_landmarks))

        if self._output_segmentation_masks and detection_result and detection_result.segmentation_masks:
            for mask_mp in detection_result.segmentation_masks:
                mask_np = mask_mp.numpy_view()
                mask_tensor = torch.from_numpy(mask_np).float()
                image_masks.append(mask_tensor)

        return image_landmarks, image_masks

    def detect(self, image_batch: torch.Tensor) -> Tuple[List, List]:
        """Detects pose landmarks in a batch of images."""
        batch_landmarks, batch_masks = [], []
        
        is_video_mode = self.running_mode == "video"
        detect_func = self.get_detect_func(is_video_mode)

        for i in range(image_batch.shape[0]):
            img_tensor = image_batch[i]
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                timestamp_ms = self._timestamp_provider.next()
                detection_result = detect_func(mp_image, timestamp_ms)
            else:
                detection_result = detect_func(mp_image)

            landmarks, masks = self._process_detection_result(detection_result)
            batch_landmarks.append(landmarks)
            batch_masks.append(masks)
            
        return batch_landmarks, batch_masks