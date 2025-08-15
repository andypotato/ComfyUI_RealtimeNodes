from typing import Any, Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import torch

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common import BaseDetector
from ..types import LandmarkPoint


class FaceLandmarkDetector(BaseDetector[Any]):
    """Detects face landmarks and blendshapes using MediaPipe FaceLandmarker."""

    def __init__(self, model_path: str, **kwargs):
        """Initialize the detector with the model path and configuration."""
        # Store these for use during result processing
        self._output_blendshapes = kwargs.get('output_blendshapes', False)
        self._output_transform_matrix = kwargs.get('output_transform_matrix', False)
        super().__init__(model_path, **kwargs)

    def _create_detector_options(self, base_options: python.BaseOptions,
                               mode_enum: vision.RunningMode, **kwargs) -> vision.FaceLandmarkerOptions:
        """Create FaceLandmarker-specific options."""
        return vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            num_faces=kwargs.get('num_faces', 1),
            min_face_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
            min_face_presence_confidence=kwargs.get('min_presence_confidence', 0.5),
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            output_face_blendshapes=kwargs.get('output_blendshapes', False),
            output_facial_transformation_matrixes=kwargs.get('output_transform_matrix', False),
        )

    def _create_detector_instance(self, options: vision.FaceLandmarkerOptions) -> vision.FaceLandmarker:
        """Create the FaceLandmarker instance from options."""
        return vision.FaceLandmarker.create_from_options(options)

    def _process_detection_result(self, detection_result: Any) -> Any:
        # This is not used because we process the multi-part result directly in detect().
        # However, the abstract method must be implemented.
        return detection_result

    def detect(
        self,
        image_batch: torch.Tensor,
    ) -> Tuple[List, List, List]:
        """Detects face landmarks in a batch of images and returns all data types."""
        batch_landmarks, batch_blendshapes, batch_matrices = [], [], []
        
        is_video_mode = self.running_mode == "video"
        detect_func = self.get_detect_func(is_video_mode)

        for i in range(image_batch.shape[0]):
            img_tensor = image_batch[i]
            # The base class's run_detection handles the image conversion and detection call
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            if is_video_mode:
                timestamp_ms = self._timestamp_provider.next()
                detection_result = detect_func(mp_image, timestamp_ms)
            else:
                detection_result = detect_func(mp_image)

            # Process the raw result for this image
            image_landmarks, image_blendshapes, image_matrices = [], [], []
            if detection_result and detection_result.face_landmarks:
                for face_idx, face_landmarks_mp in enumerate(detection_result.face_landmarks):
                    landmarks = [LandmarkPoint(index=lm_idx, x=lm.x, y=lm.y, z=lm.z) for lm_idx, lm in enumerate(face_landmarks_mp)]
                    image_landmarks.append(landmarks)

                    if self._output_blendshapes and hasattr(detection_result, 'face_blendshapes') and detection_result.face_blendshapes and face_idx < len(detection_result.face_blendshapes):
                        blendshapes = {bs.category_name: bs.score for bs in detection_result.face_blendshapes[face_idx]}
                        image_blendshapes.append(blendshapes)

                    if self._output_transform_matrix and hasattr(detection_result, 'facial_transformation_matrixes') and detection_result.facial_transformation_matrixes and face_idx < len(detection_result.facial_transformation_matrixes):
                        image_matrices.append(detection_result.facial_transformation_matrixes[face_idx])
            
            batch_landmarks.append(image_landmarks)
            batch_blendshapes.append(image_blendshapes)
            batch_matrices.append(image_matrices)

        return batch_landmarks, batch_blendshapes, batch_matrices