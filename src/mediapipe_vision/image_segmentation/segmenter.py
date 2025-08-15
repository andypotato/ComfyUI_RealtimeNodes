import logging
from typing import Any, List, Tuple

import mediapipe as mp
import numpy as np
import torch

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class ImageSegmenter(BaseDetector[Any]):
    """Performs image segmentation using MediaPipe ImageSegmenter."""

    def __init__(self, model_path: str, **kwargs):
        """Initialize the detector with the model path and configuration."""
        # Store config for use during result processing
        self._output_confidence_masks = kwargs.get('output_confidence_masks', True)
        self._output_category_mask = kwargs.get('output_category_mask', False)
        super().__init__(model_path, **kwargs)

    def _create_detector_options(self, base_options: python.BaseOptions, 
                               mode_enum: vision.RunningMode, **kwargs) -> vision.ImageSegmenterOptions:
        """Create image segmenter-specific options."""
        return vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mode_enum,
            output_confidence_masks=self._output_confidence_masks,
            output_category_mask=self._output_category_mask,
        )

    def _create_detector_instance(self, options: vision.ImageSegmenterOptions) -> vision.ImageSegmenter:
        """Create segmenter instance from options."""
        return vision.ImageSegmenter.create_from_options(options)

    def _process_detection_result(self, detection_result: Any) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process a single image's segmentation result."""
        confidence_masks = None
        category_mask = None
        
        if self._output_confidence_masks:
            if detection_result and detection_result.confidence_masks:
                confidence_masks = [torch.from_numpy(mask.numpy_view()).float() for mask in detection_result.confidence_masks]
        
        if self._output_category_mask:
            if detection_result and detection_result.category_mask:
                category_mask = torch.from_numpy(detection_result.category_mask.numpy_view()).long()
        
        return confidence_masks, category_mask

    def detect(self, image_batch: torch.Tensor) -> Tuple[List, List]:
        """Segments a batch of images."""
        batch_confidence_masks = []
        batch_category_masks = []
        
        is_video_mode = self.running_mode == "video"
        # MediaPipe uses 'segment' methods for this task
        segment_func = self._detector_instance.segment_for_video if is_video_mode else self._detector_instance.segment

        for i in range(image_batch.shape[0]):
            img_tensor = image_batch[i]
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

            try:
                if is_video_mode:
                    timestamp_ms = self._timestamp_provider.next()
                    detection_result = segment_func(mp_image, timestamp_ms)
                else:
                    detection_result = segment_func(mp_image)
                
                confidence_masks, category_mask = self._process_detection_result(detection_result)
                batch_confidence_masks.append(confidence_masks)
                batch_category_masks.append(category_mask)
            except Exception as e:
                logger.error(f"Error during segmentation for image {i}: {e}")
                batch_confidence_masks.append(None)
                batch_category_masks.append(None)

        return batch_confidence_masks, batch_category_masks
    
    def close(self):
        """Closes the segmenter instance if it exists."""
        if self._detector_instance and hasattr(self._detector_instance, "close"):
            self._detector_instance.close()
