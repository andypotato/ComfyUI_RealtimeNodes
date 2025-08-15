from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ...utils.timing import TimestampProvider

T = TypeVar('T')  # Generic type for detector results

class BaseDetector(Generic[T], ABC):
    """Base class for MediaPipe vision detectors.
    
    This class provides common functionality for all MediaPipe vision detectors,
    reducing code duplication and providing a consistent interface.
    
    Generic type T represents the detector-specific result type.
    """

    def __init__(self, model_path: str, running_mode: str = "video", delegate: str = "cpu", **kwargs):
        if not model_path:
            raise ValueError("A valid model_path must be provided.")

        self.model_path = model_path
        self.running_mode = running_mode
        self.delegate = delegate
        
        mode_enum = vision.RunningMode.IMAGE if running_mode == "image" else vision.RunningMode.VIDEO
        delegate_enum = BaseOptions.Delegate.CPU if delegate.lower() == "cpu" else BaseOptions.Delegate.GPU
        
        base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate_enum)
        options = self._create_detector_options(base_options, mode_enum, **kwargs)
        
        self._detector_instance = self._create_detector_instance(options)
        self._timestamp_provider = TimestampProvider() if mode_enum == vision.RunningMode.VIDEO else None

    @abstractmethod
    def _create_detector_options(self, base_options: python.BaseOptions, 
                                mode_enum: vision.RunningMode, **kwargs) -> Any:
        """Create detector-specific options.
        
        Args:
            base_options: MediaPipe base options with model path and delegate.
            mode_enum: RunningMode enum (IMAGE or VIDEO).
            **kwargs: Detector-specific configuration parameters.
            
        Returns:
            Detector-specific options object.
        """
        pass

    @abstractmethod
    def _process_detection_result(self, detection_result: Any) -> List[T]:
        """Process detector-specific results.
        
        Args:
            detection_result: Raw detection result from MediaPipe.
            
        Returns:
            List of processed detection results of type T.
        """
        pass

    @abstractmethod
    def _create_detector_instance(self, options: Any) -> Any:
        """Create detector-specific instance.
        
        Args:
            options: Configured detector options.
            
        Returns:
            MediaPipe detector instance.
        """
        pass

    def run_detection(self, image: torch.Tensor) -> List[T]:
        """Runs detection on a single image tensor (HWC)."""
        if image.dim() != 3:
            raise ValueError("Input tensor for run_detection must be in HWC format.")

        np_image = (image.cpu().numpy() * 255).astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

        is_video_mode = self.running_mode == "video"
        detect_func = self.get_detect_func(is_video_mode)

        if is_video_mode:
            timestamp_ms = self._timestamp_provider.next()
            detection_result = detect_func(mp_image, timestamp_ms)
        else:
            detection_result = detect_func(mp_image)
            
        return self._process_detection_result(detection_result)

    def get_detect_func(self, is_video_mode: bool):
        """Gets the appropriate detection function from the MediaPipe instance."""
        if hasattr(self._detector_instance, "detect_for_video") and is_video_mode:
            return self._detector_instance.detect_for_video
        if hasattr(self._detector_instance, "detect"):
            return self._detector_instance.detect
        
        # Fallback for other method names like 'recognize'
        func_names = [func for func in dir(self._detector_instance) if func.startswith(("detect", "recognize")) and callable(getattr(self._detector_instance, func))]
        if not func_names:
            raise AttributeError(f"No detection method found on {self._detector_instance.__class__.__name__}")
        
        video_funcs = [func for func in func_names if "video" in func.lower()]
        if is_video_mode and video_funcs:
            return getattr(self._detector_instance, video_funcs[0])
        return getattr(self._detector_instance, func_names[0])

    def close(self):
        """Closes the detector instance."""
        if self._detector_instance and hasattr(self._detector_instance, "close"):
            self._detector_instance.close()