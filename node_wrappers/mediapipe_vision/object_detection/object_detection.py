import logging
import torch

from ....src.mediapipe_vision.object_detection.detector import ObjectDetector
from ....src.mediapipe_vision.common.base_detector_node import BaseMediaPipeDetectorNode
from ....src.mediapipe_vision.common.model_loader import MediaPipeModelLoaderBaseNode

logger = logging.getLogger(__name__)
_category = "Realtime Nodes/MediaPipe Vision/ObjectDetection"


class MediaPipeObjectDetectorModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Object Detector models."""

    TASK_TYPE = "object_detector"
    RETURN_TYPES = ("OBJECT_DETECTOR_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category


class MediaPipeObjectDetectorNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Object Detection."""

    # Define class variables required by the base class
    DETECTOR_CLASS = ObjectDetector
    MODEL_INFO_TYPE = "OBJECT_DETECTOR_MODEL_INFO"
    EXPECTED_TASK_TYPE = "object_detector"
    RETURN_TYPES = ("OBJECT_DETECTIONS",)
    RETURN_NAMES = ("object_detections",)
    FUNCTION = "detect"
    CATEGORY = _category

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()

        # Add object detection specific parameters
        inputs["required"].update(
            {
                "min_confidence": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum confidence score for detected objects",
                    },
                ),
                "max_results": (
                    "INT",
                    {"default": 5, "min": 1, "max": 50, "step": 1, "tooltip": "Maximum number of objects to detect"},
                ),
                "category_allowlist": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated list of categories to detect (e.g., 'person,cat'). Empty means all categories.",
                    },
                ),
                "category_denylist": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated list of categories to exclude. Empty means not to exclude any categories.",
                    },
                ),
            }
        )

        return inputs

    def detect(
        self,
        image: torch.Tensor,
        model_info: dict,
        min_confidence: float,
        max_results: int,
        category_allowlist: str,
        category_denylist: str,
        running_mode: str,
        delegate: str,
    ):
        """Performs object detection with the configured parameters."""

        # 1. Validate model_info and get model path
        model_path = self.validate_model_info(model_info)

        # 2. Parse string lists into lists of strings or None
        allowed_categories = [cat.strip() for cat in category_allowlist.split(',') if cat.strip()] or None
        denied_categories = [cat.strip() for cat in category_denylist.split(',') if cat.strip()] or None
        
        # 3. Collect all configuration parameters
        config = {
            "running_mode": running_mode,
            "delegate": delegate,
            "score_threshold": min_confidence,
            "max_results": max_results,
            "category_allowlist": allowed_categories,
            "category_denylist": denied_categories,
        }

        # 4. Initialize or update detector using the new base class method
        # This will create/re-create the detector only if the config has changed
        detector = self.initialize_or_update_detector(model_path, **config)

        # 5. Perform detection on the image batch
        # The detector instance is now fully configured, so we just pass the image
        batch_results = detector.detect(image)

        return (batch_results,)


# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeObjectDetectorModelLoader": MediaPipeObjectDetectorModelLoaderNode,
    "MediaPipeObjectDetector": MediaPipeObjectDetectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeObjectDetectorModelLoader": "Load Object Detector Model (MediaPipe)",
    "MediaPipeObjectDetector": "Object Detector (MediaPipe)",
}
