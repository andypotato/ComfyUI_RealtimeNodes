import logging
import cv2
import numpy as np
import torch

from ....src.mediapipe_vision.image_segmentation.segmenter import ImageSegmenter
from ....src.mediapipe_vision.common.base_detector_node import BaseMediaPipeDetectorNode
from ....src.mediapipe_vision.common.model_loader import MediaPipeModelLoaderBaseNode

logger = logging.getLogger(__name__)

_category = "Realtime Nodes/MediaPipe Vision/ImageSegmentation"

# Define class names and their corresponding indices for multiclass models
MULTICLASS_NAMES = {
    "Background": 0,
    "Hair": 1,
    "Body-skin": 2,
    "Face-skin": 3,
    "Clothes": 4,
    "Accessories/Other": 5,
}

# Uses MULTICLASS_NAMES indices
CLASS_COLORS_BGR = {
    MULTICLASS_NAMES["Background"]: (0, 0, 0),  # Black
    MULTICLASS_NAMES["Hair"]: (255, 0, 0),  # Blue
    MULTICLASS_NAMES["Body-skin"]: (0, 255, 255),  # Yellow
    MULTICLASS_NAMES["Face-skin"]: (255, 200, 100),  # Light Blue
    MULTICLASS_NAMES["Clothes"]: (0, 255, 0),  # Green
    MULTICLASS_NAMES["Accessories/Other"]: (255, 0, 255),  # Magenta
}
# --- End Constants ---


# Inherit from the base loader
class MediaPipeImageSegmenterModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Image Segmenter models."""

    TASK_TYPE = "image_segmenter"
    RETURN_TYPES = ("IMAGE_SEGMENTER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    # INPUT_TYPES and FUNCTION inherited
    CATEGORY = _category


class MediaPipeImageSegmenterNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Image Segmentation, adapted from Stream-Pack.
    Manages the ImageSegmenter instance for potential reuse based on config.
    """

    # Define class variables required by the base class
    DETECTOR_CLASS = ImageSegmenter
    MODEL_INFO_TYPE = "IMAGE_SEGMENTER_MODEL_INFO"
    EXPECTED_TASK_TYPE = "image_segmenter"
    RETURN_TYPES = (
        "MASK",
        "IMAGE",
        "MP_INT_MASK",
    )
    RETURN_NAMES = ("mask", "visualization", "multiclass_segments")
    FUNCTION = "detect"
    CATEGORY = _category

    @classmethod
    def INPUT_TYPES(cls):
        # This method is correct as-is, no changes needed.
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "output_confidence_masks": ("BOOLEAN", {"default": False, "tooltip": "Output confidence mask (0-1) instead of category mask. Disables multiclass/visualization."}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold for confidence mask output mode."}),
            "generate_visualization": ("BOOLEAN", {"default": False, "tooltip": "Generate a colored visualization image (only works in category mask mode)."}),
        })
        inputs["required"]["delegate_mode"] = inputs["required"].pop("delegate")
        return inputs

    # --- Helper methods from Stream-Pack ---
    def is_multiclass_model(self, model_info: dict):
        # Improved check based on variant or path
        model_variant = model_info.get("model_variant", "").lower()
        model_path = model_info.get("model_path", "").lower()
        # Add known multiclass identifiers
        return "multiclass" in model_variant or "multiclass" in model_path

    def create_visualization(self, category_mask_np, is_multiclass=False):
        """Generates a colored visualization from a category mask."""
        if category_mask_np is None:
            return None
        if not is_multiclass:
            # Simple binary visualization (white on black)
            vis_image = np.where(category_mask_np > 0, 255, 0).astype(np.uint8)
            return cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        else:
            # Multiclass visualization
            height, width = category_mask_np.shape
            vis_image = np.zeros((height, width, 3), dtype=np.uint8)
            for class_name, class_id in MULTICLASS_NAMES.items():
                color_bgr = CLASS_COLORS_BGR.get(class_id, (128, 128, 128))  # Default grey
                vis_image[category_mask_np == class_id] = color_bgr
            return vis_image

    # --- End Helper methods ---

    def detect(
        self,
        image: torch.Tensor,
        model_info: dict,
        output_confidence_masks: bool,
        threshold: float,
        generate_visualization: bool,
        running_mode: str,
        delegate_mode: str,
    ):
        """Performs image segmentation with the configured parameters."""

        # 1. Validate model_info and get model path
        model_path = self.validate_model_info(model_info)

        # 2. Determine which masks to request from the backend
        request_confidence = output_confidence_masks
        request_category = not output_confidence_masks or generate_visualization
        if not request_confidence and not request_category:
            raise ValueError("Internal logic error: No mask type was requested for the segmenter.")

        # 3. Collect all configuration parameters
        config = {
            "running_mode": running_mode,
            "delegate": delegate_mode,
            "output_confidence_masks": request_confidence,
            "output_category_mask": request_category,
        }

        # 4. Initialize or update detector using the new base class method
        detector = self.initialize_or_update_detector(model_path, **config)

        # 5. Perform detection on the image batch
        batch_results_confidence, batch_results_category = detector.detect(image)

        # 6. Process the results into the final output tensors
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        all_primary_masks = []
        all_vis_images = []
        all_category_masks = []
        default_vis_tensor = torch.zeros((h, w, 3), dtype=torch.float32, device=image.device)

        is_multiclass = self.is_multiclass_model(model_info)

        for i in range(batch_size):
            confidence_masks_hw_list = batch_results_confidence[i] if batch_results_confidence else None
            category_mask_hw = batch_results_category[i] if batch_results_category else None

            current_primary_mask_hw = torch.zeros((h, w), dtype=torch.float32, device=image.device)
            current_vis_image_hwc = default_vis_tensor
            current_category_mask_hw = torch.zeros((h, w), dtype=torch.long, device=image.device)

            if output_confidence_masks:
                if confidence_masks_hw_list:
                    conf_mask_hw = confidence_masks_hw_list[0]
                    conf_mask_hw_thresh = torch.where(conf_mask_hw >= threshold, conf_mask_hw, torch.zeros_like(conf_mask_hw))
                    current_primary_mask_hw = conf_mask_hw_thresh
            else:
                if category_mask_hw is not None:
                    current_primary_mask_hw = (category_mask_hw > 0).float()
                    current_category_mask_hw = category_mask_hw

                    if generate_visualization:
                        category_np = category_mask_hw.cpu().numpy().astype(np.uint8)
                        vis_np = self.create_visualization(category_np, is_multiclass)
                        if vis_np is not None:
                            vis_tensor = torch.from_numpy(vis_np).to(dtype=torch.float32, device=image.device) / 255.0
                            current_vis_image_hwc = vis_tensor

            all_primary_masks.append(current_primary_mask_hw)
            all_vis_images.append(current_vis_image_hwc)
            all_category_masks.append(current_category_mask_hw)

        primary_mask_tensor = torch.stack(all_primary_masks, dim=0)
        vis_image_tensor = torch.stack(all_vis_images, dim=0)
        category_mask_tensor = torch.stack(all_category_masks, dim=0)

        return (primary_mask_tensor, vis_image_tensor, category_mask_tensor)


# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeImageSegmenterModelLoader": MediaPipeImageSegmenterModelLoaderNode,
    "MediaPipeImageSegmenter": MediaPipeImageSegmenterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeImageSegmenterModelLoader": "Load Image Segmenter Model (MediaPipe)",
    "MediaPipeImageSegmenter": "Image Segmenter (MediaPipe)",
}
