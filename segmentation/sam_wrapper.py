import numpy as np
import torch
from typing import Optional
from PIL import Image

# Try importing SAM3 from the GitHub package
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sam3 not installed. SAMWrapper will fail if used. Error: {e}")
    SAM3_AVAILABLE = False

class SAMWrapper:
    """
    Wrapper for the Segment Anything Model 3 (SAM3) using the GitHub package.
    Uses text prompts for segmentation.
    """
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: 'cuda' or 'cpu'.
        """
        self.device = device
        if not SAM3_AVAILABLE:
            raise ImportError("sam3 library is missing. Install from: https://github.com/facebookresearch/sam3")

        print(f"Loading SAM3 model from HuggingFace...")
        self.model = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=True,
            device=device
        )
        self.processor = Sam3Processor(self.model)
        self.current_image = None
        print("SAM3 model loaded successfully.")

    def set_image(self, image: np.ndarray):
        """
        Set the current image for processing.
        Args:
            image: (H, W, C) RGB image (uint8).
        """
        if isinstance(image, np.ndarray):
            self.current_image = Image.fromarray(image)
        else:
            self.current_image = image

    def segment_with_text(self, image: np.ndarray, text_prompt: str, threshold: float = 0.5) -> np.ndarray:
        """
        Segment image using text prompt.
        Args:
            image: (H, W, C) RGB image (uint8)
            text_prompt: Text description of what to segment
            threshold: Confidence threshold for detections
        Returns:
            Binary mask (H, W) with all detected instances combined
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Set confidence threshold
        self.processor.confidence_threshold = threshold

        # Process image and text prompt using the stateful API
        state = self.processor.set_image(pil_image)
        state = self.processor.set_text_prompt(text_prompt, state)

        # Extract masks from state
        if 'masks' in state and state['masks'] is not None and len(state['masks']) > 0:
            masks = state['masks']  # Already thresholded at 0.5
            # Combine all masks - masks shape is (N, 1, H, W) where N is number of detections
            combined_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
        else:
            # No detections - return empty mask
            h, w = image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)

        return combined_mask

    def segment_sky(self, image: np.ndarray) -> np.ndarray:
        """
        Segments the sky using text prompt.
        """
        return self.segment_with_text(image, "sky", threshold=0.3)

    def segment_snow(self, image: np.ndarray, sky_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segments snow using text prompt.
        """
        snow_mask = self.segment_with_text(image, "snow", threshold=0.4)

        if sky_mask is not None:
            # Ensure snow does not overlap with sky
            snow_mask = np.logical_and(snow_mask, np.logical_not(sky_mask)).astype(np.uint8)

        return snow_mask
