"""
Step 1: SAM Mask Processing

Converts SAM segmentation masks to segmented rebar images.
Handles RGBA, single-channel, and 2D mask formats.
"""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from loguru import logger

from ..config import ProjectFileNames


class SamMaskProcessor:
    """Process SAM masks for the rebar detection pipeline (Step 1)."""

    def process_mask(
        self,
        sam_mask: np.ndarray,
        base_image: np.ndarray,
        output_path: Path,
        project_id: str,
    ) -> Dict:
        """
        Convert SAM mask to a segmented rebar image.

        Returns dict with: mask_binary, segmented_image, segmented_image_path,
        mask_path, coverage, mask_pixels.
        """
        logger.info(f"[SAM Mask] mask={sam_mask.shape}, image={base_image.shape}")

        mask_binary = self._convert_to_binary_mask(sam_mask)
        mask_binary = self._resize_mask_if_needed(mask_binary, base_image.shape[:2])
        segmented = self._create_segmented_image(base_image, mask_binary)

        out_dir = output_path / "sam_segment_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        seg_path = out_dir / "segmented_rebar.png"
        cv2.imwrite(str(seg_path), segmented)
        mask_path = out_dir / "rebar_mask.npy"
        np.save(str(mask_path), mask_binary)

        coverage = float(np.sum(mask_binary > 0) / mask_binary.size)
        mask_pixels = int(np.sum(mask_binary > 0))
        logger.info(f"[SAM Mask] coverage={coverage:.1%} ({mask_pixels:,} px)")

        return {
            "mask_binary": mask_binary,
            "segmented_image": segmented,
            "segmented_image_path": seg_path,
            "mask_path": mask_path,
            "coverage": coverage,
            "mask_pixels": mask_pixels,
        }

    @staticmethod
    def _convert_to_binary_mask(sam_mask: np.ndarray) -> np.ndarray:
        if sam_mask.ndim == 3 and sam_mask.shape[-1] == 4:
            return (sam_mask[..., 3] > 0).astype(np.uint8)
        elif sam_mask.ndim == 3 and sam_mask.shape[-1] == 1:
            return (sam_mask[..., 0] > 0).astype(np.uint8)
        elif sam_mask.ndim == 2:
            return (sam_mask > 0.5).astype(np.uint8)
        raise ValueError(f"Unsupported SAM mask shape: {sam_mask.shape}")

    @staticmethod
    def _resize_mask_if_needed(mask: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
        if mask.shape[:2] != target:
            mask = cv2.resize(mask, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    @staticmethod
    def _create_segmented_image(
        base: np.ndarray,
        mask: np.ndarray,
        bg: Tuple[int, int, int] = (240, 240, 240),
    ) -> np.ndarray:
        out = base.copy()
        out[mask == 0] = bg
        return out

    @staticmethod
    def load_base_image(project_path: Path, project_id: str) -> np.ndarray:
        """Load the required rectified base image."""
        image_path = project_path / ProjectFileNames.RECT_LEFT
        if not image_path.exists():
            raise FileNotFoundError(f"Required image not found: {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read required image: {image_path}")
        logger.info(f"[SAM Mask] Base image: {image_path.name}")
        return img
