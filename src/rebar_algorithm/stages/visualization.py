"""
Step 6: Result Visualization.

Creates visual outputs: line overlays, metric annotations, comparison views.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class Visualizer:
    """Result visualization generator for rebar detection (Step 6)."""

    def create_main_visualization(self, analyzer, image_path, output_path, **kw):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load: {image_path}")
        viz = analyzer.visualize_fitted_lines(img=img, include_text_overlay=kw.get("include_metrics", True))
        out = output_path / "line_fitting_results" / "line_fitting_visualization.png"
        cv2.imwrite(str(out), viz)
        return out

    def create_segmented_overlay(self, analyzer, seg_path, output_path):
        if not seg_path.exists():
            return None
        img = cv2.imread(str(seg_path))
        if img is None:
            return None
        viz = analyzer.visualize_fitted_lines(img=img)
        out = output_path / "line_fitting_results" / "lines_on_segmented_rebar.png"
        cv2.imwrite(str(out), viz)
        return out

    def create_comparison_view(self, orig_path, result_path, output_path, title="Before / After"):
        orig = cv2.imread(str(orig_path))
        result = cv2.imread(str(result_path))
        if orig is None or result is None:
            raise ValueError("Cannot load images for comparison")
        if orig.shape != result.shape:
            result = cv2.resize(result, (orig.shape[1], orig.shape[0]))
        comp = np.hstack([orig, result])
        bar = np.zeros((50, comp.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        comp = np.vstack([bar, comp])
        out = output_path / "comparison_before_after.png"
        cv2.imwrite(str(out), comp)
        return out
