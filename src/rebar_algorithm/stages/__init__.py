"""
Pipeline stages — each module implements one step of the rebar detection pipeline.

Stages:
    1. sam_mask       — Convert SAM mask to segmented image
    2. plane_extraction — 3D plane fitting and surface layer isolation
    3. mask_grid       — Mask-based grid detection without YOLO
    4. knot_detection  — YOLO-based rebar knot detection
    5. line_fitting    — Centerline extraction from detected knots
    6. spatial_analysis — 3D spatial metrics
    7. visualization   — Result overlays and reports
"""

from .sam_mask import SamMaskProcessor
from .plane_extraction import PlaneExtractor, extract_surface_layer_from_sam_mask
from .mask_grid import MaskGridDetector
from .knot_detection import KnotDetector
from .line_fitting import LineFitter, LineFittingAnalyzer
from .spatial_analysis import SpatialAnalyzer
from .visualization import Visualizer

__all__ = [
    "SamMaskProcessor",
    "PlaneExtractor",
    "extract_surface_layer_from_sam_mask",
    "MaskGridDetector",
    "KnotDetector",
    "LineFitter",
    "LineFittingAnalyzer",
    "SpatialAnalyzer",
    "Visualizer",
]
