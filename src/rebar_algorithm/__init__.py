"""
Rebar Detection & Analysis Pipeline

A standalone demo of the rebar detection pipeline:
SAM → Plane Extraction → YOLO Knot Detection → Line Fitting → Spatial Analysis

Usage:
    from rebar_algorithm import run_pipeline, run_pipeline_auto

    final_img, json_path = run_pipeline_auto(
        project_path, output_path, sam_mask
    )
"""

from .pipeline import run_pipeline, run_pipeline_auto

__all__ = ["run_pipeline", "run_pipeline_auto"]
__version__ = "0.1.0"
