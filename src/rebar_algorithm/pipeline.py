"""
Rebar Pipeline Orchestrator

Connects the 6 modular pipeline stages:
    SAM Mask → Plane Extraction → YOLO Detection → Line Fitting → Spatial Analysis → Visualization

Public API:
    run_pipeline()      — explicit parameters
    run_pipeline_auto()  — automatic config loading with overrides (recommended)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from .config import ProjectFileNames
from .stages import (
    SamMaskProcessor,
    PlaneExtractor,
    KnotDetector,
    LineFitter,
    SpatialAnalyzer,
    Visualizer,
)


def run_pipeline(
    project_path: Path,
    output_path: Path,
    sam_mask: np.ndarray,
    server_url: str = "http://localhost:2001",
    use_existing_annotations: bool = False,
    ai_matcher=None,
    use_rectified_for_visualization: bool = True,
    enable_plane_extraction: bool = True,
    plane_distance_threshold: float = 0.03,
    sam_prompt_points=None,
) -> Tuple[Path, Optional[Path]]:
    """
    Run the complete rebar detection pipeline.

    Returns:
        (final_image_path, analysis_json_path)  — json_path may be None
        if YOLO detection fails.
    """
    logger.info("=" * 70)
    logger.info(" Rebar Detection Pipeline")
    logger.info("=" * 70)
    logger.info(f"Project: {project_path}")
    logger.info(f"Output:  {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    project_id = project_path.name

    # ── Step 1: SAM Mask Processing ──────────────────────────────────────
    logger.info("\n[Step 1] SAM Mask Processing")
    proc = SamMaskProcessor()
    base_image = proc.load_base_image(project_path, project_id)
    mask_result = proc.process_mask(sam_mask, base_image, output_path, project_id)

    mask_binary = mask_result["mask_binary"]
    segmented_image_path = mask_result["segmented_image_path"]
    sam_coverage = mask_result["coverage"]

    # ── Step 2: Plane Extraction (optional) ──────────────────────────────
    refined_mask = None
    plane_metadata = None

    if enable_plane_extraction:
        logger.info("\n[Step 2] 3D Plane Extraction")
        try:
            ext = PlaneExtractor(distance_threshold=plane_distance_threshold)
            plane_result = ext.extract_surface_layer(
                sam_mask=mask_binary,
                project_path=project_path,
                output_path=output_path,
                sam_prompt_points=sam_prompt_points,
            )
            refined_mask = plane_result["refined_mask_original_res"]
            mask_binary = refined_mask
            plane_metadata = plane_result["plane_metadata"]
            ext.update_segmented_image(segmented_image_path, refined_mask, base_image)
        except Exception as e:
            logger.error(f"Plane extraction failed: {e} — continuing with SAM mask")
    else:
        logger.info("\n[Step 2] Skipped (plane extraction disabled)")

    # ── Step 3: YOLO Knot Detection ──────────────────────────────────────
    logger.info("\n[Step 3] YOLO Rebar Knot Detection")
    detector = KnotDetector(server_url=server_url)
    try:
        knot_result = detector.detect_knots(
            segmented_image_path, output_path, use_existing=use_existing_annotations,
        )
        pose_data_path = knot_result["pose_data_path"]
        if not detector.validate_detection_results(pose_data_path):
            logger.warning("No YOLO results — stopping at segmentation")
            return segmented_image_path, None
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        return segmented_image_path, None

    # ── Step 4: Line Fitting ─────────────────────────────────────────────
    logger.info("\n[Step 4] Rebar Line Fitting")
    if use_rectified_for_visualization:
        candidates = [
            project_path / ProjectFileNames.RECT_LEFT,
            project_path / ProjectFileNames.RAW_LEFT,
            project_path / f"{project_id}_rect.jpg",
            project_path / f"{project_id}.jpg",
        ]
        viz_image_path = next((c for c in candidates if c.exists()), segmented_image_path)
    else:
        viz_image_path = segmented_image_path

    fitter = LineFitter(ai_matcher=ai_matcher)
    line_result = fitter.fit_lines(
        pose_data_path=pose_data_path,
        image_path=viz_image_path,
        output_path=output_path,
        refined_mask=refined_mask,
        plane_metadata=plane_metadata,
        depth_meter_path=project_path / "depth_meter.npy",
    )
    final_image_path = line_result["final_image_path"]
    analysis_json_path = line_result["analysis_json_path"]

    # ── Step 5: Spatial Analysis (optional) ───────────────────────────────
    if ai_matcher:
        logger.info("\n[Step 5] 3D Spatial Analysis")
        spatial = SpatialAnalyzer(ai_matcher=ai_matcher)
        spatial.calculate_3d_metrics(analyzer=line_result["analyzer"], pose_data_path=pose_data_path)
    else:
        logger.info("\n[Step 5] Skipped (no AIMatcher)")

    # ── Step 6: Visualization ────────────────────────────────────────────
    logger.info("\n[Step 6] Visualization")
    fitter.create_segmented_overlay(segmented_image_path, output_path)

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete")
    logger.info(f"  Result image: {final_image_path}")
    logger.info(f"  Analysis JSON: {analysis_json_path}")
    logger.info("=" * 70)

    return final_image_path, analysis_json_path


def run_pipeline_auto(
    project_path: Path,
    output_path: Path,
    sam_mask: np.ndarray,
    server_url: Optional[str] = None,
    use_existing_annotations: Optional[bool] = None,
    ai_matcher=None,
    use_rectified_for_visualization: Optional[bool] = None,
    enable_plane_extraction: Optional[bool] = None,
    plane_distance_threshold: Optional[float] = None,
    config_path: Optional[str] = None,
    sam_prompt_points=None,
) -> Tuple[Path, Optional[Path]]:
    """
    Run the pipeline with automatic config loading.

    Parameters set to None fall through to config defaults.  Explicit values
    override the config.
    """
    from .config import get_rebar_config

    cfg = get_rebar_config(config_path)

    return run_pipeline(
        project_path=project_path,
        output_path=output_path,
        sam_mask=sam_mask,
        server_url=server_url if server_url is not None else cfg.get_server_url(),
        use_existing_annotations=(
            use_existing_annotations if use_existing_annotations is not None
            else cfg.use_existing_annotations()
        ),
        ai_matcher=ai_matcher,
        use_rectified_for_visualization=(
            use_rectified_for_visualization if use_rectified_for_visualization is not None
            else cfg.use_rectified_for_visualization()
        ),
        enable_plane_extraction=(
            enable_plane_extraction if enable_plane_extraction is not None
            else cfg.is_plane_extraction_enabled()
        ),
        plane_distance_threshold=(
            plane_distance_threshold if plane_distance_threshold is not None
            else cfg.get_plane_distance_threshold()
        ),
        sam_prompt_points=sam_prompt_points,
    )
