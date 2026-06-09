"""
Rebar Pipeline Orchestrator

Connects the modular pipeline stages:
    SAM Mask → Plane Extraction → Mask Grid Detection
or:
    SAM Mask → Plane Extraction → YOLO Detection → Line Fitting → Spatial Analysis → Visualization

Public API:
    run_pipeline()      — explicit parameters
    run_pipeline_auto()  — automatic config loading with overrides (recommended)
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from .config import ProjectFileNames
from .stages import (
    SamMaskProcessor,
    PlaneExtractor,
    MaskGridDetector,
    KnotDetector,
    LineFitter,
    SpatialAnalyzer,
    Visualizer,
)


def _select_visualization_image(project_path: Path, project_id: str, fallback: Path) -> Path:
    candidates = [
        project_path / ProjectFileNames.RECT_LEFT,
        project_path / ProjectFileNames.RAW_LEFT,
        project_path / f"{project_id}_rect.jpg",
        project_path / f"{project_id}.jpg",
    ]
    return next((candidate for candidate in candidates if candidate.exists()), fallback)


def _write_refined_input_image(output_path: Path, refined_mask: np.ndarray, base_image: np.ndarray) -> Path:
    refined_dir = output_path / "refined_input_results"
    refined_dir.mkdir(parents=True, exist_ok=True)
    segmented = base_image.copy()
    segmented[refined_mask == 0] = [240, 240, 240]
    out_path = refined_dir / "segmented_rebar_refined.png"
    cv2.imwrite(str(out_path), segmented)
    return out_path


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
    use_mask_grid_detector: bool = False,
    input_mask_stage: str = "sam",
    sam_prompt_points=None,
) -> Tuple[Path, Optional[Path]]:
    """
    Run the complete rebar detection pipeline.

    Returns:
        (final_image_path, analysis_json_path) — json_path may be None
        if YOLO detection fails in the YOLO path.
    """
    logger.info("=" * 70)
    logger.info(" Rebar Detection Pipeline")
    logger.info("=" * 70)
    logger.info(f"Project: {project_path}")
    logger.info(f"Output:  {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    project_id = project_path.name
    if input_mask_stage not in {"sam", "refined"}:
        raise ValueError(f"Unsupported input_mask_stage: {input_mask_stage!r}")

    proc = SamMaskProcessor()
    base_image = proc.load_base_image(project_path, project_id)

    if input_mask_stage == "refined":
        logger.info("\n[Step 1] Refined Mask Input")
        mask_binary = proc._convert_to_binary_mask(sam_mask)
        mask_binary = proc._resize_mask_if_needed(mask_binary, base_image.shape[:2])
        refined_mask = mask_binary
        plane_metadata = None
        coverage = float(np.sum(mask_binary > 0) / mask_binary.size)
        logger.info(f"[Refined Mask] coverage={coverage:.1%} ({int(np.sum(mask_binary > 0)):,} px)")
        segmented_image_path = _write_refined_input_image(output_path, refined_mask, base_image)
        logger.info("\n[Step 2] Skipped (input mask is already plane-refined)")
    else:
        # ── Step 1: SAM Mask Processing ──────────────────────────────────
        logger.info("\n[Step 1] SAM Mask Processing")
        mask_result = proc.process_mask(sam_mask, base_image, output_path, project_id)

        mask_binary = mask_result["mask_binary"]
        segmented_image_path = mask_result["segmented_image_path"]

        # ── Step 2: Plane Extraction (optional) ──────────────────────────
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
                segmented_image_path = ext.update_segmented_image(output_path, refined_mask, base_image)
            except Exception as e:
                logger.error(f"Plane extraction failed: {e} — continuing with SAM mask")
        else:
            logger.info("\n[Step 2] Skipped (plane extraction disabled)")

    if use_rectified_for_visualization:
        viz_image_path = _select_visualization_image(project_path, project_id, segmented_image_path)
    else:
        viz_image_path = segmented_image_path

    # ── Step 3A: Mask Grid Detection (YOLO-free path) ─────────────────────
    if use_mask_grid_detector:
        logger.info("\n[Step 3] Mask Grid Detection (YOLO-free)")
        detector = MaskGridDetector()
        grid_mask = refined_mask if refined_mask is not None else mask_binary
        grid_result = detector.detect_grid(
            refined_mask=grid_mask,
            output_path=output_path,
            image_path=viz_image_path,
            refined_image_path=segmented_image_path,
            project_path=project_path,
        )
        final_image_path = grid_result["final_image_path"] or segmented_image_path
        analysis_json_path = grid_result["analysis_json_path"]

        logger.info("\n" + "=" * 70)
        logger.info("Pipeline Complete (Mask Grid)")
        logger.info(f"  Result image: {final_image_path}")
        logger.info(f"  Analysis JSON: {analysis_json_path}")
        logger.info("=" * 70)
        return final_image_path, analysis_json_path

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
    fitter = LineFitter(ai_matcher=ai_matcher)
    line_result = fitter.fit_lines(
        pose_data_path=pose_data_path,
        image_path=viz_image_path,
        output_path=output_path,
        refined_mask=refined_mask,
        plane_metadata=plane_metadata,
        project_path=project_path,
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
    use_mask_grid_detector: Optional[bool] = None,
    input_mask_stage: str = "sam",
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
        use_mask_grid_detector=(
            use_mask_grid_detector if use_mask_grid_detector is not None
            else cfg.use_mask_grid_detector()
        ),
        input_mask_stage=input_mask_stage,
        sam_prompt_points=sam_prompt_points,
    )
