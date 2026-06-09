"""Reusable application-facing helpers for CLI and lightweight GUIs."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from .config import ProjectFileNames


Point = Tuple[int, int]


@dataclass
class PipelineRunResult:
    """Result paths returned by a GUI/API pipeline run."""

    final_image_path: Path
    analysis_json_path: Optional[Path]
    output_path: Path
    sam_mask_path: Path
    prompt_points_path: Path


def find_project_image(project_path: Path) -> Path:
    """Return the required rectified image for point picking."""
    image_path = project_path / ProjectFileNames.RECT_LEFT
    if image_path.exists():
        return image_path
    raise FileNotFoundError(f"Required image not found: {image_path}")


def get_sam_mask(
    project_path: Path,
    points: Sequence[Point],
    output_path: Path,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Call the SAM server with positive prompt points and return (mask, points_xy)."""
    image_path = find_project_image(project_path)
    image = np.array(Image.open(image_path))
    logger.info(f"Loaded image: {image_path.name} ({image.shape[1]}x{image.shape[0]})")

    from .clients.sam_client import SamClient
    from .config import get_sam_config

    sam_cfg = get_sam_config()
    server_config = sam_cfg.get_server_config()
    client = SamClient(
        server_url=server_config["server_url"],
        model=server_config.get("model", "vit_h"),
        use_tensorrt=server_config.get("use_tensorrt", True),
        alpha=server_config.get("alpha", 0.5),
        timeout=server_config.get("timeout", 30),
    )

    sam_points = [[x, y, 1] for x, y in points]
    logger.info(f"Requesting SAM segmentation with {len(sam_points)} point(s)...")
    masks = client.segment(image, sam_points)
    sam_mask = masks[0]
    logger.info(f"SAM mask: {sam_mask.shape}, coverage={sam_mask.mean() * 100:.1f}%")

    sam_dir = output_path / "sam_segment_results"
    sam_dir.mkdir(parents=True, exist_ok=True)
    sam_mask_path = sam_dir / "sam_mask.npy"
    prompt_points_path = sam_dir / "sam_prompt_points.npy"
    np.save(str(sam_mask_path), sam_mask)
    points_xy = [[x, y] for x, y in points]
    np.save(str(prompt_points_path), np.array(points_xy))

    return sam_mask, points_xy


def run_pipeline_from_points(
    project_path: Path,
    points: Sequence[Point],
    output_path: Optional[Path] = None,
    detector: str = "mask-grid",
    yolo_url: Optional[str] = None,
    use_existing_annotations: Optional[bool] = None,
    enable_plane_extraction: bool = True,
    plane_distance_threshold: Optional[float] = None,
    config_path: Optional[str] = None,
) -> PipelineRunResult:
    """Run the configured pipeline from SAM prompt points."""
    project_path = project_path.resolve()
    output_path = (output_path or (project_path / "rebar_output")).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    sam_mask, prompt_points = get_sam_mask(project_path, points, output_path)

    from .pipeline import run_pipeline_auto

    final_image_path, analysis_json_path = run_pipeline_auto(
        project_path=project_path,
        output_path=output_path,
        sam_mask=sam_mask,
        server_url=yolo_url or None,
        use_existing_annotations=use_existing_annotations,
        enable_plane_extraction=enable_plane_extraction,
        plane_distance_threshold=plane_distance_threshold,
        use_mask_grid_detector=detector == "mask-grid",
        input_mask_stage="sam",
        config_path=config_path,
        sam_prompt_points=prompt_points,
    )

    sam_dir = output_path / "sam_segment_results"
    return PipelineRunResult(
        final_image_path=final_image_path,
        analysis_json_path=analysis_json_path,
        output_path=output_path,
        sam_mask_path=sam_dir / "sam_mask.npy",
        prompt_points_path=sam_dir / "sam_prompt_points.npy",
    )
