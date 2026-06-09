"""
CLI entry point for the rebar detection pipeline.

Usage:
    # Run with SAM prompt points (calls SAM server automatically)
    rebar-demo -p ~/DCIM/A_579901304753 --points 836,902 1778,705 --detector mask-grid

    # SAM mask → plane extraction → mask-grid
    rebar-demo -p ~/DCIM/A_579901304753 --sam-mask sam_mask.npy --detector mask-grid

    # Already-refined mask → mask-grid
    rebar-demo -p ~/DCIM/A_579901304753 --refined-mask refined_mask.npy --detector mask-grid

    # Old YOLO path
    rebar-demo -p ~/DCIM/A_579901304753 --sam-mask sam_mask.npy --detector yolo
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from loguru import logger

from .app_api import get_sam_mask as _get_sam_mask


DETECTOR_CHOICES = ("yolo", "mask-grid")


def _parse_point(s: str) -> Tuple[int, int]:
    """Parse 'x,y' into (x, y)."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected x,y but got: {s!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Non-integer coordinate: {s!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebar detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Run with SAM prompt points (positive foreground points)
  rebar-demo -p ~/DCIM/A_579901304753 --points 836,902 1778,705 --detector mask-grid

  # SAM mask -> plane extraction -> mask-grid
  rebar-demo -p ~/DCIM/A_579901304753 --sam-mask sam_mask.npy --detector mask-grid

  # Cached refined mask -> mask-grid
  rebar-demo -p ~/DCIM/A_579901304753 --refined-mask refined_mask.npy --detector mask-grid

  # Reuse <output>/plane_extraction_results/refined_mask.npy
  rebar-demo -p ~/DCIM/A_579901304753 --reuse-refined --detector mask-grid

  # Old YOLO path with existing annotations
  rebar-demo -p ~/DCIM/A_579901304753 --sam-mask sam_mask.npy \\
             --detector yolo --use-existing

  # Explain the resolved steps without running
  rebar-demo -p ~/DCIM/A_579901304753 --sam-mask sam_mask.npy \\
             --detector mask-grid --explain

  # Custom YOLO server + plane threshold
  rebar-demo -p ~/DCIM/A_579901304753 --sam-mask sam_mask.npy \\
             --yolo-url http://localhost:2001 --plane-threshold 0.05
""",
    )

    parser.add_argument("--project", "-p", required=True, type=Path,
                        help="Path to project directory")

    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument("--sam-mask", type=Path,
                            help="Path to a pre-computed SAM mask (.npy)")
    mask_group.add_argument("--refined-mask", type=Path,
                            help="Path to a plane-refined mask (.npy); skips SAM/plane stages")
    mask_group.add_argument("--reuse-refined", action="store_true",
                            help="Use <output>/plane_extraction_results/refined_mask.npy")
    mask_group.add_argument("--points", nargs="+", type=_parse_point, metavar="X,Y",
                            help="SAM positive prompt points as x,y pairs")

    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: <project>/rebar_output)")
    parser.add_argument("--yolo-url", type=str, default=None,
                        help="YOLO server URL (default: from config)")
    parser.add_argument("--no-plane", action="store_true",
                        help="Disable plane extraction step")
    parser.add_argument("--plane-threshold", type=float, default=None,
                        help="Plane distance threshold in metres (default: 0.03)")
    parser.add_argument("--use-existing", action="store_true",
                        help="Reuse existing YOLO annotations if available")
    parser.add_argument("--detector", choices=DETECTOR_CHOICES, default=None,
                        help="Detection backend: yolo or mask-grid (default: config)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Custom rebar_conf.yaml path")
    parser.add_argument("--explain", "--dry-run", dest="explain", action="store_true",
                        help="Print resolved pipeline steps and exit without running")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug-level logging")
    return parser


def _resolve_detector(args, cfg) -> str:
    if args.detector:
        return args.detector
    return "mask-grid" if cfg.use_mask_grid_detector() else "yolo"


def _resolve_input(args, output_path: Path) -> Tuple[str, Path | None]:
    if args.points:
        return "points", None
    if args.refined_mask:
        return "refined", args.refined_mask
    if args.reuse_refined:
        return "refined", output_path / "plane_extraction_results" / "refined_mask.npy"
    return "sam", args.sam_mask


def _log_plan(input_stage: str, input_path: Path | None, detector: str, output_path: Path, run_plane: bool) -> None:
    steps = []
    if input_stage == "points":
        steps.append("SAM server from prompt points")
        steps.append("plane extraction")
    elif input_stage == "sam":
        steps.append(f"load SAM mask: {input_path}")
        if run_plane:
            steps.append("plane extraction")
        else:
            steps.append("skip plane extraction")
    else:
        steps.append(f"load refined mask: {input_path}")
        steps.append("skip SAM and plane extraction")
    steps.append("mask-grid detection" if detector == "mask-grid" else "YOLO detection + line fitting")

    logger.info("Resolved pipeline:")
    logger.info(f"  detector: {detector}")
    logger.info(f"  output:   {output_path}")
    for i, step in enumerate(steps, start=1):
        logger.info(f"  {i}. {step}")


def main(argv=None):
    parser = _build_parser()

    args = parser.parse_args(argv)

    # Configure logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{message}</level>")

    # Validate inputs
    if not args.project.is_dir():
        logger.error(f"Project directory not found: {args.project}")
        return 1

    output_path = (args.output or (args.project / "rebar_output")).resolve()

    from .config import get_rebar_config

    cfg = get_rebar_config(str(args.config) if args.config else None)
    detector = _resolve_detector(args, cfg)

    input_stage, input_path = _resolve_input(args, output_path)
    run_plane = input_stage != "refined" and not args.no_plane

    if args.explain:
        _log_plan(input_stage, input_path, detector, output_path, run_plane)
        return 0

    # Get mask input — prompt points, SAM mask, or already-refined mask.
    prompt_points = None
    if input_stage == "points":
        sam_mask, prompt_points = _get_sam_mask(args.project, args.points, output_path)
    else:
        if input_path is None or not input_path.exists():
            logger.error(f"Mask file not found: {input_path}")
            return 1
        sam_mask = np.load(str(input_path))
        label = "refined mask" if input_stage == "refined" else "SAM mask"
        logger.info(f"Loaded {label}: {sam_mask.shape} dtype={sam_mask.dtype}")

    # Run pipeline
    from .pipeline import run_pipeline_auto

    final_img, json_path = run_pipeline_auto(
        project_path=args.project.resolve(),
        output_path=output_path.resolve(),
        sam_mask=sam_mask,
        server_url=args.yolo_url,
        use_existing_annotations=args.use_existing or None,
        enable_plane_extraction=run_plane,
        plane_distance_threshold=args.plane_threshold,
        use_mask_grid_detector=detector == "mask-grid",
        input_mask_stage="refined" if input_stage == "refined" else "sam",
        config_path=str(args.config) if args.config else None,
        sam_prompt_points=prompt_points,
    )

    logger.info("\nOutputs")
    logger.info(f"  mode:         {detector}")
    logger.info(f"  input stage:  {input_stage}")
    logger.info(f"  result image: {final_img}")
    if json_path:
        logger.info(f"  analysis json: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
