"""
CLI entry point for the rebar detection pipeline.

Usage:
    rebar-demo --project ~/DCIM/A_579901304753 --mask mask.npy --output ./output
    rebar-demo --project ~/DCIM/A_579901304753 --mask mask.npy --no-plane
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebar detection pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full pipeline with plane extraction
  rebar-demo --project ~/DCIM/A_579901304753 --mask sam_mask.npy -o ./output

  # Skip plane extraction, use existing YOLO results
  rebar-demo --project ~/DCIM/A_579901304753 --mask sam_mask.npy --no-plane --use-existing

  # Custom YOLO server + plane threshold
  rebar-demo --project ~/DCIM/A_579901304753 --mask sam_mask.npy \\
             --yolo-url http://localhost:2001 --plane-threshold 0.05
""",
    )

    parser.add_argument("--project", "-p", required=True, type=Path,
                        help="Path to project directory (contains cloud.ply, img0.jpg, etc.)")
    parser.add_argument("--mask", "-m", required=True, type=Path,
                        help="Path to SAM mask (.npy file, H×W or H×W×4 RGBA)")
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
    parser.add_argument("--config", type=Path, default=None,
                        help="Custom rebar_conf.yaml path")
    parser.add_argument("--prompt-points", type=Path, default=None,
                        help="Optional .npy file with SAM prompt points [[x,y], ...]")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug-level logging")

    args = parser.parse_args(argv)

    # Configure logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{message}</level>")

    # Validate inputs
    if not args.project.is_dir():
        logger.error(f"Project directory not found: {args.project}")
        return 1
    if not args.mask.exists():
        logger.error(f"Mask file not found: {args.mask}")
        return 1

    # Load mask
    sam_mask = np.load(str(args.mask))
    logger.info(f"Loaded SAM mask: {sam_mask.shape} dtype={sam_mask.dtype}")

    # Output directory
    output_path = args.output or (args.project / "rebar_output")

    # Prompt points
    prompt_points = None
    if args.prompt_points and args.prompt_points.exists():
        prompt_points = np.load(str(args.prompt_points)).tolist()
        logger.info(f"Loaded {len(prompt_points)} prompt points")

    # Run pipeline
    from .pipeline import run_pipeline_auto

    final_img, json_path = run_pipeline_auto(
        project_path=args.project.resolve(),
        output_path=output_path.resolve(),
        sam_mask=sam_mask,
        server_url=args.yolo_url,
        use_existing_annotations=args.use_existing or None,
        enable_plane_extraction=False if args.no_plane else None,
        plane_distance_threshold=args.plane_threshold,
        config_path=str(args.config) if args.config else None,
        sam_prompt_points=prompt_points,
    )

    logger.info(f"\nResult image: {final_img}")
    if json_path:
        logger.info(f"Analysis JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
