"""
CLI entry point for the rebar detection pipeline.

Usage:
    # Run with SAM prompt points (calls SAM server automatically)
    rebar-demo -p ~/DCIM/A_579901304753 --points 836,902 1778,705 2020,1649

    # Run with a pre-computed SAM mask
    rebar-demo -p ~/DCIM/A_579901304753 --mask mask.npy --output ./output

    # Skip plane extraction, use existing YOLO results
    rebar-demo -p ~/DCIM/A_579901304753 --mask mask.npy --no-plane --use-existing
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger


def _parse_point(s: str) -> Tuple[int, int]:
    """Parse 'x,y' into (x, y)."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected x,y but got: {s!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Non-integer coordinate: {s!r}")


def _get_sam_mask(
    project_path: Path,
    points: List[Tuple[int, int]],
    output_path: Path,
) -> Tuple[np.ndarray, list]:
    """Call the SAM server with positive prompt points and return (mask, points_xy)."""
    from PIL import Image

    from .clients.sam_client import SamClient
    from .config import get_sam_config

    image_path = project_path / "rect_left.jpg"
    if not image_path.exists():
        image_path = project_path / "raw_left.jpg"
    if not image_path.exists():
        raise FileNotFoundError(f"No image found in {project_path}")

    image = np.array(Image.open(image_path))
    logger.info(f"Loaded image: {image_path.name} ({image.shape[1]}x{image.shape[0]})")

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
    logger.info(f"SAM mask: {sam_mask.shape}, coverage={sam_mask.mean()*100:.1f}%")

    sam_dir = output_path / "sam_segment_results"
    sam_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(sam_dir / "sam_mask.npy"), sam_mask)
    points_xy = [[x, y] for x, y in points]
    np.save(str(sam_dir / "sam_prompt_points.npy"), np.array(points_xy))

    return sam_mask, points_xy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebar detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Run with SAM prompt points (positive foreground points)
  rebar-demo -p ~/DCIM/A_579901304753 --points 836,902 1778,705 2020,1649

  # Run with a pre-computed SAM mask
  rebar-demo -p ~/DCIM/A_579901304753 --mask sam_mask.npy -o ./output

  # Skip plane extraction, use existing YOLO results
  rebar-demo -p ~/DCIM/A_579901304753 --mask sam_mask.npy --no-plane --use-existing

  # Custom YOLO server + plane threshold
  rebar-demo -p ~/DCIM/A_579901304753 --mask sam_mask.npy \\
             --yolo-url http://localhost:2001 --plane-threshold 0.05
""",
    )

    parser.add_argument("--project", "-p", required=True, type=Path,
                        help="Path to project directory")

    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument("--mask", "-m", type=Path,
                            help="Path to pre-computed SAM mask (.npy)")
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
    parser.add_argument("--config", type=Path, default=None,
                        help="Custom rebar_conf.yaml path")
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

    output_path = args.output or (args.project / "rebar_output")

    # Get SAM mask — either from file or by calling the SAM server
    prompt_points = None
    if args.points:
        sam_mask, prompt_points = _get_sam_mask(args.project, args.points, output_path)
    else:
        if not args.mask.exists():
            logger.error(f"Mask file not found: {args.mask}")
            return 1
        sam_mask = np.load(str(args.mask))
        logger.info(f"Loaded SAM mask: {sam_mask.shape} dtype={sam_mask.dtype}")

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
