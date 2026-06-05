#!/usr/bin/env python3
"""
Verify extracted rebar_algorithm against the original JetsonReborn pipeline.

Runs SAM segmentation + full pipeline on stereo project 222 and writes
results to <project>/rebar_output_verify for comparison with
<project>/rebar_output_sam (the original GUI's output).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rebar_algorithm.clients.sam_client import SamClient
from rebar_algorithm.pipeline import run_pipeline_auto

PROJECT_PATH = Path("/Users/andyliu/DCIM/222")
OUTPUT_PATH = PROJECT_PATH / "rebar_output_verify"

POSITIVE_POINTS = [
    (1392, 1240),
    (2138, 1358),
    (1911, 1981),
    (2879, 2273),
]


def get_sam_mask() -> np.ndarray:
    """Call SAM server with rect_left.jpg and the given positive points."""
    image_path = PROJECT_PATH / "rect_left.jpg"
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    logger.info(f"Loaded {image_path.name}: {image_rgb.shape}")

    client = SamClient(
        server_url="https://stereo-hq.andy6.link",
        model="vit_h",
        use_tensorrt=True,
        alpha=0.5,
        timeout=60,
    )

    points_with_label = [[x, y, 1] for x, y in POSITIVE_POINTS]
    logger.info(f"SAM prompt points: {POSITIVE_POINTS}")

    masks = client.segment(image_rgb, points_with_label)
    logger.info(f"SAM returned {len(masks)} mask(s), shape={masks[0].shape}")

    mask = masks[0]
    coverage = np.sum(mask > 0) / mask.size
    logger.info(f"SAM mask coverage: {coverage:.1%}")
    return mask


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    logger.info("=" * 70)
    logger.info(" Verification: extracted algorithm vs original GUI")
    logger.info("=" * 70)

    # Step 1: Get SAM mask
    logger.info("\n--- SAM Segmentation ---")
    sam_mask = get_sam_mask()

    # Save the SAM mask for inspection
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_PATH / "sam_mask_from_server.npy"), sam_mask)
    logger.info(f"Saved SAM mask to {OUTPUT_PATH / 'sam_mask_from_server.npy'}")

    # Step 2: Run pipeline
    logger.info("\n--- Running Pipeline ---")
    prompt_points = list(POSITIVE_POINTS)

    final_img, json_path = run_pipeline_auto(
        project_path=PROJECT_PATH.resolve(),
        output_path=OUTPUT_PATH.resolve(),
        sam_mask=sam_mask,
        enable_plane_extraction=True,
        sam_prompt_points=prompt_points,
    )

    logger.info("\n" + "=" * 70)
    logger.info(" Verification Complete")
    logger.info(f"  Result image:  {final_img}")
    logger.info(f"  Analysis JSON: {json_path}")
    logger.info(f"  Compare with:  {PROJECT_PATH / 'rebar_output_sam'}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
