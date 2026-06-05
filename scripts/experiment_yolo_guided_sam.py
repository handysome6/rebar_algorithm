#!/usr/bin/env python3
"""
Experiment: YOLO-guided SAM segmentation.

1. Send rect_left.jpg to YOLO → get all rebar knot detections
2. Pick top-10 knot centers by confidence
3. Use those as SAM prompt points to generate the mask
4. Run the rest of the pipeline as normal
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rebar_algorithm.clients.sam_client import SamClient
from rebar_algorithm.clients.yolo_client import YoloClient
from rebar_algorithm.clients.yolo_parser import YoloResultParser
from rebar_algorithm.pipeline import run_pipeline_auto

PROJECT_PATH = Path("/Users/andyliu/DCIM/222")
OUTPUT_PATH = PROJECT_PATH / "rebar_output_yolo_guided"
TOP_K = 10


def step1_yolo_on_raw_image() -> tuple[list[dict], tuple[int, int]]:
    """Send rect_left.jpg directly to YOLO and return knot centers + image size."""
    image_path = PROJECT_PATH / "rect_left.jpg"
    yolo_dir = OUTPUT_PATH / "yolo_seed_results"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    pose_path = yolo_dir / "pose_data.json"
    if not pose_path.exists():
        client = YoloClient.from_config()
        client.process_image(str(image_path), str(yolo_dir))
    else:
        logger.info("[Step 1] Using existing YOLO seed results")

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    parser = YoloResultParser(str(pose_path), image_width=w, image_height=h)

    centers = parser.get_centers_info()
    logger.info(f"[Step 1] YOLO detected {len(centers)} knots on rect_left.jpg ({w}x{h})")
    return centers, (w, h)


def step2_pick_top_k(
    centers: list[dict],
    k: int = TOP_K,
    image_size: tuple[int, int] = (4032, 3036),
    edge_margin_ratio: float = 0.05,
) -> list[tuple[int, int]]:
    """
    Pick k spatially-spread knot centers, filtering edge outliers.

    1. Discard points within edge_margin_ratio of the image border.
    2. Greedy farthest-point sampling weighted by confidence:
       - Seed with the highest-confidence point.
       - Each subsequent pick maximises  min_dist_to_selected * confidence.
    """
    w, h = image_size
    mx = w * edge_margin_ratio
    my = h * edge_margin_ratio

    filtered = [
        c for c in centers
        if mx <= c["center_x"] <= w - mx and my <= c["center_y"] <= h - my
    ]
    logger.info(f"[Step 2] {len(centers)} knots → {len(filtered)} after filtering {edge_margin_ratio:.0%} edge margin")

    if len(filtered) <= k:
        selected = sorted(filtered, key=lambda c: c["object_confidence"], reverse=True)
    else:
        coords = np.array([[c["center_x"], c["center_y"]] for c in filtered])
        confs = np.array([c["object_confidence"] for c in filtered])

        seed_idx = int(np.argmax(confs))
        chosen = [seed_idx]
        remaining = set(range(len(filtered))) - {seed_idx}

        for _ in range(k - 1):
            chosen_coords = coords[chosen]
            best_idx, best_score = -1, -1.0
            for idx in remaining:
                min_dist = np.min(np.linalg.norm(chosen_coords - coords[idx], axis=1))
                score = min_dist * confs[idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx
            chosen.append(best_idx)
            remaining.discard(best_idx)

        selected = [filtered[i] for i in chosen]

    points = [(int(c["center_x"]), int(c["center_y"])) for c in selected]

    logger.info(f"[Step 2] Selected {len(points)} spatially-spread points:")
    for i, c in enumerate(selected):
        logger.info(f"  {i+1}. ({points[i][0]}, {points[i][1]}) conf={c['object_confidence']:.3f}")

    return points


def step3_sam_segment(points: list[tuple[int, int]]) -> np.ndarray:
    """Call SAM server with the selected points as positive prompts."""
    image_path = PROJECT_PATH / "rect_left.jpg"
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    from rebar_algorithm.config import get_sam_config
    sam_cfg = get_sam_config()
    cfg = sam_cfg.get_server_config()

    client = SamClient(
        server_url=cfg.get("server_url", "https://stereo-hq.andy6.link"),
        model=cfg.get("model", "vit_h"),
        use_tensorrt=cfg.get("use_tensorrt", True),
        alpha=cfg.get("alpha", 0.5),
        timeout=cfg.get("timeout", 30),
    )

    points_with_label = [[x, y, 1] for x, y in points]
    masks = client.segment(image_rgb, points_with_label)
    mask = masks[0]

    coverage = np.sum(mask > 0) / mask.size
    logger.info(f"[Step 3] SAM mask coverage: {coverage:.1%}")

    np.save(str(OUTPUT_PATH / "sam_mask_yolo_guided.npy"), mask)
    return mask


def step4_run_pipeline(sam_mask: np.ndarray, prompt_points: list[tuple[int, int]]):
    """Run the rest of the pipeline with the YOLO-guided SAM mask."""
    final_img, json_path = run_pipeline_auto(
        project_path=PROJECT_PATH.resolve(),
        output_path=OUTPUT_PATH.resolve(),
        sam_mask=sam_mask,
        enable_plane_extraction=True,
        sam_prompt_points=list(prompt_points),
    )
    return final_img, json_path


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    logger.info("=" * 70)
    logger.info(" Experiment: YOLO-guided SAM segmentation")
    logger.info("=" * 70)

    centers, image_size = step1_yolo_on_raw_image()
    top_points = step2_pick_top_k(centers, TOP_K, image_size=image_size)
    sam_mask = step3_sam_segment(top_points)
    final_img, json_path = step4_run_pipeline(sam_mask, top_points)

    logger.info("\n" + "=" * 70)
    logger.info(" Experiment Complete")
    logger.info(f"  Result image:  {final_img}")
    logger.info(f"  Analysis JSON: {json_path}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
