"""
Step 6: Result Visualization.

Creates visual outputs: line overlays, metric annotations, comparison views.

Additional overlays (ported from JetsonReborn GUI):
    draw_spacing_overlay()   — H/V lines + 3D distance between consecutive grid nodes
    draw_knot_boxes_overlay() — H/V lines + boxes around each detected knot
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger


def _line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-8:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


def _build_grid_nodes(
    analysis_data: Dict,
    image_shape: Tuple[int, int],
) -> Optional[Tuple[List[List[Optional[Tuple[float, float]]]], list, list]]:
    """Build a 2D grid of H/V line intersection points.

    Returns (nodes[i_h][j_v], h_lines_xy, v_lines_xy) or None.
    """
    line_details = analysis_data.get("line_details") or {}
    h_lines = line_details.get("horizontal") or []
    v_lines = line_details.get("vertical") or []

    def _coords(item):
        c = item.get("coordinates") if isinstance(item, dict) else None
        if not c:
            return None
        try:
            x1, y1, x2, y2 = c
            return float(x1), float(y1), float(x2), float(y2)
        except Exception:
            return None

    h_coords = [c for c in (_coords(it) for it in h_lines) if c is not None]
    v_coords = [c for c in (_coords(it) for it in v_lines) if c is not None]
    if not h_coords or not v_coords:
        return None

    H, W = image_shape[:2]
    h_sorted = sorted(
        [(float((y1 + y2) * 0.5), (x1, y1, x2, y2)) for x1, y1, x2, y2 in h_coords],
        key=lambda t: t[0],
    )
    v_sorted = sorted(
        [(float((x1 + x2) * 0.5), (x1, y1, x2, y2)) for x1, y1, x2, y2 in v_coords],
        key=lambda t: t[0],
    )

    h_lines_xy = [((x1, y1), (x2, y2)) for _, (x1, y1, x2, y2) in h_sorted]
    v_lines_xy = [((x1, y1), (x2, y2)) for _, (x1, y1, x2, y2) in v_sorted]
    n_h, n_v = len(h_lines_xy), len(v_lines_xy)
    if n_h < 2 or n_v < 2:
        return None

    nodes: List[List[Optional[Tuple[float, float]]]] = [
        [None for _ in range(n_v)] for _ in range(n_h)
    ]
    for i in range(n_h):
        for j in range(n_v):
            p = _line_intersection(
                h_lines_xy[i][0], h_lines_xy[i][1],
                v_lines_xy[j][0], v_lines_xy[j][1],
            )
            if p is None:
                continue
            x = max(0.0, min(p[0], float(W - 1)))
            y = max(0.0, min(p[1], float(H - 1)))
            nodes[i][j] = (x, y)

    return nodes, h_lines_xy, v_lines_xy


def _compute_3d_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    xyz_map: np.ndarray,
) -> Optional[float]:
    """Compute 3D Euclidean distance (mm) between two pixel positions via xyz_map."""
    H, W = xyz_map.shape[:2]
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    x1, y1 = max(0, min(x1, W - 1)), max(0, min(y1, H - 1))
    x2, y2 = max(0, min(x2, W - 1)), max(0, min(y2, H - 1))

    pt1 = xyz_map[y1, x1].astype(np.float64)
    pt2 = xyz_map[y2, x2].astype(np.float64)

    if not (np.isfinite(pt1).all() and np.isfinite(pt2).all()):
        return None
    if pt1[2] <= 0 or pt2[2] <= 0:
        return None

    dist_m = float(np.linalg.norm(pt1 - pt2))
    return dist_m * 1000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_spacing_overlay(
    base_image: np.ndarray,
    analysis_data: Dict,
    xyz_map: np.ndarray,
    draw_lines: bool = True,
) -> np.ndarray:
    """Draw H/V lines with 3D distance labels between consecutive grid nodes.

    Args:
        base_image: BGR image (rect_left.jpg or line fitting visualization).
        analysis_data: Line fitting analysis dict (from line_fitting_analysis.json).
        xyz_map: Per-pixel 3D coordinate map, shape (H, W, 3) in metres.
        draw_lines: Whether to draw the fitted H/V lines on the image.

    Returns:
        Annotated BGR image.
    """
    img = base_image.copy()
    grid = _build_grid_nodes(analysis_data, img.shape)
    if grid is None:
        logger.warning("[Visualizer] Cannot build grid — not enough lines")
        return img

    nodes, h_lines_xy, v_lines_xy = grid
    n_h, n_v = len(h_lines_xy), len(v_lines_xy)
    H, W = img.shape[:2]

    if draw_lines:
        for seg in h_lines_xy:
            cv2.line(img, _int_pt(seg[0]), _int_pt(seg[1]), (0, 255, 0), 6)
        for seg in v_lines_xy:
            cv2.line(img, _int_pt(seg[0]), _int_pt(seg[1]), (255, 0, 0), 6)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.26
    thickness = 3

    def _draw_label(text: str, x: int, y: int, color: Tuple[int, int, int]):
        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Horizontal spacing (between consecutive vertical lines on same H-line)
    for i in range(n_h):
        for j in range(n_v - 1):
            p1, p2 = nodes[i][j], nodes[i][j + 1]
            if p1 is None or p2 is None:
                continue
            dist_mm = _compute_3d_distance(p1, p2, xyz_map)
            if dist_mm is None or not np.isfinite(dist_mm):
                continue
            txt = f"{dist_mm:.1f}"
            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
            mx = int((p1[0] + p2[0]) * 0.5)
            my = int((p1[1] + p2[1]) * 0.5)
            x = max(2, min(int(mx - tw / 2), W - tw - 2))
            y = max(th + 2, min(int(my - 6 - (6 if i % 2 else 0)), H - 2))
            _draw_label(txt, x, y, (255, 80, 0))

    # Vertical spacing (between consecutive horizontal lines on same V-line)
    for j in range(n_v):
        for i in range(n_h - 1):
            p1, p2 = nodes[i][j], nodes[i + 1][j]
            if p1 is None or p2 is None:
                continue
            dist_mm = _compute_3d_distance(p1, p2, xyz_map)
            if dist_mm is None or not np.isfinite(dist_mm):
                continue
            txt = f"{dist_mm:.1f}"
            (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
            mx = int((p1[0] + p2[0]) * 0.5)
            my = int((p1[1] + p2[1]) * 0.5)
            x = max(2, min(int(mx - tw - 6 - (6 if j % 2 else 0)), W - tw - 2))
            y = max(th + 2, min(int(my + th // 2), H - 2))
            _draw_label(txt, x, y, (0, 255, 0))

    return img


def draw_knot_boxes_overlay(
    base_image: np.ndarray,
    analysis_data: Dict,
    yolo_json_path: Union[str, Path],
    draw_lines: bool = True,
) -> np.ndarray:
    """Draw H/V lines with boxes around each YOLO-detected knot.

    Falls back to grid intersection points if YOLO data is unavailable.

    Args:
        base_image: BGR image.
        analysis_data: Line fitting analysis dict.
        yolo_json_path: Path to pose_data.json from YOLO detection.
        draw_lines: Whether to draw the fitted H/V lines on the image.

    Returns:
        Annotated BGR image.
    """
    img = base_image.copy()
    H, W = img.shape[:2]

    # Draw fitted lines
    if draw_lines:
        line_details = analysis_data.get("line_details") or {}
        for item in line_details.get("horizontal") or []:
            c = item.get("coordinates")
            if c:
                cv2.line(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 255, 0), 6)
        for item in line_details.get("vertical") or []:
            c = item.get("coordinates")
            if c:
                cv2.line(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (255, 0, 0), 6)

    # Load YOLO bounding boxes
    centers: List[Tuple[float, float]] = []
    sizes: List[float] = []
    yolo_path = Path(yolo_json_path)
    if yolo_path.exists():
        try:
            with open(yolo_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    bbox = item.get("bbox") if isinstance(item, dict) else None
                    if not isinstance(bbox, dict):
                        continue

                    x1 = y1 = x2 = y2 = None
                    if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                        x1, y1 = float(bbox["x1"]), float(bbox["y1"])
                        x2, y2 = float(bbox["x2"]), float(bbox["y2"])
                        if 0.0 <= x2 <= 1.5 and 0.0 <= y2 <= 1.5:
                            x1 *= W; x2 *= W; y1 *= H; y2 *= H
                    elif all(k in bbox for k in ("x", "y", "w", "h")):
                        bx, by = float(bbox["x"]), float(bbox["y"])
                        bw, bh = float(bbox["w"]), float(bbox["h"])
                        if 0.0 <= bx <= 1.5 and 0.0 <= by <= 1.5:
                            bx *= W; bw *= W; by *= H; bh *= H
                        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

                    if x1 is None:
                        continue
                    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
                        continue

                    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                    centers.append((cx, cy))
                    size = min(abs(x2 - x1), abs(y2 - y1))
                    if np.isfinite(size) and size > 0:
                        sizes.append(float(size))
        except Exception as e:
            logger.warning(f"[Visualizer] Failed to load YOLO data: {e}")

    # Fallback: use grid intersection nodes
    use_grid = False
    node_centers: List[Tuple[float, float]] = []
    node_spacings: List[float] = []
    if not centers:
        use_grid = True
        grid = _build_grid_nodes(analysis_data, img.shape)
        if grid is not None:
            nodes, h_lines_xy, v_lines_xy = grid
            n_h, n_v = len(h_lines_xy), len(v_lines_xy)
            for i in range(n_h):
                for j in range(n_v):
                    if nodes[i][j] is not None:
                        node_centers.append(nodes[i][j])
            for i in range(n_h):
                for j in range(n_v - 1):
                    p1, p2 = nodes[i][j], nodes[i][j + 1]
                    if p1 and p2:
                        d = abs(p2[0] - p1[0])
                        if np.isfinite(d) and d > 0:
                            node_spacings.append(d)
            for j in range(n_v):
                for i in range(n_h - 1):
                    p1, p2 = nodes[i][j], nodes[i + 1][j]
                    if p1 and p2:
                        d = abs(p2[1] - p1[1])
                        if np.isfinite(d) and d > 0:
                            node_spacings.append(d)

    if not centers and not node_centers:
        logger.warning("[Visualizer] No knots or grid nodes to draw")
        return img

    # Compute adaptive box half-size
    half = 12
    if not use_grid and sizes:
        s = float(np.median(sizes))
        if np.isfinite(s) and s > 0:
            half = int(max(6, min(22, s * 0.45)))
    elif use_grid and node_spacings:
        s = float(np.median(node_spacings))
        if np.isfinite(s) and s > 0:
            half = int(max(6, min(22, s * 0.18)))
    half = int(max(12, min(44, half * 2)))

    box_color = (0, 255, 255)
    box_thickness = 4
    pts = centers if not use_grid else node_centers
    for cx, cy in pts:
        x1i = int(max(0, min(cx - half, W - 1)))
        y1i = int(max(0, min(cy - half, H - 1)))
        x2i = int(max(0, min(cx + half, W - 1)))
        y2i = int(max(0, min(cy + half, H - 1)))
        if x2i <= x1i or y2i <= y1i:
            continue
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), box_color, box_thickness)

    src = "YOLO" if not use_grid else "grid"
    logger.info(f"[Visualizer] Drew {len(pts)} knot boxes (source: {src}, half={half}px)")
    return img


def _int_pt(p: Tuple[float, float]) -> Tuple[int, int]:
    return int(p[0]), int(p[1])


# ---------------------------------------------------------------------------
# Orchestrator-facing class (existing)
# ---------------------------------------------------------------------------

class Visualizer:
    """Result visualization generator for rebar detection (Step 6)."""

    def create_main_visualization(self, analyzer, image_path, output_path, **kw):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load: {image_path}")
        viz = analyzer.visualize_fitted_lines(img=img, include_text_overlay=kw.get("include_metrics", True))
        out = output_path / "line_fitting_results" / "line_fitting_visualization.png"
        cv2.imwrite(str(out), viz)
        return out

    def create_segmented_overlay(self, analyzer, seg_path, output_path):
        if not seg_path.exists():
            return None
        img = cv2.imread(str(seg_path))
        if img is None:
            return None
        viz = analyzer.visualize_fitted_lines(img=img)
        out = output_path / "line_fitting_results" / "lines_on_segmented_rebar.png"
        cv2.imwrite(str(out), viz)
        return out

    def create_comparison_view(self, orig_path, result_path, output_path, title="Before / After"):
        orig = cv2.imread(str(orig_path))
        result = cv2.imread(str(result_path))
        if orig is None or result is None:
            raise ValueError("Cannot load images for comparison")
        if orig.shape != result.shape:
            result = cv2.resize(result, (orig.shape[1], orig.shape[0]))
        comp = np.hstack([orig, result])
        bar = np.zeros((50, comp.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        comp = np.vstack([bar, comp])
        out = output_path / "comparison_before_after.png"
        cv2.imwrite(str(out), comp)
        return out
