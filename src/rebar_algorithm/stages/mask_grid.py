"""
Mask-based rebar grid detection.

Detects the visible rebar line families and their intersections directly from
the refined plane-filtered mask. This stage is intended to replace YOLO knot
detection when the refined mask is clean enough to carry the grid geometry.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


Line = Tuple[float, float, float, float]


class MaskGridDetector:
    """Detect rebar centerlines and intersections from a binary refined mask."""

    def __init__(
        self,
        profile_threshold_ratio: float = 0.20,
        min_intersection_support: float = 0.10,
        line_segment_percentiles: Tuple[float, float] = (1.0, 99.0),
    ):
        self.profile_threshold_ratio = profile_threshold_ratio
        self.min_intersection_support = min_intersection_support
        self.line_segment_percentiles = line_segment_percentiles

    # -- public API ---------------------------------------------------------

    def detect_grid(
        self,
        refined_mask: np.ndarray,
        output_path: Path,
        image_path: Optional[Path] = None,
        refined_image_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
    ) -> Dict:
        """Run mask-grid detection, save JSON/overlays, and return paths/results."""
        grid_dir = output_path / "mask_grid_results"
        grid_dir.mkdir(parents=True, exist_ok=True)

        analysis = self.analyze_mask(refined_mask, project_path=project_path)
        analysis_path = grid_dir / "mask_grid_analysis.json"

        final_image_path = None
        refined_overlay_path = None

        if image_path and Path(image_path).exists():
            image = cv2.imread(str(image_path))
            if image is not None:
                final_image_path = grid_dir / "mask_grid_on_rect_left.png"
                cv2.imwrite(str(final_image_path), self.draw_overlay(image, analysis))

        if refined_image_path and Path(refined_image_path).exists():
            refined_image = cv2.imread(str(refined_image_path))
            if refined_image is not None:
                refined_overlay_path = grid_dir / "mask_grid_on_segmented_rebar_refined.png"
                cv2.imwrite(str(refined_overlay_path), self.draw_overlay(refined_image, analysis))

        horizontal_mask = analysis.get("_debug_horizontal_mask")
        vertical_mask = analysis.get("_debug_vertical_mask")
        if isinstance(horizontal_mask, np.ndarray):
            cv2.imwrite(str(grid_dir / "directional_horizontal_mask.png"), horizontal_mask)
        if isinstance(vertical_mask, np.ndarray):
            cv2.imwrite(str(grid_dir / "directional_vertical_mask.png"), vertical_mask)

        # Debug masks are useful on disk but should not pollute JSON-facing data.
        analysis.pop("_debug_horizontal_mask", None)
        analysis.pop("_debug_vertical_mask", None)
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        return {
            "analysis": analysis,
            "analysis_json_path": analysis_path,
            "final_image_path": final_image_path,
            "refined_overlay_path": refined_overlay_path,
        }

    def analyze_mask(
        self,
        refined_mask: np.ndarray,
        project_path: Optional[Path] = None,
    ) -> Dict:
        """Detect lines and intersections without writing files."""
        mask = self._to_binary_mask(refined_mask)
        height, width = mask.shape
        params = self._scaled_params(mask.shape)

        clean = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["short_kernel"], params["short_kernel"])),
        )

        angle_a, angle_b = self._detect_grid_angles(clean)
        family_a = self._extract_family(clean, angle_a, params)
        family_b = self._extract_family(clean, angle_b, params)

        horizontal_lines, vertical_lines, horizontal_mask, vertical_mask = self._classify_families(
            family_a,
            angle_a,
            family_b,
            angle_b,
        )
        horizontal_lines = self._sort_lines(horizontal_lines, "horizontal")
        vertical_lines = self._sort_lines(vertical_lines, "vertical")

        intersections = self._compute_intersections(
            mask=mask,
            horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines,
            patch_radius=params["intersection_patch_radius"],
            segment_tolerance=params["segment_tolerance"],
        )
        self._attach_3d_points(intersections, project_path)

        analysis = self._build_analysis(
            mask_shape=(height, width),
            angles=(angle_a, angle_b),
            horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines,
            intersections=intersections,
            params=params,
        )
        analysis["_debug_horizontal_mask"] = horizontal_mask
        analysis["_debug_vertical_mask"] = vertical_mask

        logger.info(
            f"[MaskGridDetector] {len(horizontal_lines)}H, "
            f"{len(vertical_lines)}V, {len(intersections)} intersections"
        )
        return analysis

    # -- visualisation ------------------------------------------------------

    @staticmethod
    def draw_overlay(image: np.ndarray, analysis: Dict) -> np.ndarray:
        """Draw detected lines/intersections on a BGR image."""
        vis = image.copy()
        line_details = analysis.get("line_details", {})
        h_lines = line_details.get("horizontal", [])
        v_lines = line_details.get("vertical", [])
        intersections = analysis.get("intersection_analysis", {}).get("points", [])

        for item in h_lines:
            x1, y1, x2, y2 = [int(round(v)) for v in item["coordinates"]]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 80, 0), 10, cv2.LINE_AA)
        for item in v_lines:
            x1, y1, x2, y2 = [int(round(v)) for v in item["coordinates"]]
            cv2.line(vis, (x1, y1), (x2, y2), (80, 0, 0), 10, cv2.LINE_AA)
        for item in h_lines:
            x1, y1, x2, y2 = [int(round(v)) for v in item["coordinates"]]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
        for item in v_lines:
            x1, y1, x2, y2 = [int(round(v)) for v in item["coordinates"]]
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 4, cv2.LINE_AA)
        for item in intersections:
            x, y = int(round(item["x"])), int(round(item["y"]))
            cv2.circle(vis, (x, y), 13, (0, 0, 120), -1, cv2.LINE_AA)
            cv2.circle(vis, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)

        summary = analysis.get("line_fitting_summary", {})
        label = (
            f"mask grid: H={summary.get('horizontal_lines', 0)} "
            f"V={summary.get('vertical_lines', 0)} "
            f"I={analysis.get('intersection_analysis', {}).get('count', 0)}"
        )
        cv2.putText(vis, label, (36, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(vis, label, (36, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        return vis

    # -- line/intersection extraction --------------------------------------

    def _extract_family(self, mask: np.ndarray, angle_deg: float, params: Dict) -> Dict:
        rotated, _, inverse = self._rotate_full_canvas(mask, -angle_deg)
        directional = cv2.morphologyEx(
            rotated,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (params["long_kernel"], params["short_kernel"])),
        )
        directional = cv2.morphologyEx(
            directional,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (params["close_long_kernel"], params["close_short_kernel"])),
        )

        intervals = self._profile_intervals(directional, params)
        lines = []
        for interval in intervals:
            line = self._fit_line_from_rotated_band(
                directional,
                interval,
                inverse,
                original_shape=mask.shape,
                params=params,
            )
            if line:
                lines.append(line)

        original_mask = cv2.warpAffine(
            directional,
            inverse,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        return {
            "angle": float(angle_deg % 180.0),
            "lines": lines,
            "directional_mask": original_mask,
        }

    def _fit_line_from_rotated_band(
        self,
        directional: np.ndarray,
        interval: Dict,
        inverse: np.ndarray,
        original_shape: Tuple[int, int],
        params: Dict,
    ) -> Optional[Dict]:
        height, width = original_shape
        y0 = max(0, interval["start"] - params["band_padding"])
        y1 = min(directional.shape[0], interval["end"] + params["band_padding"] + 1)
        ys, xs = np.where(directional[y0:y1, :] > 0)
        if len(xs) < params["min_line_support_pixels"]:
            return None
        ys = ys + y0

        xo = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
        yo = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
        valid = (xo >= 0) & (xo < width) & (yo >= 0) & (yo < height)
        if int(valid.sum()) < params["min_line_support_pixels"]:
            return None

        pts = np.column_stack([xo[valid], yo[valid]]).astype(np.float32)
        vx, vy, px, py = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        vx, vy, px, py = map(float, (vx, vy, px, py))

        t = (pts[:, 0] - px) * vx + (pts[:, 1] - py) * vy
        t1, t2 = np.percentile(t, self.line_segment_percentiles)
        x1, y1 = px + vx * t1, py + vy * t1
        x2, y2 = px + vx * t2, py + vy * t2
        angle = float(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        length = float(math.hypot(x2 - x1, y2 - y1))
        return {
            "coordinates": [float(x1), float(y1), float(x2), float(y2)],
            "length": length,
            "angle": angle,
            "support_pixels": int(valid.sum()),
            "profile_center": float(interval["center"]),
            "profile_peak": float(interval["peak"]),
        }

    def _profile_intervals(self, directional: np.ndarray, params: Dict) -> List[Dict]:
        profile = directional.sum(axis=1).astype(np.float32) / 255.0
        profile_smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, params["profile_smooth"]), 0).ravel()
        threshold = max(
            float(profile_smooth.max()) * self.profile_threshold_ratio,
            float(params["profile_abs_min"]),
        )
        above = profile_smooth > threshold

        intervals = []
        i = 0
        while i < len(above):
            if not above[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            if j - i + 1 >= params["min_profile_width"]:
                xs = np.arange(i, j + 1, dtype=np.float32)
                weights = profile_smooth[i:j + 1]
                center = float((xs * weights).sum() / max(float(weights.sum()), 1e-6))
                intervals.append(
                    {
                        "start": int(i),
                        "end": int(j),
                        "center": center,
                        "peak": float(weights.max()),
                    }
                )
            i = j + 1
        return intervals

    def _compute_intersections(
        self,
        mask: np.ndarray,
        horizontal_lines: List[Dict],
        vertical_lines: List[Dict],
        patch_radius: int,
        segment_tolerance: float,
    ) -> List[Dict]:
        height, width = mask.shape
        intersections = []
        for hi, h_line in enumerate(horizontal_lines):
            for vi, v_line in enumerate(vertical_lines):
                point = self._line_intersection(h_line["coordinates"], v_line["coordinates"])
                if point is None:
                    continue
                x, y = point
                if not (0.0 <= x < width and 0.0 <= y < height):
                    continue
                if not self._within_segment(point, h_line["coordinates"], segment_tolerance):
                    continue
                if not self._within_segment(point, v_line["coordinates"], segment_tolerance):
                    continue

                yi, xi = int(round(y)), int(round(x))
                patch = mask[
                    max(0, yi - patch_radius):min(height, yi + patch_radius + 1),
                    max(0, xi - patch_radius):min(width, xi + patch_radius + 1),
                ]
                support = float(patch.mean() / 255.0) if patch.size else 0.0
                if support < self.min_intersection_support:
                    continue

                intersections.append(
                    {
                        "intersection_id": len(intersections),
                        "horizontal_line_id": hi,
                        "vertical_line_id": vi,
                        "x": float(x),
                        "y": float(y),
                        "local_mask_support": support,
                    }
                )
        return intersections

    # -- analysis helpers ---------------------------------------------------

    @staticmethod
    def _build_analysis(
        mask_shape: Tuple[int, int],
        angles: Tuple[float, float],
        horizontal_lines: List[Dict],
        vertical_lines: List[Dict],
        intersections: List[Dict],
        params: Dict,
    ) -> Dict:
        def _line_detail(line_id: int, line: Dict) -> Dict:
            x1, y1, x2, y2 = line["coordinates"]
            return {
                "line_id": line_id,
                "coordinates": [float(x1), float(y1), float(x2), float(y2)],
                "length": float(line["length"]),
                "angle": float(line["angle"]),
                "support_pixels": int(line["support_pixels"]),
                "profile_center": float(line["profile_center"]),
                "profile_peak": float(line["profile_peak"]),
            }

        h_details = [_line_detail(i, line) for i, line in enumerate(horizontal_lines)]
        v_details = [_line_detail(i, line) for i, line in enumerate(vertical_lines)]
        h_spacings = [
            abs((h_details[i + 1]["coordinates"][1] + h_details[i + 1]["coordinates"][3]) * 0.5 -
                (h_details[i]["coordinates"][1] + h_details[i]["coordinates"][3]) * 0.5)
            for i in range(len(h_details) - 1)
        ]
        v_spacings = [
            abs((v_details[i + 1]["coordinates"][0] + v_details[i + 1]["coordinates"][2]) * 0.5 -
                (v_details[i]["coordinates"][0] + v_details[i]["coordinates"][2]) * 0.5)
            for i in range(len(v_details) - 1)
        ]

        def _spacing_summary(values: List[float]) -> Dict:
            if not values:
                return {"values": [], "average": None, "std": None}
            return {
                "values": [float(v) for v in values],
                "average": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        return {
            "method": "refined_mask_grid_detection",
            "source": "refined_mask",
            "mask_shape": [int(mask_shape[0]), int(mask_shape[1])],
            "detected_angles_degrees": [float(angles[0] % 180.0), float(angles[1] % 180.0)],
            "total_centers": len(intersections),
            "line_fitting_summary": {
                "horizontal_lines": len(h_details),
                "vertical_lines": len(v_details),
                "total_lines": len(h_details) + len(v_details),
            },
            "line_details": {"horizontal": h_details, "vertical": v_details},
            "intersection_analysis": {
                "count": len(intersections),
                "points": intersections,
            },
            "grid_analysis": {
                "horizontal_spacing": _spacing_summary(h_spacings),
                "vertical_spacing": _spacing_summary(v_spacings),
            },
            "parameters": {
                k: (float(v) if isinstance(v, float) else int(v))
                for k, v in params.items()
            },
        }

    @staticmethod
    def _sort_lines(lines: List[Dict], direction: str) -> List[Dict]:
        if direction == "horizontal":
            return sorted(lines, key=lambda line: (line["coordinates"][1] + line["coordinates"][3]) * 0.5)
        return sorted(lines, key=lambda line: (line["coordinates"][0] + line["coordinates"][2]) * 0.5)

    @staticmethod
    def _classify_families(
        family_a: Dict,
        angle_a: float,
        family_b: Dict,
        angle_b: float,
    ) -> Tuple[List[Dict], List[Dict], np.ndarray, np.ndarray]:
        def h_deviation(angle: float) -> float:
            angle = angle % 180.0
            return min(angle, abs(angle - 180.0))

        if h_deviation(angle_a) <= h_deviation(angle_b):
            return (
                family_a["lines"],
                family_b["lines"],
                family_a["directional_mask"],
                family_b["directional_mask"],
            )
        return (
            family_b["lines"],
            family_a["lines"],
            family_b["directional_mask"],
            family_a["directional_mask"],
        )

    @staticmethod
    def _attach_3d_points(intersections: List[Dict], project_path: Optional[Path]) -> None:
        if not project_path:
            return
        xyz_path = Path(project_path) / "xyz_map.npz"
        if not xyz_path.exists():
            return
        try:
            xyz_map = np.load(str(xyz_path))["xyz_map"]
        except Exception:
            return
        height, width = xyz_map.shape[:2]
        for item in intersections:
            x = int(np.clip(round(item["x"]), 0, width - 1))
            y = int(np.clip(round(item["y"]), 0, height - 1))
            point = xyz_map[y, x].astype(float)
            if np.isfinite(point).all() and point[2] > 0:
                item["point_3d_m"] = [float(point[0]), float(point[1]), float(point[2])]
            else:
                item["point_3d_m"] = None

    # -- geometry -----------------------------------------------------------

    @staticmethod
    def _line_intersection(line_a: Line, line_b: Line) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2 = line_a
        x3, y3, x4, y4 = line_b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return float(px), float(py)

    @staticmethod
    def _within_segment(point: Tuple[float, float], line: Line, tolerance: float) -> bool:
        x1, y1, x2, y2 = line
        vx, vy = x2 - x1, y2 - y1
        length_sq = vx * vx + vy * vy
        if length_sq < 1.0:
            return False
        t = ((point[0] - x1) * vx + (point[1] - y1) * vy) / length_sq
        distance = abs((point[0] - x1) * vy - (point[1] - y1) * vx) / math.sqrt(length_sq)
        return -0.05 <= t <= 1.05 and distance <= tolerance

    @staticmethod
    def _rotate_full_canvas(image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        transform = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos_v, sin_v = abs(transform[0, 0]), abs(transform[0, 1])
        new_width = int(height * sin_v + width * cos_v)
        new_height = int(height * cos_v + width * sin_v)
        transform[0, 2] += new_width / 2.0 - center[0]
        transform[1, 2] += new_height / 2.0 - center[1]
        rotated = cv2.warpAffine(
            image,
            transform,
            (new_width, new_height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        inverse = cv2.invertAffineTransform(transform)
        return rotated, transform, inverse

    # -- angle and parameter helpers ---------------------------------------

    @staticmethod
    def _detect_grid_angles(mask: np.ndarray) -> Tuple[float, float]:
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=20)
        if lines is None or len(lines) == 0:
            return 0.0, 90.0

        angles = []
        weights = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 20.0:
                continue
            angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0)
            weights.append(length)
        if not angles:
            return 0.0, 90.0

        hist, edges = np.histogram(angles, bins=180, range=(0, 180), weights=weights)
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        padded = np.r_[hist[-2:], hist, hist[:2]]
        smooth = np.convolve(padded, kernel, mode="same")[2:-2]

        min_peak = float(smooth.max()) * 0.10
        peaks = []
        for i, value in enumerate(smooth):
            if value > min_peak and value > smooth[(i - 1) % len(smooth)] and value >= smooth[(i + 1) % len(smooth)]:
                peaks.append(i)
        if not peaks:
            peak = int(np.argmax(smooth))
            angle = float((edges[peak] + edges[peak + 1]) * 0.5)
            return angle, (angle + 90.0) % 180.0

        peaks = sorted(peaks, key=lambda idx: smooth[idx], reverse=True)
        chosen = []
        for peak in peaks:
            angle = float((edges[peak] + edges[peak + 1]) * 0.5)
            if all(MaskGridDetector._angle_distance(angle, other) >= 25.0 for other in chosen):
                chosen.append(angle)
            if len(chosen) == 2:
                break
        if len(chosen) == 1:
            chosen.append((chosen[0] + 90.0) % 180.0)
        return chosen[0], chosen[1]

    @staticmethod
    def _angle_distance(a: float, b: float) -> float:
        diff = abs((a % 180.0) - (b % 180.0))
        return min(diff, 180.0 - diff)

    @staticmethod
    def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask[..., 0]
        return ((mask > 0).astype(np.uint8)) * 255

    @staticmethod
    def _scaled_params(shape: Tuple[int, int]) -> Dict:
        height, width = shape
        scale = min(height, width)

        def odd(value: float) -> int:
            result = max(3, int(round(value)))
            return result if result % 2 else result + 1

        return {
            "long_kernel": max(31, int(round(scale * 0.060))),
            "short_kernel": odd(max(5, scale * 0.003)),
            "close_long_kernel": max(31, int(round(scale * 0.026))),
            "close_short_kernel": odd(max(7, scale * 0.005)),
            "profile_smooth": odd(max(31, scale * 0.033)),
            "profile_abs_min": max(30, int(round(scale * 0.033))),
            "min_profile_width": max(6, int(round(scale * 0.003))),
            "band_padding": max(8, int(round(scale * 0.010))),
            "min_line_support_pixels": max(80, int(round(scale * 0.030))),
            "intersection_patch_radius": max(12, int(round(scale * 0.012))),
            "segment_tolerance": max(20, float(scale * 0.026)),
        }
