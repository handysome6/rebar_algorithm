"""
Step 4: Rebar Line Fitting

Fits centre-lines through detected rebar knots using clustering and
orientation-aware algorithms.  Supports pixel-space, metric plane-space,
and fallback x/y sequential clustering.

Replaces PySide6.QtCore.QPointF usage with plain tuples.
"""

import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from ..clients.yolo_parser import YoloResultParser


# ---------------------------------------------------------------------------
# Core analyser (no Qt dependencies)
# ---------------------------------------------------------------------------

class LineFittingAnalyzer:
    """Fit rebar centre-lines from YOLO-detected knot centres."""

    def __init__(self):
        self.json_path: Optional[Path] = None
        self.image_path: Optional[Path] = None
        self.image_width: Optional[int] = None
        self.image_height: Optional[int] = None
        self.output_folder: Optional[Path] = None

        self.centers: List[Tuple[float, float]] = []
        self.fitted_lines: Dict[str, List[Tuple[float, float, float, float]]] = {
            "horizontal": [], "vertical": [],
        }
        self.analysis_results: Dict = {}

        # Plane-space metric clustering
        self.plane_normal: Optional[np.ndarray] = None
        self.plane_d: Optional[float] = None
        self.depth_meter: Optional[np.ndarray] = None
        self.K_matrix: Optional[np.ndarray] = None
        self._plane_basis: Optional[Dict] = None

        self.ai_matcher = None

        self.colors = {
            "center": (255, 0, 255),
            "h_line": (0, 255, 0),
            "v_line": (255, 0, 0),
            "text": (255, 255, 255),
        }

    # -- public API -----------------------------------------------------------

    def process_all(
        self,
        json_path: str,
        image_path: str,
        output_folder: str = None,
        save_json: bool = True,
        save_image: bool = True,
        refined_mask: Optional[np.ndarray] = None,
        plane_metadata: Optional[Dict] = None,
        project_path: Optional[Path] = None,
        depth_meter_path: Optional[Path] = None,
    ) -> bool:
        self.output_folder = Path(output_folder) if output_folder else Path("line_fitting_results")
        self.output_folder.mkdir(parents=True, exist_ok=True)

        if not self.set_input_files(json_path, image_path):
            return False

        if plane_metadata is not None and (project_path is not None or depth_meter_path is not None):
            self.set_plane_data(plane_metadata, project_path=project_path, depth_path=depth_meter_path)

        if not self.extract_centers():
            return False

        self.fit_lines_from_centers((self.image_height, self.image_width), refined_mask=refined_mask)
        self.analyze_line_fitting_results()

        if save_json:
            self.save_analysis_results()
        if save_image:
            self.save_visualization()
        return True

    def set_input_files(self, json_path: str, image_path: str) -> bool:
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            return False
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            return False
        from PIL import Image as PILImage
        with PILImage.open(self.image_path) as img:
            self.image_width, self.image_height = img.size
        return True

    def set_ai_matcher(self, ai_matcher):
        self.ai_matcher = ai_matcher

    def extract_centers(self) -> bool:
        try:
            parser = YoloResultParser(str(self.json_path), self.image_width, self.image_height)
            self.centers = parser.get_centers_coordinates()
            logger.info(f"[LineFitting] Extracted {len(self.centers)} centres")
            return True
        except Exception as e:
            logger.error(f"[LineFitting] Centre extraction failed: {e}")
            return False

    # -- line fitting ---------------------------------------------------------

    def fit_lines_from_centers(
        self,
        img_shape: Optional[Tuple[int, int]] = None,
        refined_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
        if len(self.centers) < 2:
            return {"horizontal": [], "vertical": []}

        if img_shape is None:
            if self.image_height and self.image_width:
                img_shape = (self.image_height, self.image_width)
            else:
                pts = np.array(self.centers)
                img_shape = (int(np.max(pts[:, 1]) * 1.2), int(np.max(pts[:, 0]) * 1.2))

        points = np.array(self.centers)

        # Determine orientations from Hough on refined mask
        angle1, angle2 = None, None
        if refined_mask is not None:
            ori = self._analyze_rebar_orientation_from_mask(refined_mask)
            if ori.get("valid"):
                dom = ori.get("dominant_angles", [])
                if len(dom) >= 2:
                    angle1, angle2 = float(dom[0]), float(dom[1])
                elif len(dom) == 1:
                    angle1 = float(dom[0])
                    angle2 = (angle1 + 90.0) % 180.0

        if angle1 is not None and self.plane_normal is not None \
                and self.depth_meter is not None and self.K_matrix is not None:
            # Metric plane-space clustering
            metric_pts, plane_basis, valid = self._project_knots_to_plane(points)
            if valid.sum() >= 2:
                vm = metric_pts[valid]
                adaptive_max_dist, spacing_est = self._estimate_metric_max_dist(vm)
                adaptive_min_length = max(0.8 * spacing_est, 0.01)
                l1m = self._fit_lines_by_perpendicular_projection(vm, angle1, max_dist=adaptive_max_dist, min_length=adaptive_min_length)
                l2m = self._fit_lines_by_perpendicular_projection(vm, angle2, max_dist=adaptive_max_dist, min_length=adaptive_min_length)
                l1 = [x for x in (self._backproject_line_to_image(l, plane_basis) for l in l1m) if x]
                l2 = [x for x in (self._backproject_line_to_image(l, plane_basis) for l in l2m) if x]
                self.fitted_lines = self._classify_lines_hv(l1, angle1, l2, angle2)
            else:
                l1 = self._fit_lines_by_perpendicular_projection(points, angle1, img_shape)
                l2 = self._fit_lines_by_perpendicular_projection(points, angle2, img_shape)
                self.fitted_lines = self._classify_lines_hv(l1, angle1, l2, angle2)

        elif angle1 is not None:
            # Pixel-space clustering
            l1 = self._fit_lines_by_perpendicular_projection(points, angle1, img_shape)
            l2 = self._fit_lines_by_perpendicular_projection(points, angle2, img_shape)
            self.fitted_lines = self._classify_lines_hv(l1, angle1, l2, angle2)
        else:
            # Fallback: x/y sequential clustering
            h_lines = self._fit_lines_with_clustering(points, "y", img_shape[0])
            v_lines = self._fit_lines_with_clustering(points, "x", img_shape[1])
            self.fitted_lines = {"horizontal": h_lines, "vertical": v_lines}

        logger.info(f"[LineFitting] {len(self.fitted_lines['horizontal'])}H, {len(self.fitted_lines['vertical'])}V")
        return self.fitted_lines

    # -- perpendicular-projection clustering ----------------------------------

    def _fit_lines_by_perpendicular_projection(
        self, points, angle_deg, img_shape=None, min_knots=2,
        max_dist=None, min_length=None,
    ):
        if len(points) < min_knots:
            return []

        angle_norm = angle_deg % 180.0
        near_axis = 15.0
        if max_dist is None:
            max_dist = max(img_shape) * 0.14
        if min_length is None:
            min_length = min(img_shape) * 0.10

        if abs(angle_norm - 90.0) <= near_axis:
            use_axis = "x"
        elif angle_norm <= near_axis or angle_norm >= 180.0 - near_axis:
            use_axis = "y"
        else:
            use_axis = "perp"

        def _fit_cluster(pts):
            xs, ys = pts[:, 0], pts[:, 1]
            if use_axis == "y":
                if np.ptp(xs) < 1e-9:
                    return None
                c = np.polyfit(xs, ys, 1)
                x1, x2 = float(np.min(xs)), float(np.max(xs))
                return (x1, float(np.polyval(c, x1)), x2, float(np.polyval(c, x2)))
            else:
                if np.ptp(ys) < 1e-9:
                    return None
                c = np.polyfit(ys, xs, 1)
                y1, y2 = float(np.min(ys)), float(np.max(ys))
                return (float(np.polyval(c, y1)), y1, float(np.polyval(c, y2)), y2)

        def _check_length(ln):
            if ln is None:
                return None
            x1, y1, x2, y2 = ln
            return ln if math.hypot(x2 - x1, y2 - y1) >= min_length else None

        # Near-axis: directional NN graph + union-find
        if use_axis in ("x", "y"):
            N = len(points)
            parent = list(range(N))

            def _find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def _union(a, b):
                pa, pb = _find(a), _find(b)
                if pa != pb:
                    parent[pa] = pb

            ar = math.radians(angle_deg)
            d_vec = np.array([math.cos(ar), math.sin(ar)])
            n_vec = np.array([-math.sin(ar), math.cos(ar)])
            aspect = 5.0

            for i in range(N):
                for j in range(i + 1, N):
                    diff = points[j] - points[i]
                    if np.linalg.norm(diff) > max_dist:
                        continue
                    if abs(float(diff @ d_vec)) > aspect * abs(float(diff @ n_vec)):
                        _union(i, j)

            comp: Dict[int, list] = {}
            for i in range(N):
                comp.setdefault(_find(i), []).append(i)

            lines = []
            for idx_list in comp.values():
                if len(idx_list) < min_knots:
                    continue
                ln = _check_length(_fit_cluster(points[idx_list]))
                if ln:
                    lines.append(ln)
            return lines

        # General angle: 1D gap-adaptive clustering
        n_vec = np.array([-math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))])
        vals = points @ n_vec
        thresh = self._estimate_cluster_threshold(vals, img_shape)
        order = np.argsort(vals)
        sv = vals[order]

        groups: list[list] = []
        cur = [int(order[0])]
        for i in range(1, len(order)):
            if sv[i] - sv[i - 1] <= thresh:
                cur.append(int(order[i]))
            else:
                groups.append(cur)
                cur = [int(order[i])]
        groups.append(cur)

        lines = []
        for g in groups:
            if len(g) < min_knots:
                continue
            ln = _check_length(_fit_cluster(points[g]))
            if ln:
                lines.append(ln)
        return lines

    @staticmethod
    def _estimate_cluster_threshold(vals, img_shape):
        lo = max(img_shape) * 0.005
        hi = max(img_shape) * 0.06
        n = len(vals)
        if n < 4:
            return lo * 4
        gaps = np.diff(np.sort(vals))
        if len(gaps) < 2:
            return hi
        sg = np.sort(gaps)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(sg[:-1] > 0, sg[1:] / sg[:-1], np.inf)
        knee = int(np.argmax(ratios))
        return float(np.clip(math.sqrt(max(sg[knee], lo) * sg[knee + 1]), lo, hi))

    @staticmethod
    def _estimate_metric_max_dist(pts):
        N = len(pts)
        if N < 2:
            return 0.5, 0.15
        D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        est = float(np.median(D.min(axis=1)))
        return min(3.5 * est, 0.5), est

    @staticmethod
    def _classify_lines_hv(l1, a1, l2, a2):
        def _hdev(a):
            a = a % 180.0
            return min(a, abs(a - 180.0))
        if _hdev(a1) <= _hdev(a2):
            h, v = l1, l2
        else:
            h, v = l2, l1
        return {
            "horizontal": sorted(h, key=lambda ln: (ln[1] + ln[3]) / 2.0),
            "vertical": sorted(v, key=lambda ln: (ln[0] + ln[2]) / 2.0),
        }

    def _fit_lines_with_clustering(self, points, axis, img_size):
        eps = img_size * 0.03
        sort_axis = 1 if axis == "y" else 0
        sp = points[np.argsort(points[:, sort_axis])]
        clusters, cur = [], []
        for pt in sp:
            if not cur or abs(pt[sort_axis] - cur[-1][sort_axis]) <= eps:
                cur.append(pt)
            else:
                if len(cur) >= 2:
                    clusters.append(np.array(cur))
                cur = [pt]
        if len(cur) >= 2:
            clusters.append(np.array(cur))

        lines = []
        for cl in clusters:
            if axis == "y":
                x, y = cl[:, 0], cl[:, 1]
            else:
                x, y = cl[:, 1], cl[:, 0]
            c = np.polyfit(x, y, 1)
            xmin, xmax = float(x.min()), float(x.max())
            ymin, ymax = float(np.polyval(c, xmin)), float(np.polyval(c, xmax))
            if axis == "y":
                lines.append((xmin, ymin, xmax, ymax))
            else:
                lines.append((ymin, xmin, ymax, xmax))
        return lines

    # -- Hough orientation analysis -------------------------------------------

    def _analyze_rebar_orientation_from_mask(self, mask):
        if mask.dtype != np.uint8:
            m8 = (mask.astype(np.uint8)) * 255
        else:
            m8 = mask * 255
        if np.sum(mask) < 1000:
            return {"valid": False, "dominant_angles": [0.0, 90.0]}
        edges = cv2.Canny(m8, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
        angles = []
        if lines is not None:
            for ln in lines:
                a = np.degrees(np.arctan2(ln[0][3] - ln[0][1], ln[0][2] - ln[0][0]))
                if a < 0:
                    a += 180
                angles.append(a)
        if len(angles) < 10:
            return {"valid": True, "dominant_angles": [0.0, 90.0]}
        dom = self._cluster_dominant_angles(angles)
        return {"valid": True, "dominant_angles": dom}

    @staticmethod
    def _cluster_dominant_angles(angles, max_clusters=3):
        if len(angles) < 2:
            return list(angles)
        hist, edges = np.histogram(angles, bins=36, range=(0, 180))
        h = hist.astype(float)
        n_bins = len(h)
        min_h = float(max(h)) * 0.1
        peaks = []
        for i in range(n_bins):
            if h[i] >= min_h and h[i] > h[(i - 1) % n_bins] and h[i] > h[(i + 1) % n_bins]:
                peaks.append(i)
        if not peaks:
            mx = int(np.argmax(h))
            return [float((edges[mx] + edges[mx + 1]) / 2)]
        by_height = sorted(peaks, key=lambda p: h[p], reverse=True)
        kept = []
        for p in by_height:
            if all(min(abs(p - k), n_bins - abs(p - k)) >= 3 for k in kept):
                kept.append(p)
        kept.sort()
        return [float((edges[i] + edges[i + 1]) / 2) for i in kept[:max_clusters]]

    # -- plane-space projection -----------------------------------------------

    def set_plane_data(self, meta, project_path=None, depth_path=None, k_path=None):
        n = np.array(meta["plane_normal"], dtype=float)
        if np.linalg.norm(n) < 1e-6:
            return False
        self.plane_normal = n / np.linalg.norm(n)
        self.plane_d = float(meta["plane_distance"])

        pp = Path(project_path) if project_path else None

        # Load depth: prefer xyz_map.npz Z-channel, fall back to depth_meter.npy
        if pp and (pp / "xyz_map.npz").exists():
            xyz = np.load(str(pp / "xyz_map.npz"))["xyz_map"]
            self.depth_meter = xyz[..., 2]
        elif depth_path and Path(depth_path).exists():
            self.depth_meter = np.load(str(depth_path))
        else:
            return False

        # Load K: try project_path/K.txt, then explicit k_path, then depth_path sibling
        if k_path:
            self.K_matrix = self._load_K(Path(k_path))
        elif pp and (pp / "K.txt").exists():
            self.K_matrix = self._load_K(pp / "K.txt")
        elif depth_path:
            self.K_matrix = self._load_K(Path(depth_path).parent / "K.txt")
        else:
            self.K_matrix = None

        return self.K_matrix is not None

    @staticmethod
    def _load_K(path):
        try:
            with open(path) as f:
                vals = list(map(float, f.readline().split()))
            return np.array(vals).reshape(3, 3) if len(vals) == 9 else None
        except Exception:
            return None

    def _project_knots_to_plane(self, knot_pixels):
        K = self.K_matrix
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        n, d, depth = self.plane_normal, self.plane_d, self.depth_meter
        H, W = depth.shape[:2]
        N = len(knot_pixels)
        pts_3d = np.zeros((N, 3))
        valid = np.zeros(N, dtype=bool)

        for i, (u, v) in enumerate(knot_pixels):
            ui, vi = int(np.clip(round(u), 0, W - 1)), int(np.clip(round(v), 0, H - 1))
            z = float(depth[vi, ui])
            if z > 0 and np.isfinite(z):
                P = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
            else:
                ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
                denom = float(n @ ray)
                if abs(denom) < 1e-6:
                    continue
                lam = -d / denom
                if lam <= 0:
                    continue
                P = ray * lam
            pts_3d[i] = P - (float(n @ P) + d) * n
            valid[i] = True

        if valid.sum() < 2:
            return np.zeros((N, 2)), {}, valid

        origin = pts_3d[valid].mean(axis=0)
        e1 = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ n) * n
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.array([0.0, 1.0, 0.0]) - (np.array([0.0, 1.0, 0.0]) @ n) * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2)

        metric = np.zeros((N, 2))
        for i in range(N):
            if valid[i]:
                diff = pts_3d[i] - origin
                metric[i] = [float(diff @ e1), float(diff @ e2)]

        self._plane_basis = {"origin": origin, "e1": e1, "e2": e2}
        return metric, self._plane_basis, valid

    def _backproject_line_to_image(self, line_metric, basis):
        o, e1, e2 = basis["origin"], basis["e1"], basis["e2"]
        K = self.K_matrix
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        def _px(s, t):
            P = o + s * e1 + t * e2
            if P[2] <= 0:
                return None
            return float(fx * P[0] / P[2] + cx), float(fy * P[1] / P[2] + cy)

        s1, t1, s2, t2 = line_metric
        p1, p2 = _px(s1, t1), _px(s2, t2)
        if p1 is None or p2 is None:
            return None
        return (p1[0], p1[1], p2[0], p2[1])

    # -- analysis & output ----------------------------------------------------

    def analyze_line_fitting_results(self):
        analysis = {
            "total_centers": len(self.centers),
            "line_fitting_summary": {
                "horizontal_lines": len(self.fitted_lines["horizontal"]),
                "vertical_lines": len(self.fitted_lines["vertical"]),
                "total_lines": len(self.fitted_lines["horizontal"]) + len(self.fitted_lines["vertical"]),
            },
            "line_details": {"horizontal": [], "vertical": []},
            "grid_analysis": {},
        }
        for direction in ("horizontal", "vertical"):
            for i, ln in enumerate(self.fitted_lines[direction]):
                x1, y1, x2, y2 = ln
                analysis["line_details"][direction].append({
                    "line_id": i, "coordinates": ln,
                    "length": float(math.hypot(x2 - x1, y2 - y1)),
                    "angle": float(math.degrees(math.atan2(y2 - y1, x2 - x1))),
                })

        h, v = self.fitted_lines["horizontal"], self.fitted_lines["vertical"]
        if len(h) > 1 and len(v) > 1:
            hs = [abs((h[i + 1][1] + h[i + 1][3]) / 2 - (h[i][1] + h[i][3]) / 2) for i in range(len(h) - 1)]
            vs = [abs((v[i + 1][0] + v[i + 1][2]) / 2 - (v[i][0] + v[i][2]) / 2) for i in range(len(v) - 1)]
            analysis["grid_analysis"] = {
                "horizontal_spacing": {"values": hs, "average": float(np.mean(hs)), "std": float(np.std(hs))},
                "vertical_spacing": {"values": vs, "average": float(np.mean(vs)), "std": float(np.std(vs))},
            }
        self.analysis_results = analysis
        return analysis

    def visualize_fitted_lines(self, img=None, use_filtered_background=False, include_text_overlay=True):
        if img is None:
            img = cv2.imread(str(self.image_path))
        vis = img.copy()
        for pt in self.centers:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, self.colors["center"], -1)
        for ln in self.fitted_lines["horizontal"]:
            cv2.line(vis, (int(ln[0]), int(ln[1])), (int(ln[2]), int(ln[3])), self.colors["h_line"], 6)
        for ln in self.fitted_lines["vertical"]:
            cv2.line(vis, (int(ln[0]), int(ln[1])), (int(ln[2]), int(ln[3])), self.colors["v_line"], 6)
        if include_text_overlay:
            cv2.putText(vis, f"H: {len(self.fitted_lines['horizontal'])}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["h_line"], 2)
            cv2.putText(vis, f"V: {len(self.fitted_lines['vertical'])}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["v_line"], 2)
        return vis

    def save_analysis_results(self, filename="line_fitting_analysis.json"):
        with open(self.output_folder / filename, "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)

    def save_visualization(self, filename="line_fitting_visualization.png"):
        vis = self.visualize_fitted_lines()
        cv2.imwrite(str(self.output_folder / filename), vis)

    def calculate_3d_spatial_metrics(self, ai_matcher=None, use_fine_grained_spacing=True):
        """Stub — requires AIMatcher which is external to this demo project."""
        return {}


# ---------------------------------------------------------------------------
# Orchestrator-facing wrapper
# ---------------------------------------------------------------------------

class LineFitter:
    """Wrapper used by the pipeline orchestrator (Step 4)."""

    def __init__(self, ai_matcher=None):
        self.ai_matcher = ai_matcher
        self.analyzer: Optional[LineFittingAnalyzer] = None

    def fit_lines(
        self,
        pose_data_path: Path,
        image_path: Path,
        output_path: Path,
        refined_mask: Optional[np.ndarray] = None,
        plane_metadata: Optional[Dict] = None,
        project_path: Optional[Path] = None,
        depth_meter_path: Optional[Path] = None,
        save_json: bool = True,
        save_image: bool = True,
    ) -> Dict:
        self.analyzer = LineFittingAnalyzer()
        if self.ai_matcher:
            self.analyzer.set_ai_matcher(self.ai_matcher)

        fit_dir = output_path / "line_fitting_results"
        fit_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer.process_all(
            json_path=str(pose_data_path),
            image_path=str(image_path),
            output_folder=str(fit_dir),
            save_json=save_json,
            save_image=save_image,
            refined_mask=refined_mask,
            plane_metadata=plane_metadata,
            project_path=project_path,
            depth_meter_path=depth_meter_path,
        )

        return {
            "analyzer": self.analyzer,
            "final_image_path": fit_dir / "line_fitting_visualization.png",
            "analysis_json_path": fit_dir / "line_fitting_analysis.json",
            "line_count_horizontal": len(self.analyzer.fitted_lines["horizontal"]),
            "line_count_vertical": len(self.analyzer.fitted_lines["vertical"]),
            "used_hough": refined_mask is not None,
        }

    def create_segmented_overlay(self, segmented_image_path: Path, output_path: Path):
        if not self.analyzer or not segmented_image_path.exists():
            return None
        img = cv2.imread(str(segmented_image_path))
        if img is None:
            return None
        overlay = self.analyzer.visualize_fitted_lines(img=img)
        viz_dir = output_path / "visualization_results"
        viz_dir.mkdir(parents=True, exist_ok=True)
        out = viz_dir / "lines_on_segmented_rebar.png"
        cv2.imwrite(str(out), overlay)
        return out
