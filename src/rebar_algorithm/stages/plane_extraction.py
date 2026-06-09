"""
Step 2: 3D Plane Extraction for rebar surface layer isolation.

Uses RANSAC (or SVD through seed points) to fit a plane to the SAM-masked
point cloud, then filters pixels within ±distance_threshold of the plane.

Standalone function:  extract_surface_layer_from_sam_mask()
Wrapper class:        PlaneExtractor (used by the orchestrator)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import json
import numpy as np
from loguru import logger

from ..data import StereoProject


# ---------------------------------------------------------------------------
# Core implementation class
# ---------------------------------------------------------------------------

class PlaneExtractorImpl:
    """RANSAC / SVD plane fitting on a SAM-masked point cloud."""

    def __init__(
        self,
        distance_threshold: float = 0.03,
        ransac_iterations: int = 500,
        ransac_distance_threshold: float = 0.01,
    ):
        self.distance_threshold = distance_threshold
        self.ransac_iterations = ransac_iterations
        self.ransac_distance_threshold = ransac_distance_threshold
        self.plane_model: Optional[Dict] = None
        self.plane_normal: Optional[np.ndarray] = None

    def extract_surface_layer(
        self,
        sam_mask: np.ndarray,
        project_path: Path,
        visualize: bool = True,
        sam_prompt_points=None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full extraction: load PCD → fit plane → filter → return refined mask + metadata.
        """
        logger.info(f"[PlaneExtraction] project={project_path}, threshold={self.distance_threshold * 100:.1f}cm")

        original_mask_shape = sam_mask.shape
        mask_was_resized = False

        # Step 1: load 3D data
        project = StereoProject(project_path)
        points_3d = project.points
        h, w = project.h, project.w

        # Resize mask if needed
        if sam_mask.shape != (h, w):
            logger.warning(f"[PlaneExtraction] Mask {sam_mask.shape} != PCD ({h},{w}), resizing")
            sam_mask = cv2.resize(sam_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_was_resized = True

        # Step 2: SAM-masked points
        mask_flat = sam_mask.flatten().astype(bool)
        if mask_flat.sum() == 0:
            raise ValueError("SAM mask is empty")
        masked_points = points_3d[mask_flat]
        logger.info(f"[PlaneExtraction] {mask_flat.sum():,} masked pixels")

        # Step 3a: resolve seed points to 3D
        seed_points_3d = None
        if sam_prompt_points:
            seeds = []
            for pt in sam_prompt_points:
                x, y = int(pt[0]), int(pt[1])
                idx = y * w + x
                if 0 <= idx < len(points_3d):
                    p = points_3d[idx]
                    if np.isfinite(p).all() and p[2] > 0:
                        seeds.append(p)
            if seeds:
                seed_points_3d = np.array(seeds)
                logger.info(f"[PlaneExtraction] {len(seeds)}/{len(sam_prompt_points)} valid seed points")

        # Step 3b: fit plane
        if seed_points_3d is not None and len(seed_points_3d) >= 3:
            logger.info(f"[PlaneExtraction] SVD from {len(seed_points_3d)} seeds (primary)")
            plane_model, plane_normal = self._fit_plane_svd(seed_points_3d)
            inlier_count = int(np.sum(np.abs(masked_points @ plane_normal + plane_model["d"]) <= self.ransac_distance_threshold))
        else:
            logger.info("[PlaneExtraction] RANSAC fallback")
            plane_model, inlier_mask, plane_normal = self._fit_plane_ransac(
                masked_points, self.ransac_iterations, self.ransac_distance_threshold,
                seed_points=seed_points_3d,
            )
            if plane_model is None:
                raise RuntimeError("RANSAC failed to detect a surface plane")
            inlier_count = int(np.sum(inlier_mask))

        self.plane_model = plane_model
        self.plane_normal = plane_normal

        # Step 4: proximity envelope
        proximity_envelope = None
        if seed_points_3d is not None:
            proximity_envelope = self._compute_proximity_envelope(seed_points_3d, plane_normal, plane_model["d"])

        # Step 5: extract surface layer
        refined_mask = self._extract_points_near_plane(
            points_3d, plane_model, plane_normal, (h, w), self.distance_threshold, proximity_envelope,
        )

        # Step 6: metadata
        metadata = self._compute_metadata(sam_mask, refined_mask, plane_model, plane_normal, masked_points, inlier_count)
        metadata["plane_fitting_method"] = (
            "svd_from_seeds" if (seed_points_3d is not None and len(seed_points_3d) >= 3) else "ransac"
        )
        if proximity_envelope:
            metadata["proximity_envelope_near_mm"] = float(proximity_envelope[0] * 1000)
            metadata["proximity_envelope_far_mm"] = float(proximity_envelope[1] * 1000)
            metadata["prompt_points_used"] = len(seed_points_3d)
        else:
            metadata["proximity_envelope_near_mm"] = None
            metadata["proximity_envelope_far_mm"] = None
            metadata["prompt_points_used"] = 0
        metadata["original_mask_shape"] = list(original_mask_shape)
        metadata["point_cloud_shape"] = [h, w]
        metadata["mask_was_resized"] = mask_was_resized

        logger.info(f"[PlaneExtraction] SAM={metadata['sam_coverage']:.1%} -> refined={metadata['refined_coverage']:.1%}")
        return refined_mask, metadata

    # -- RANSAC ---------------------------------------------------------------

    def _fit_plane_ransac(self, points, max_iter, dist_thresh, seed_points=None):
        valid_mask = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        vp = points[valid_mask]
        if len(vp) < 3:
            return None, None, None

        best_plane, best_inliers, best_normal = None, None, None
        max_inliers = 0
        n_seeded = max_iter // 5 if seed_points is not None and len(seed_points) > 0 else 0

        for i in range(max_iter):
            if i < n_seeded:
                seed = seed_points[i % len(seed_points)]
                rand2 = vp[np.random.choice(len(vp), 2, replace=False)]
                sample = np.vstack([seed, rand2])
            else:
                sample = vp[np.random.choice(len(vp), 3, replace=False)]

            v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
            n = np.cross(v1, v2)
            nl = np.linalg.norm(n)
            if nl < 1e-6:
                continue
            n /= nl
            if n[2] < 0:
                n = -n
            d = -np.dot(n, sample[0])
            dists = np.abs(vp @ n + d)
            inliers = dists < dist_thresh
            cnt = int(inliers.sum())
            if cnt > max_inliers:
                max_inliers = cnt
                best_normal = n.copy()
                best_inliers = inliers.copy()
                best_plane = {"a": float(n[0]), "b": float(n[1]), "c": float(n[2]), "d": float(d)}

        if best_plane is None:
            return None, None, None

        # SVD refinement on inliers
        inlier_pts = vp[best_inliers]
        if len(inlier_pts) >= 3:
            centroid = inlier_pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
            rn = Vt[-1]
            if rn[2] < 0:
                rn = -rn
            rd = -float(rn @ centroid)
            best_plane = {"a": float(rn[0]), "b": float(rn[1]), "c": float(rn[2]), "d": rd}
            best_normal = rn

        full_mask = np.zeros(len(points), dtype=bool)
        full_mask[valid_mask] = best_inliers
        return best_plane, full_mask, best_normal

    # -- SVD from seed points -------------------------------------------------

    @staticmethod
    def _fit_plane_svd(points: np.ndarray) -> Tuple[Dict, np.ndarray]:
        centroid = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
        n = Vt[-1]
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        d = -float(n @ centroid)
        return {"a": float(n[0]), "b": float(n[1]), "c": float(n[2]), "d": d}, n

    # -- proximity envelope ---------------------------------------------------

    @staticmethod
    def _compute_proximity_envelope(seeds_3d, normal, d):
        signed = normal @ seeds_3d.T + d
        return (float(signed.min()) - 0.005, float(signed.max()) + 0.015)

    # -- surface extraction ---------------------------------------------------

    @staticmethod
    def _extract_points_near_plane(points, model, normal, shape, threshold, envelope=None):
        h, w = shape
        signed = points @ normal + model["d"]
        if envelope is not None:
            within = (signed >= envelope[0]) & (signed <= envelope[1])
        else:
            within = np.abs(signed) <= threshold
        return within.reshape(h, w).astype(np.uint8)

    # -- metadata -------------------------------------------------------------

    @staticmethod
    def _compute_metadata(sam_mask, refined, model, normal, masked_pts, inlier_count):
        total = sam_mask.size
        sam_px = int(np.sum(sam_mask > 0))
        ref_px = int(np.sum(refined > 0))
        return {
            "sam_coverage": sam_px / total,
            "refined_coverage": ref_px / total,
            "coverage_reduction": (sam_px - ref_px) / total if sam_px > 0 else 0,
            "reduction_ratio": ref_px / sam_px if sam_px > 0 else 0,
            "plane_model": model,
            "plane_normal": normal.tolist(),
            "plane_distance": float(model["d"]),
            "ransac_inlier_count": inlier_count,
            "ransac_inlier_ratio": inlier_count / len(masked_pts) if len(masked_pts) > 0 else 0,
            "distance_threshold": 0,  # will be overwritten by caller
            "sam_pixels": sam_px,
            "refined_pixels": ref_px,
        }


# ---------------------------------------------------------------------------
# Convenience function (matches original API)
# ---------------------------------------------------------------------------

def extract_surface_layer_from_sam_mask(
    sam_mask: np.ndarray,
    project_path: Path,
    distance_threshold: float = 0.03,
    visualize: bool = True,
    output_path=None,
    sam_prompt_points=None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """Top-level function wrapping PlaneExtractorImpl."""
    ext = PlaneExtractorImpl(distance_threshold=distance_threshold)
    return ext.extract_surface_layer(
        sam_mask=sam_mask,
        project_path=project_path,
        visualize=visualize,
        sam_prompt_points=sam_prompt_points,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Orchestrator-facing wrapper
# ---------------------------------------------------------------------------

class PlaneExtractor:
    """Wrapper used by the pipeline orchestrator (Step 2)."""

    def __init__(self, distance_threshold: float = 0.03):
        self.distance_threshold = distance_threshold

    def extract_surface_layer(
        self,
        sam_mask: np.ndarray,
        project_path: Path,
        output_path: Path,
        visualize: bool = True,
        sam_prompt_points=None,
        **kwargs,
    ) -> Dict:
        plane_out = output_path / "plane_extraction_results"
        plane_out.mkdir(parents=True, exist_ok=True)

        refined_mask, metadata = extract_surface_layer_from_sam_mask(
            sam_mask=sam_mask,
            project_path=project_path,
            distance_threshold=self.distance_threshold,
            visualize=visualize,
            sam_prompt_points=sam_prompt_points,
            **kwargs,
        )

        # Save metadata
        serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, (np.float32, np.float64)) else v)
            for k, v in metadata.items()
        }
        with open(plane_out / "plane_metadata.json", "w") as f:
            json.dump(serializable, f, indent=2)

        # Resize to original resolution
        orig_shape = sam_mask.shape[:2]
        if refined_mask.shape != orig_shape:
            refined_orig = cv2.resize(refined_mask.astype(np.uint8), (orig_shape[1], orig_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        else:
            refined_orig = refined_mask

        np.save(str(plane_out / "refined_mask.npy"), refined_orig)

        return {
            "refined_mask_pcd_res": refined_mask,
            "refined_mask_original_res": refined_orig,
            "refined_mask_path": plane_out / "refined_mask.npy",
            "plane_metadata": metadata,
            "coverage_sam": metadata["sam_coverage"],
            "coverage_refined": metadata["refined_coverage"],
            "coverage_reduction": metadata["coverage_reduction"],
        }

    @staticmethod
    def update_segmented_image(output_path: Path, refined_mask: np.ndarray, base_image: np.ndarray) -> Path:
        seg = base_image.copy()
        seg[refined_mask == 0] = [240, 240, 240]
        out = output_path / "plane_extraction_results" / "segmented_rebar_refined.png"
        cv2.imwrite(str(out), seg)
        return out
