"""
Stereo project data loading.

Provides:
- StereoProject: Loads a stereo project from xyz_map.npz + rect_left.jpg + K.txt
- read_ply_points(): Fast binary PLY reader (kept for backward compatibility)
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .config import ProjectFileNames


class StereoProject:
    """Loads a stereo project's 3D data, image, and camera intrinsics.

    Required files in project_path:
        - xyz_map.npz   — per-pixel 3D coordinates (H, W, 3), metres
        - rect_left.jpg — rectified left image (or raw_left.jpg)
        - K.txt          — camera intrinsics (3x3 matrix) + baseline on second line
    """

    def __init__(self, project_path: Union[str, Path]):
        self.project_path = Path(project_path)

        # Load xyz_map
        xyz_path = self.project_path / "xyz_map.npz"
        if not xyz_path.exists():
            raise FileNotFoundError(f"xyz_map.npz not found in {project_path}")
        self.xyz_map: np.ndarray = np.load(str(xyz_path))["xyz_map"]
        self.h, self.w = self.xyz_map.shape[:2]
        self.points: np.ndarray = self.xyz_map.reshape(-1, 3)

        # Load image
        self.image_path = self._find_image()
        self.image: np.ndarray = np.array(Image.open(self.image_path))

        # Load camera intrinsics
        self.K: Optional[np.ndarray] = None
        self.baseline: Optional[float] = None
        k_path = self.project_path / "K.txt"
        if k_path.exists():
            self.K, self.baseline = self._load_K(k_path)

    def _find_image(self) -> Path:
        candidates = [
            self.project_path / ProjectFileNames.RECT_LEFT,
            self.project_path / ProjectFileNames.RAW_LEFT,
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"No rectified image found in {self.project_path}. "
            f"Tried: {', '.join(c.name for c in candidates)}"
        )

    @staticmethod
    def _load_K(path: Path) -> Tuple[Optional[np.ndarray], Optional[float]]:
        try:
            with open(path) as f:
                lines = f.read().strip().split("\n")
            vals = list(map(float, lines[0].split()))
            K = np.array(vals).reshape(3, 3) if len(vals) == 9 else None
            baseline = float(lines[1]) if len(lines) > 1 else None
            return K, baseline
        except Exception:
            return None, None

    def get_3d_point(self, x: int, y: int) -> Optional[np.ndarray]:
        """Get the 3D coordinate at pixel (x, y), or None if invalid."""
        if 0 <= x < self.w and 0 <= y < self.h:
            pt = self.xyz_map[y, x]
            if np.isfinite(pt).all() and pt[2] > 0:
                return pt
        return None

    def compute_depth(self) -> np.ndarray:
        """Extract per-pixel depth (Z coordinate) from xyz_map."""
        return self.xyz_map[..., 2]


# ---------------------------------------------------------------------------
# Legacy PLY reader (kept for backward compatibility)
# ---------------------------------------------------------------------------

def read_ply_points(
    file_path: Union[str, Path],
    load_colors: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Lightweight PLY file reader optimised for rebar pipeline usage.

    Handles mixed data types (double x,y,z + uchar r,g,b) in binary and
    ASCII PLY files.  ~38x faster than a Python loop via ``np.fromfile``.

    Args:
        file_path: Path to PLY file.
        load_colors: If True, also return RGB colours.

    Returns:
        If load_colors is False: ``points`` array (N, 3) float64.
        If load_colors is True:  ``(points, colors)`` where colours is
        (N, 3) uint8 or None.
    """
    with open(file_path, "rb") as f:
        line = f.readline().decode("ascii").strip()
        if line != "ply":
            raise ValueError("Not a valid PLY file")

        vertex_count = 0
        properties: list[str] = []
        property_types: list[str] = []
        format_binary = False

        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format binary"):
                format_binary = True
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    property_types.append(parts[1])
                    properties.append(parts[-1])
            elif line == "end_header":
                break

        has_rgb = len(properties) >= 6 and any(
            p.lower() in ("red", "r") for p in properties
        )

        if format_binary:
            dtype_map = {
                "double": "<f8",
                "float": "<f4",
                "uchar": "u1",
                "int": "<i4",
            }
            dtype = np.dtype(
                [
                    (f"f{i}", dtype_map.get(t, "<f4"))
                    for i, t in enumerate(property_types)
                ]
            )
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            points = np.column_stack([data["f0"], data["f1"], data["f2"]])
            colors = (
                np.column_stack([data["f3"], data["f4"], data["f5"]])
                if has_rgb and load_colors and len(property_types) >= 6
                else None
            )
        else:
            rows: list[list[float]] = []
            colour_rows: list[list[int]] = []
            for _ in range(vertex_count):
                parts = f.readline().decode("ascii").strip().split()
                if len(parts) >= 3:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if load_colors and len(parts) >= 6:
                        colour_rows.append([int(parts[3]), int(parts[4]), int(parts[5])])
            points = np.array(rows) if rows else np.empty((0, 3))
            colors = np.array(colour_rows, dtype=np.uint8) if colour_rows else None

    # Filter invalid points
    if len(points) > 0:
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if load_colors and colors is not None and len(colors) == len(valid):
            colors = colors[valid]

    if load_colors:
        return points, colors
    return points
