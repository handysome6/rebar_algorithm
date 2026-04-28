"""
Point cloud data loading and project management.

Provides:
- read_ply_points(): Fast binary PLY reader (replaces Open3D for performance)
- PCDProject: Loads and manages a stereo project's point cloud, images, and depth data
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .config import ProjectFileNames


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


class PCDProject:
    """Loads and manages a stereo project's point cloud, images, and depth data."""

    def __init__(self, project_path: Union[str, Path]):
        self.project_path = Path(project_path)
        self.pcd_path = self.project_path / "cloud.ply"

        # Scaled image (img0)
        img_jpg = self.project_path / "img0.jpg"
        img_png = self.project_path / "img0.png"
        if img_jpg.exists():
            self.image_path = img_jpg
        elif img_png.exists():
            self.image_path = img_png
        else:
            raise FileNotFoundError(f"Neither img0.jpg nor img0.png found in {project_path}")

        self.disp_path = self.project_path / "disp.npy"

        # Load point cloud
        self.points: np.ndarray = read_ply_points(str(self.pcd_path))

        self.image = np.array(Image.open(self.image_path))
        self.disp_data = np.load(self.disp_path)
        self.h, self.w = self.image.shape[:2]

        assert len(self.points) == self.h * self.w, (
            f"Point count ({len(self.points)}) != image pixels ({self.h * self.w})"
        )
        self.reshape_points = self.points.reshape(self.h, self.w, 3)

        # Original (rectified) image
        self.original_image_path = self._find_original_image()
        if self.original_image_path is not None:
            self.original_image = np.array(Image.open(self.original_image_path))
            self.scale = self.w / self.original_image.shape[1]
        else:
            self.original_image = None
            self.scale = None

    def _find_original_image(self) -> Optional[Path]:
        candidates = [
            self.project_path / ProjectFileNames.RECT_LEFT,
            self.project_path / ProjectFileNames.RAW_LEFT,
        ]
        for c in candidates:
            if c.exists():
                return c
        # Legacy: file starting with folder name
        for f in self.project_path.iterdir():
            if f.is_file() and f.name.startswith(self.project_path.name):
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    return f
        return None

    def mask_scaled_image(self, mask: np.ndarray, background_color=None) -> np.ndarray:
        assert mask.shape == (self.h, self.w)
        out = self.image.copy()
        m = mask.astype(bool)
        if background_color is None:
            out[~m] = 0
        else:
            out[~m] = background_color
        return out

    def mask_original_image(self, mask: np.ndarray, background_color=None) -> np.ndarray:
        scaled = cv2.resize(mask, (self.original_image.shape[1], self.original_image.shape[0]))
        m = scaled.astype(bool)
        out = self.original_image.copy()
        if background_color is None:
            out[~m] = 0
        else:
            out[~m] = background_color
        return out.astype(np.uint8)
