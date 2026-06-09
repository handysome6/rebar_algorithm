"""Tests for individual pipeline stages (unit-level, no server needed)."""

import numpy as np
import pytest

from rebar_algorithm.data import StereoProject
from rebar_algorithm.stages.sam_mask import SamMaskProcessor
from rebar_algorithm.stages.line_fitting import LineFittingAnalyzer
from rebar_algorithm.stages.mask_grid import MaskGridDetector


class TestSamMaskProcessor:
    def test_binary_mask_2d(self):
        mask = np.random.rand(100, 200) > 0.5
        proc = SamMaskProcessor()
        binary = proc._convert_to_binary_mask(mask.astype(np.float32))
        assert binary.shape == (100, 200)
        assert binary.dtype == np.uint8

    def test_binary_mask_rgba(self):
        mask = np.zeros((100, 200, 4), dtype=np.uint8)
        mask[20:80, 50:150, 3] = 255
        proc = SamMaskProcessor()
        binary = proc._convert_to_binary_mask(mask)
        assert binary.sum() == 60 * 100

    def test_resize_mask(self):
        mask = np.ones((50, 100), dtype=np.uint8)
        proc = SamMaskProcessor()
        resized = proc._resize_mask_if_needed(mask, (100, 200))
        assert resized.shape == (100, 200)

    def test_load_base_image_requires_rect_left_even_if_project_named_image_exists(self, tmp_path):
        project = tmp_path / "project_001"
        project.mkdir()
        (project / "project_001_rect.jpg").write_bytes(b"not used")

        proc = SamMaskProcessor()
        with pytest.raises(FileNotFoundError):
            proc.load_base_image(project, "project_001")


def test_stereo_project_requires_rect_left_even_if_raw_exists(tmp_path):
    np.savez(str(tmp_path / "xyz_map.npz"), xyz_map=np.ones((2, 2, 3), dtype=float))
    (tmp_path / "raw_left.jpg").write_bytes(b"not used")

    with pytest.raises(FileNotFoundError):
        StereoProject(tmp_path)


class TestLineFittingAnalyzer:
    def test_classify_lines_hv(self):
        analyzer = LineFittingAnalyzer()
        h_lines = [(0, 50, 100, 50)]
        v_lines = [(50, 0, 50, 100)]
        result = analyzer._classify_lines_hv(h_lines, 0.0, v_lines, 90.0)
        assert len(result["horizontal"]) == 1
        assert len(result["vertical"]) == 1

    def test_cluster_dominant_angles(self):
        angles = [5.0] * 50 + [92.0] * 40 + [45.0] * 5
        dom = LineFittingAnalyzer._cluster_dominant_angles(angles)
        assert len(dom) >= 2
        # The two strongest modes should be near 5 and 92
        assert any(abs(a - 5.0) < 10 for a in dom)
        assert any(abs(a - 92.0) < 10 for a in dom)

    def test_fit_lines_with_clustering(self):
        analyzer = LineFittingAnalyzer()
        # 3 horizontal clusters
        pts = np.array([
            [10, 50], [50, 52], [100, 48],
            [10, 150], [50, 148], [100, 152],
            [10, 250], [50, 252], [100, 248],
        ], dtype=float)
        lines = analyzer._fit_lines_with_clustering(pts, "y", 300)
        assert len(lines) == 3


class TestMaskGridDetector:
    def test_analyze_synthetic_grid(self):
        cv2 = pytest.importorskip("cv2")

        mask = np.zeros((220, 320), dtype=np.uint8)
        for y in (50, 110, 170):
            cv2.line(mask, (20, y), (300, y), 1, 9)
        for x in (80, 220):
            cv2.line(mask, (x, 20), (x, 200), 1, 9)

        analysis = MaskGridDetector().analyze_mask(mask)

        assert analysis["line_fitting_summary"]["horizontal_lines"] == 3
        assert analysis["line_fitting_summary"]["vertical_lines"] == 2
        assert analysis["intersection_analysis"]["count"] == 6
