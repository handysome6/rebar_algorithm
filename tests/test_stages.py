"""Tests for individual pipeline stages (unit-level, no server needed)."""

import numpy as np
import pytest

from rebar_algorithm.stages.sam_mask import SamMaskProcessor
from rebar_algorithm.stages.line_fitting import LineFittingAnalyzer


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
