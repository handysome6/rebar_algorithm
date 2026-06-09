"""Tests for CLI argument resolution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rebar_algorithm.cli import _build_parser, _resolve_detector, _resolve_input


class _Cfg:
    def __init__(self, use_mask_grid=False):
        self._use_mask_grid = use_mask_grid

    def use_mask_grid_detector(self):
        return self._use_mask_grid


def test_parser_accepts_refined_mask_detector_mode():
    parser = _build_parser()
    args = parser.parse_args([
        "--project", "/tmp/project",
        "--refined-mask", "/tmp/refined.npy",
        "--detector", "mask-grid",
    ])

    assert args.refined_mask == Path("/tmp/refined.npy")
    assert args.detector == "mask-grid"


def test_detector_defaults_to_config_yolo():
    args = SimpleNamespace(detector=None)
    assert _resolve_detector(args, _Cfg(use_mask_grid=False)) == "yolo"


def test_detector_defaults_to_config_mask_grid():
    args = SimpleNamespace(detector=None)
    assert _resolve_detector(args, _Cfg(use_mask_grid=True)) == "mask-grid"


def test_detector_explicit_yolo_wins_over_config():
    args = SimpleNamespace(detector="yolo")
    assert _resolve_detector(args, _Cfg(use_mask_grid=True)) == "yolo"


def test_deprecated_mask_flags_are_removed():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--project", "/tmp/project", "--mask", "/tmp/sam.npy"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--project", "/tmp/project", "--sam-mask", "/tmp/sam.npy", "--mask-grid"])


def test_reuse_refined_resolves_output_mask():
    args = SimpleNamespace(points=None, refined_mask=None, reuse_refined=True, sam_mask=None)
    stage, path = _resolve_input(args, Path("/tmp/out"))

    assert stage == "refined"
    assert path == Path("/tmp/out/plane_extraction_results/refined_mask.npy")
