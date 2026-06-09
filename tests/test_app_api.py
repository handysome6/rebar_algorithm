"""Tests for reusable application helpers."""

from pathlib import Path

import pytest

from rebar_algorithm.app_api import find_project_image


def test_find_project_image_prefers_rect_left(tmp_path: Path):
    raw = tmp_path / "raw_left.jpg"
    rect = tmp_path / "rect_left.jpg"
    raw.write_bytes(b"raw")
    rect.write_bytes(b"rect")

    assert find_project_image(tmp_path) == rect


def test_find_project_image_falls_back_to_raw_left(tmp_path: Path):
    raw = tmp_path / "raw_left.jpg"
    raw.write_bytes(b"raw")

    assert find_project_image(tmp_path) == raw


def test_find_project_image_requires_known_image(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_project_image(tmp_path)
