"""Tests for configuration loading."""

from rebar_algorithm.config import RebarConfig, SAMConfig, ProjectFileNames


def test_rebar_config_defaults():
    cfg = RebarConfig(config_path="/nonexistent/path.yaml")
    assert cfg.get_server_url() == "http://localhost:2001"
    assert cfg.get_timeout() == 60
    assert cfg.is_plane_extraction_enabled() is True
    assert cfg.get_plane_distance_threshold() == 0.03


def test_sam_config_defaults():
    cfg = SAMConfig(config_path="/nonexistent/path.yaml")
    sc = cfg.get_server_config()
    assert "server_url" in sc
    assert sc["model"] == "vit_h"


def test_project_file_names():
    assert ProjectFileNames.RECT_LEFT == "rect_left.jpg"
    assert ProjectFileNames.RAW_LEFT == "raw_left.jpg"


def test_rebar_config_fallback():
    cfg = RebarConfig(config_path="/nonexistent/path.yaml")
    assert cfg.current_server == "primary"
    # Fallback disabled by default
    assert cfg.switch_to_fallback() is False
