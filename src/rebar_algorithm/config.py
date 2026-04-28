"""
Unified configuration for the rebar detection pipeline.

Merges RebarConfig (YOLO server, plane extraction) and SAMConfig (SAM server)
into a single config manager. Also includes ProjectFileNames constants.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Standardised file names expected in each project folder
# ---------------------------------------------------------------------------
class ProjectFileNames:
    """Standardized file names for project data."""
    RAW_LEFT = "raw_left.jpg"
    RAW_RIGHT = "raw_right.jpg"
    RECT_LEFT = "rect_left.jpg"
    RECT_RIGHT = "rect_right.jpg"


# ---------------------------------------------------------------------------
# Rebar pipeline configuration
# ---------------------------------------------------------------------------
class RebarConfig:
    """YAML-based configuration for the rebar detection pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            root_dir = Path(__file__).parent.parent.parent
            config_path = root_dir / "configuration" / "rebar_conf.yaml"
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config()
        self.current_server = "primary"
        logger.info(f"[RebarConfig] Loaded from {self.config_path}")

    # -- loading --------------------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        try:
            if not self.config_path.exists():
                logger.warning(f"[RebarConfig] Config not found: {self.config_path}")
                return self._defaults()
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[RebarConfig] Load failed: {e}")
            return self._defaults()

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "primary_server": {
                "server_url": "http://localhost:2001",
                "timeout": 60,
                "enabled": True,
            },
            "fallback_server": {
                "server_url": "http://localhost:8000",
                "timeout": 60,
                "enabled": False,
            },
            "detection": {
                "use_existing_annotations": True,
                "output_dir_name": "rebar_output_sam",
            },
            "visualization": {"use_rectified_for_visualization": True},
            "plane_extraction": {
                "enabled": True,
                "distance_threshold": 0.03,
                "ransac_iterations": 500,
                "ransac_distance_threshold": 0.01,
            },
        }

    # -- server ---------------------------------------------------------------
    def get_server_url(self) -> str:
        key = f"{self.current_server}_server"
        cfg = self.config.get(key, self.config.get("primary_server", {}))
        return cfg.get("server_url", "http://localhost:2001")

    def get_timeout(self) -> int:
        key = f"{self.current_server}_server"
        cfg = self.config.get(key, self.config.get("primary_server", {}))
        return cfg.get("timeout", 60)

    def switch_to_fallback(self) -> bool:
        fb = self.config.get("fallback_server", {})
        if self.current_server == "primary" and fb.get("enabled", False):
            self.current_server = "fallback"
            logger.info("[RebarConfig] Switched to fallback server")
            return True
        return False

    def reset_to_primary(self) -> None:
        self.current_server = "primary"

    # -- detection ------------------------------------------------------------
    def use_existing_annotations(self) -> bool:
        return self.config.get("detection", {}).get("use_existing_annotations", True)

    # -- visualization --------------------------------------------------------
    def use_rectified_for_visualization(self) -> bool:
        return self.config.get("visualization", {}).get("use_rectified_for_visualization", True)

    # -- plane extraction -----------------------------------------------------
    def is_plane_extraction_enabled(self) -> bool:
        return self.config.get("plane_extraction", {}).get("enabled", True)

    def get_plane_distance_threshold(self) -> float:
        return self.config.get("plane_extraction", {}).get("distance_threshold", 0.03)

    def get_plane_ransac_iterations(self) -> int:
        return self.config.get("plane_extraction", {}).get("ransac_iterations", 500)

    def get_plane_ransac_distance_threshold(self) -> float:
        return self.config.get("plane_extraction", {}).get("ransac_distance_threshold", 0.01)

    def __repr__(self) -> str:
        return f"RebarConfig(server={self.current_server}, url={self.get_server_url()})"


# ---------------------------------------------------------------------------
# SAM configuration
# ---------------------------------------------------------------------------
class SAMConfig:
    """YAML-based configuration for the SAM segmentation server."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            root_dir = Path(__file__).parent.parent.parent
            config_path = root_dir / "configuration" / "sam_conf.yaml"
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config()
        self.current_server = "primary"
        logger.info(f"[SAMConfig] Loaded from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        try:
            if not self.config_path.exists():
                logger.warning(f"[SAMConfig] Config not found: {self.config_path}")
                return self._defaults()
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[SAMConfig] Load failed: {e}")
            return self._defaults()

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "primary_server": {
                "server_url": "https://segmentation.ensightful.xyz",
                "model": "vit_h",
                "use_tensorrt": True,
                "alpha": 0.5,
                "timeout": 30,
                "enabled": True,
            },
            "fallback_server": {
                "server_url": "https://segmentation-backup.ensightful.xyz",
                "model": "vit_l",
                "use_tensorrt": True,
                "alpha": 0.5,
                "timeout": 20,
                "enabled": False,
            },
        }

    def get_server_config(self, server_type: Optional[str] = None) -> Dict[str, Any]:
        key = f"{server_type or self.current_server}_server"
        return self.config.get(key, self.config.get("primary_server", {}))

    def switch_to_fallback(self) -> bool:
        fb = self.get_server_config("fallback")
        if self.current_server != "fallback" and fb.get("enabled", False):
            self.current_server = "fallback"
            logger.warning("[SAMConfig] Switched to fallback server")
            return True
        return False

    def reset_to_primary(self) -> None:
        self.current_server = "primary"

    def __repr__(self) -> str:
        url = self.get_server_config().get("server_url", "?")
        return f"SAMConfig(server={self.current_server}, url={url})"


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------
_rebar_config: Optional[RebarConfig] = None
_sam_config: Optional[SAMConfig] = None


def get_rebar_config(config_path: Optional[str] = None) -> RebarConfig:
    global _rebar_config
    if _rebar_config is None:
        _rebar_config = RebarConfig(config_path)
    return _rebar_config


def get_sam_config(config_path: Optional[str] = None) -> SAMConfig:
    global _sam_config
    if _sam_config is None:
        _sam_config = SAMConfig(config_path)
    return _sam_config
