"""
HTTP client for the YOLO rebar knot detection server.

Sends a segmented rebar image, receives a ZIP with pose_data.json + visualisations.
"""

import io
import os
import zipfile
from pathlib import Path

import requests
from loguru import logger


class YoloClient:
    """Synchronous HTTP client for the YOLO knot detection server."""

    def __init__(self, server_url: str = "http://localhost:2001"):
        self.server_url = server_url.rstrip("/")
        self.endpoint = f"{self.server_url}/process_pose/"

    @classmethod
    def from_config(cls, config_path: str = None) -> "YoloClient":
        from rebar_algorithm.config import get_rebar_config

        cfg = get_rebar_config(config_path)
        return cls(server_url=cfg.get_server_url())

    def process_image(self, image_path: str, output_dir: str = "./client_results") -> dict:
        """
        Upload image for knot detection and save results.

        Returns:
            dict mapping filename -> saved path for each extracted file.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info(f"[YOLO] Uploading {image_path}")

        with open(image_path, "rb") as f:
            files = {"image": ("image.png", f, "image/png")}
            resp = requests.post(
                self.endpoint,
                files=files,
                timeout=60,
                proxies={"http": None, "https": None},
            )
        resp.raise_for_status()

        return self._extract_zip(resp.content, out)

    @staticmethod
    def _extract_zip(content: bytes, output_dir: Path) -> dict:
        result_files = {}
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            for info in zf.infolist():
                dest = output_dir / info.filename
                with zf.open(info.filename) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                result_files[info.filename] = str(dest)
                logger.info(f"[YOLO] Extracted: {info.filename}")
        return result_files
