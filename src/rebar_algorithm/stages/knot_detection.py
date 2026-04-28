"""
Step 3: Rebar Knot Detection via YOLO server.

Sends segmented rebar images to a remote YOLO server.
Returns pose data (JSON with 4-keypoint quadrilateral detections).
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class KnotDetector:
    """YOLO-based rebar knot detector (Step 3)."""

    def __init__(self, server_url: str, timeout: int = 60):
        self.server_url = server_url
        self.timeout = timeout

    def detect_knots(
        self,
        segmented_image_path: Path,
        output_path: Path,
        use_existing: bool = False,
    ) -> Dict:
        logger.info(f"[KnotDetector] server={self.server_url}")
        yolo_dir = output_path / "yolo_results"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        pose_path = yolo_dir / "pose_data.json"

        if use_existing and self._check_existing(pose_path, output_path):
            logger.info("[KnotDetector] Using existing annotations")
            return {"pose_data_path": pose_path, "found_existing": True,
                    "knot_count": self._count_knots(pose_path)}

        logger.info("[KnotDetector] Running YOLO detection...")
        result = self._run_detection(segmented_image_path, yolo_dir)
        return {"pose_data_path": pose_path, "found_existing": False,
                "knot_count": self._count_knots(pose_path), "result": result}

    def _check_existing(self, pose_path: Path, output_path: Path) -> bool:
        if pose_path.exists():
            return True
        legacy = output_path.parent / "yolo_results" / "pose_data.json"
        if legacy.exists():
            shutil.copy2(legacy, pose_path)
            return True
        return False

    def _run_detection(self, image_path: Path, output_dir: Path) -> Dict:
        from ..clients.yolo_client import YoloClient

        client = YoloClient(server_url=self.server_url)
        return client.process_image(str(image_path), str(output_dir))

    @staticmethod
    def _count_knots(path: Path) -> Optional[int]:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return len(data) if isinstance(data, list) else None
        except Exception:
            return None

    @staticmethod
    def validate_detection_results(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                return False
            logger.info(f"[KnotDetector] Validated: {len(data)} knots")
            return True
        except Exception:
            return False
