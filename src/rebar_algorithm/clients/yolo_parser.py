"""
Parser for YOLO pose detection results (pose_data.json).

Extracts weighted centre points from 4-keypoint quadrilateral detections.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class YoloResultParser:
    """Parse YOLO keypoint detection results and extract knot centres."""

    def __init__(
        self,
        json_path: str,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        self.json_path = json_path
        self.image_width = image_width
        self.image_height = image_height
        self.raw_data: list = []
        self.parsed_objects: List[Dict] = []
        self.centers: List[Dict] = []

        self._load_data()
        self._parse_data()

    def _load_data(self) -> None:
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

    def _parse_data(self) -> None:
        self.parsed_objects = []
        self.centers = []

        for i, item in enumerate(self.raw_data):
            obj = {
                "object_id": i,
                "class_id": item["class"]["id"],
                "class_name": item["class"]["name"],
                "object_confidence": item["confidence"],
                "bbox": item["bbox"],
                "keypoints": item.get("keypoints", []),
                "visible_keypoints": [],
            }
            for j, kp in enumerate(obj["keypoints"]):
                if kp["visibility"] == 2:
                    obj["visible_keypoints"].append(
                        {
                            "index": j,
                            "x": kp["x"],
                            "y": kp["y"],
                            "confidence": kp["confidence"],
                            "visibility": kp["visibility"],
                        }
                    )
            self.parsed_objects.append(obj)
            center = self._calculate_center(obj)
            if center:
                self.centers.append(center)

    def _calculate_center(self, obj: Dict) -> Optional[Dict]:
        visible = obj["visible_keypoints"]
        if not visible:
            return None
        total_w = 0.0
        wx, wy = 0.0, 0.0
        for kp in visible:
            c = kp["confidence"]
            x, y = kp["x"], kp["y"]
            if self.image_width is not None and self.image_height is not None:
                x *= self.image_width
                y *= self.image_height
            wx += x * c
            wy += y * c
            total_w += c
        if total_w <= 0:
            return None
        return {
            "object_id": obj["object_id"],
            "class_id": obj["class_id"],
            "class_name": obj["class_name"],
            "object_confidence": obj["object_confidence"],
            "center_x": wx / total_w,
            "center_y": wy / total_w,
            "keypoint_count": len(visible),
            "total_weight": total_w,
            "bbox": obj["bbox"],
        }

    def get_centers_coordinates(self) -> List[Tuple[float, float]]:
        return [(c["center_x"], c["center_y"]) for c in self.centers]

    def get_centers_by_class(self, class_name: str) -> List[Tuple[float, float]]:
        return [
            (c["center_x"], c["center_y"])
            for c in self.centers
            if c["class_name"] == class_name
        ]

    def get_centers_info(self) -> List[Dict]:
        return self.centers.copy()

    def set_image_dimensions(self, width: int, height: int) -> None:
        self.image_width = width
        self.image_height = height
        self._parse_data()

    def draw_centers_on_image(self, image_path: str, output_path: str) -> None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]
        if self.image_width != w or self.image_height != h:
            self.set_image_dimensions(w, h)
        for c in self.centers:
            x, y = int(c["center_x"]), int(c["center_y"])
            cv2.circle(img, (x, y), 8, (0, 255, 255), -1)
            cv2.circle(img, (x, y), 10, (0, 0, 0), 2)
        cv2.imwrite(output_path, img)
