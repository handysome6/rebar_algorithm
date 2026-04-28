"""
Pure-HTTP SAM segmentation client (no Qt dependency).

Usage:
    client = SamClient("https://segmentation.ensightful.xyz")
    masks = client.segment(image_rgb, points=[[100, 200, 1]])
"""

import base64
import io
import time
from typing import List, Optional

import cv2
import numpy as np
import requests
from loguru import logger
from PIL import Image


class SamClient:
    """Synchronous HTTP client for the SAM segmentation server."""

    def __init__(
        self,
        server_url: str = "https://segmentation.ensightful.xyz",
        model: str = "vit_h",
        use_tensorrt: bool = True,
        alpha: float = 0.5,
        timeout: int = 30,
    ):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.use_tensorrt = use_tensorrt
        self.alpha = alpha
        self.timeout = timeout

    def segment(
        self,
        image: np.ndarray,
        points: List[List[int]],
    ) -> List[np.ndarray]:
        """
        Send an image + prompt points to the SAM server and return masks.

        Args:
            image: (H, W, 3) RGB uint8 array.
            points: List of [x, y, label] where label=1 for foreground.

        Returns:
            List of binary masks (H, W) uint8 (0 or 1).
        """
        orig_h, orig_w = image.shape[:2]

        # Resize to max 1024 long edge
        resized, scaled_pts, scale = self._resize(image, points)
        logger.info(f"[SAM] {orig_w}x{orig_h} -> {resized.shape[1]}x{resized.shape[0]}")

        # Encode
        b64 = self._encode(resized)

        # Request
        payload = {
            "model": self.model,
            "prompt_type": "point",
            "image": b64,
            "use_tensorrt": self.use_tensorrt,
            "alpha": self.alpha,
            "points": scaled_pts,
        }

        t0 = time.time()
        resp = requests.post(
            f"{self.server_url}/api/segment",
            json=payload,
            timeout=self.timeout,
            proxies={"http": None, "https": None},
        )
        logger.info(f"[SAM] Response in {time.time() - t0:.2f}s")

        if resp.status_code != 200:
            raise RuntimeError(f"SAM server error {resp.status_code}: {resp.text[:200]}")

        result = resp.json()
        masks_small = self._decode_masks(result)
        if not masks_small:
            raise RuntimeError("SAM returned no masks")

        # Resize masks back to original
        masks = []
        for m in masks_small:
            if scale != 1.0:
                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            masks.append(m)

        logger.info(f"[SAM] Returned {len(masks)} mask(s)")
        return masks

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _resize(image: np.ndarray, points: list) -> tuple:
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side <= 1024:
            return image, points, 1.0
        scale = 1024.0 / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scaled_pts = [[int(p[0] * scale), int(p[1] * scale), p[2]] for p in points]
        return resized, scaled_pts, scale

    @staticmethod
    def _encode(image_rgb: np.ndarray) -> str:
        pil = Image.fromarray(image_rgb.astype("uint8"), "RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _decode_masks(self, result: dict) -> List[np.ndarray]:
        masks_data = result.get("result", {}).get("masks", [])
        out: List[np.ndarray] = []
        for item in masks_data:
            raw = item.get("mask", item) if isinstance(item, dict) else item
            if "," in raw:
                raw = raw.split(",", 1)[1]
            img = Image.open(io.BytesIO(base64.b64decode(raw)))
            arr = np.array(img)
            # Convert to binary mask (H, W) uint8
            if arr.ndim == 3 and arr.shape[2] == 4:
                binary = (arr[..., 3] > 0).astype(np.uint8)
            elif arr.ndim == 3:
                binary = (arr.mean(axis=2) > 0).astype(np.uint8)
            else:
                binary = (arr > 0).astype(np.uint8)
            out.append(binary)
        return out
