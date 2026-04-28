"""
Step 5: 3D Spatial Analysis (optional, requires AIMatcher).

Calculates real-world spacing, uniformity, and mesh quality metrics
for fitted rebar lines.
"""

from typing import Dict, Optional

from loguru import logger


class SpatialAnalyzer:
    """3D spatial metrics calculator for rebar meshes (Step 5)."""

    def __init__(self, ai_matcher=None):
        self.ai_matcher = ai_matcher

    def calculate_3d_metrics(
        self,
        analyzer,
        pose_data_path=None,
        use_fine_grained: bool = True,
    ) -> Optional[Dict]:
        if not self.ai_matcher:
            logger.warning("[SpatialAnalyzer] No AIMatcher — skipping 3D analysis")
            return None
        try:
            if not hasattr(analyzer, "ai_matcher") or not analyzer.ai_matcher:
                analyzer.set_ai_matcher(self.ai_matcher)
            metrics = analyzer.calculate_3d_spatial_metrics(
                ai_matcher=self.ai_matcher,
                use_fine_grained_spacing=use_fine_grained,
            )
            if metrics:
                logger.info("[SpatialAnalyzer] 3D metrics computed")
            return metrics or None
        except Exception as e:
            logger.error(f"[SpatialAnalyzer] Failed: {e}")
            return None

    @staticmethod
    def get_spacing_uniformity(metrics: Dict) -> Dict:
        def _u(d):
            m, s = d.get("mean", 0), d.get("std", 0)
            return 1 / (1 + s / m) if m else None

        hu = _u(metrics.get("horizontal_3d_spacing", {}))
        vu = _u(metrics.get("vertical_3d_spacing", {}))
        vals = [x for x in (hu, vu) if x is not None]
        return {
            "horizontal_uniformity": hu,
            "vertical_uniformity": vu,
            "overall_uniformity": sum(vals) / len(vals) if vals else None,
        }
