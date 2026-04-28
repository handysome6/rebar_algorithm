"""Shared type definitions for the rebar pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Line represented as (x1, y1, x2, y2) float tuple
Line = Tuple[float, float, float, float]

FittedLines = Dict[str, List[Line]]


@dataclass
class PlaneModel:
    """Fitted 3D plane: ax + by + cz + d = 0."""
    a: float
    b: float
    c: float
    d: float

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}
