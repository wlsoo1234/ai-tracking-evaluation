"""Coordinate convention: bottom-center of the person bbox, as a floor-position
approximation. These are IMAGE-PLANE pixel coordinates, not physical/metric
ones — no camera calibration or homography exists for this sample, so no
real-world conversion is performed. See system_architecture.md for the
optional (undemonstrated) branch that would use one if it existed.
"""
from __future__ import annotations

from typing import List, Tuple


def bottom_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def normalize(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    return x / width, y / height
