"""Dataclasses for storing selection results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(slots=True)
class PixelSelection:
    """Represents a selection in pixel coordinates."""

    u: int
    v: int
    width: int
    height: int

    def normalised(self) -> Tuple[float, float]:
        """Return the (u,v) in [0,1] range."""
        return ((self.u + 0.5) / self.width, (self.v + 0.5) / self.height)


@dataclass(slots=True)
class PointMeasurement:
    """Stores the complete result of a measurement."""

    pixel: PixelSelection
    depth_m: float
    enu_vector: Tuple[float, float, float]
    geodetic: Tuple[float, float, float]
    quality_score: Optional[float] = None
    quality_label: str = "N/A"

    @property
    def latitude(self) -> float:
        return self.geodetic[0]

    @property
    def longitude(self) -> float:
        return self.geodetic[1]

    @property
    def altitude(self) -> float:
        return self.geodetic[2]

    def serialize(self) -> dict:
        """Serialize record for export."""
        return {
            "pixel_u": self.pixel.u,
            "pixel_v": self.pixel.v,
            "depth_m": round(float(self.depth_m), 6),
            "enu_e": round(float(self.enu_vector[0]), 6),
            "enu_n": round(float(self.enu_vector[1]), 6),
            "enu_u": round(float(self.enu_vector[2]), 6),
            "latitude": round(float(self.latitude), 6),
            "longitude": round(float(self.longitude), 6),
            "altitude": round(float(self.altitude), 6),
            "quality_score": None if self.quality_score is None else round(float(self.quality_score), 2),
            "quality_label": self.quality_label,
        }
