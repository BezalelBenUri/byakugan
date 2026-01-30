"""Dataclasses for storing selection results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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
            "depth_m": self.depth_m,
            "enu_e": self.enu_vector[0],
            "enu_n": self.enu_vector[1],
            "enu_u": self.enu_vector[2],
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
        }
