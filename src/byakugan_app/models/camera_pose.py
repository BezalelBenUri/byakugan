"""Camera pose domain models."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict


@dataclass(slots=True)
class CameraPose:
    """Represents the pose of the panorama capture location."""

    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters
    bearing: float = 0.0  # degrees, clockwise from north

    def to_dict(self) -> Dict[str, float]:
        """Return a serialisable mapping."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "bearing": self.bearing,
        }

    @property
    def bearing_rad(self) -> float:
        """Bearing in radians."""
        return math.radians(self.bearing)


@dataclass(slots=True)
class PanoramaMetadata:
    """Metadata attached to a panorama asset."""

    camera_pose: CameraPose = field(default_factory=lambda: CameraPose(0.0, 0.0, 0.0))
    source_path: str | None = None
    depth_path: str | None = None
    notes: str | None = None

    def to_dict(self) -> Dict[str, float | str | None]:
        """Return metadata suitable for JSON export."""
        base = self.camera_pose.to_dict()
        base.update(
            {
                "source_path": self.source_path,
                "depth_path": self.depth_path,
                "notes": self.notes,
            }
        )
        return base
