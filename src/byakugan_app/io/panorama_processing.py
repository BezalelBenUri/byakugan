"""Panorama format detection and fisheye correction utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from .depth_utils import StereoFormat, split_stereo_views


class PanoramaInputFormat(Enum):
    """Supported panorama encodings presented to the viewer."""

    AUTO = "Auto"
    MONO_EQUI = "Mono Equirectangular"
    STEREO_TB = "Stereo Top-Bottom"
    STEREO_SBS = "Stereo Side-by-Side"
    FISHEYE_MONO = "Fisheye Mono"
    FISHEYE_STEREO = "Fisheye Stereo"

    def __str__(self) -> str:  # pragma: no cover - user friendly label
        return self.value


@dataclass(slots=True)
class PanoramaDetectionResult:
    """Outcome of analysing a panorama file."""

    format: PanoramaInputFormat
    confidence: float
    notes: str = ""
    stereo_format: Optional[StereoFormat] = None
    probable_fisheye: bool = False


@dataclass(slots=True)
class FisheyeConversionParams:
    """Parameters supplied when remapping fisheye imagery to equirectangular."""

    fov_deg: float = 195.0
    output_width: Optional[int] = None
    center_x: Optional[float] = None  # normalised [0, 1]
    center_y: Optional[float] = None  # normalised [0, 1]

    def resolve_output_size(self, source_shape: Tuple[int, int]) -> Tuple[int, int]:
        height, width = source_shape
        target_width = self.output_width or (max(width * 2, 2048))
        target_height = target_width // 2
        return target_height, target_width

    def resolve_center(self, source_shape: Tuple[int, int]) -> Tuple[float, float]:
        height, width = source_shape
        cx = (self.center_x if self.center_x is not None else 0.5) * width
        cy = (self.center_y if self.center_y is not None else 0.5) * height
        return float(cx), float(cy)


def detect_panorama_format(image: np.ndarray) -> PanoramaDetectionResult:
    """Inspect dimensions/intensity to infer panorama layout."""
    height, width = image.shape[:2]
    aspect = width / float(height)
    tolerance = 0.08
    probable_fisheye = _is_probable_fisheye(image)
    notes = []
    stereo_fmt: Optional[StereoFormat] = None

    if abs(aspect - 2.0) <= tolerance:
        if probable_fisheye:
            notes.append("Dark border suggests fisheye capture.")
            return PanoramaDetectionResult(
                PanoramaInputFormat.FISHEYE_MONO,
                confidence=max(0.5, 1.0 - abs(aspect - 2.0)),
                notes=" ".join(notes),
                probable_fisheye=True,
            )
        return PanoramaDetectionResult(
            PanoramaInputFormat.MONO_EQUI,
            confidence=max(0.5, 1.0 - abs(aspect - 2.0)),
            notes="Dimensions consistent with 2:1 equirectangular.",
        )

    if abs(aspect - 4.0) <= tolerance:
        stereo_fmt = StereoFormat.SIDE_BY_SIDE
        if probable_fisheye:
            format_guess = PanoramaInputFormat.FISHEYE_STEREO
            notes.append("4:1 aspect with circular content; treating as fisheye SBS stereo.")
        else:
            format_guess = PanoramaInputFormat.STEREO_SBS
            notes.append("Width approximately twice each eye; treating as side-by-side stereo.")
        return PanoramaDetectionResult(
            format_guess,
            confidence=max(0.4, 1.0 - abs(aspect - 4.0)),
            notes=" ".join(notes),
            stereo_format=stereo_fmt,
            probable_fisheye=probable_fisheye,
        )

    if abs(aspect - 1.0) <= tolerance:
        stereo_fmt = StereoFormat.TOP_BOTTOM
        if probable_fisheye:
            format_guess = PanoramaInputFormat.FISHEYE_STEREO
            notes.append("Square frame with circular halves; treating as fisheye top-bottom stereo.")
        else:
            format_guess = PanoramaInputFormat.STEREO_TB
            notes.append("Square frame assumed to be top-bottom stereo.")
        return PanoramaDetectionResult(
            format_guess,
            confidence=max(0.4, 1.0 - abs(aspect - 1.0)),
            notes=" ".join(notes),
            stereo_format=stereo_fmt,
            probable_fisheye=probable_fisheye,
        )

    # Fallback heuristics
    if probable_fisheye:
        notes.append("Unrecognised aspect but strong fisheye cues present.")
        return PanoramaDetectionResult(
            PanoramaInputFormat.FISHEYE_MONO,
            confidence=0.3,
            notes=" ".join(notes),
            probable_fisheye=True,
        )

    notes.append("Unrecognised aspect ratio; defaulting to mono equirectangular.")
    return PanoramaDetectionResult(
        PanoramaInputFormat.MONO_EQUI,
        confidence=0.2,
        notes=" ".join(notes),
    )


def split_panorama_views(
    image: np.ndarray,
    fmt: PanoramaInputFormat,
    stereo_format: Optional[StereoFormat] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a stereo panorama into left/right images."""
    if fmt not in {
        PanoramaInputFormat.STEREO_SBS,
        PanoramaInputFormat.STEREO_TB,
        PanoramaInputFormat.FISHEYE_STEREO,
    }:
        raise ValueError(f"Format {fmt} does not contain stereo views")

    if stereo_format is None:
        if fmt in {PanoramaInputFormat.STEREO_TB, PanoramaInputFormat.FISHEYE_STEREO}:
            height, width = image.shape[:2]
            stereo_format = StereoFormat.TOP_BOTTOM if height >= width else StereoFormat.SIDE_BY_SIDE
        else:
            stereo_format = StereoFormat.SIDE_BY_SIDE

    left, right = split_stereo_views(image, stereo_format)
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def convert_fisheye_to_equirectangular(
    image: np.ndarray,
    params: Optional[FisheyeConversionParams] = None,
) -> np.ndarray:
    """Project a fisheye frame into equirectangular space."""
    if params is None:
        params = FisheyeConversionParams()

    src_height, src_width = image.shape[:2]
    dst_height, dst_width = params.resolve_output_size((src_height, src_width))
    cx, cy = params.resolve_center((src_height, src_width))

    fov = math.radians(params.fov_deg)
    focal = src_width / max(fov, 1e-6)

    lon = np.linspace(-math.pi, math.pi, dst_width, dtype=np.float32)
    lat = np.linspace(math.pi / 2.0, -math.pi / 2.0, dst_height, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.sin(lon_grid)

    theta = np.arccos(np.clip(x, -1.0, 1.0))
    sin_theta = np.sin(theta)
    phi = np.arctan2(z, y)

    with np.errstate(divide="ignore", invalid="ignore"):
        ux = np.where(sin_theta > 1e-6, z / sin_theta, 0.0)
        uy = np.where(sin_theta > 1e-6, y / sin_theta, 0.0)

    radius = focal * theta
    map_x = (cx + radius * ux).astype(np.float32)
    map_y = (cy + radius * uy).astype(np.float32)

    remapped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.ascontiguousarray(remapped)


def _is_probable_fisheye(image: np.ndarray) -> bool:
    """Check border intensity for black corners indicative of fisheye capture."""
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return False

    border = int(min(height, width) * 0.12)
    if border <= 0:
        return False

    corners = [
        image[:border, :border],
        image[:border, -border:],
        image[-border:, :border],
        image[-border:, -border:],
    ]
    darkness = [float(np.mean(cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY))) for corner in corners]
    dark_fraction = sum(val < 18.0 for val in darkness) / len(darkness)
    return dark_fraction >= 0.75
