"""Central application state objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..io.depth_utils import StereoDetectionResult, StereoFormat
from ..io.panorama_processing import (
    FisheyeConversionParams,
    PanoramaDetectionResult,
    PanoramaInputFormat,
    convert_fisheye_to_equirectangular,
    detect_panorama_format,
    split_panorama_views,
)
from .camera_pose import PanoramaMetadata


@dataclass(slots=True)
class PanoramaState:
    """Holds the currently loaded panorama assets and metadata."""

    image: Optional[np.ndarray] = None
    raw_image: Optional[np.ndarray] = None
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
    anaglyph_image: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    metadata: PanoramaMetadata = field(default_factory=PanoramaMetadata)
    stereo_detection: Optional[StereoDetectionResult] = None
    format_detection: Optional[PanoramaDetectionResult] = None
    format_override: Optional[PanoramaInputFormat] = None
    applied_format: PanoramaInputFormat = PanoramaInputFormat.MONO_EQUI
    depth_source: Optional[str] = None
    distortion_params: FisheyeConversionParams = field(default_factory=FisheyeConversionParams)
    distortion_corrected: bool = False
    render_eye: str = "left"

    def reset(self) -> None:
        self.image = None
        self.raw_image = None
        self.left_image = None
        self.right_image = None
        self.anaglyph_image = None
        self.depth = None
        self.metadata = PanoramaMetadata()
        self.stereo_detection = None
        self.format_detection = None
        self.format_override = None
        self.applied_format = PanoramaInputFormat.MONO_EQUI
        self.depth_source = None
        self.distortion_params = FisheyeConversionParams()
        self.distortion_corrected = False
        self.render_eye = "left"

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        if self.image is None:
            return None
        return int(self.image.shape[1]), int(self.image.shape[0])

    # ------------------------------------------------------------------
    def set_image(
        self,
        image: np.ndarray,
        source_path: Optional[Path] = None,
        detection: Optional[PanoramaDetectionResult] = None,
        override: Optional[PanoramaInputFormat] = None,
    ) -> None:
        """Assign the panorama image and derive the viewer-ready representation."""
        self.raw_image = np.ascontiguousarray(image)
        self.depth = None
        self.depth_source = None
        self.anaglyph_image = None
        self.render_eye = "left"
        self.distortion_corrected = False

        self.format_detection = detection or detect_panorama_format(self.raw_image)
        self.format_override = override if override not in {None, PanoramaInputFormat.AUTO} else None

        selected_format = self.format_override or self.format_detection.format
        viewer_image = self.raw_image
        self.left_image = None
        self.right_image = None

        # Construct auxiliary stereo detection for legacy consumers
        if self.format_detection.stereo_format is not None:
            message = self.format_detection.notes or "Auto-detected stereo panorama."
            self.stereo_detection = StereoDetectionResult(
                self.format_detection.stereo_format,
                confidence=self.format_detection.confidence,
                message=message,
            )
        else:
            self.stereo_detection = None

        if selected_format == PanoramaInputFormat.STEREO_SBS:
            left, right = split_panorama_views(
                self.raw_image,
                PanoramaInputFormat.STEREO_SBS,
                stereo_format=self.format_detection.stereo_format,
            )
            self.left_image, self.right_image = left, right
            viewer_image = left
        elif selected_format == PanoramaInputFormat.STEREO_TB:
            left, right = split_panorama_views(
                self.raw_image,
                PanoramaInputFormat.STEREO_TB,
                stereo_format=self.format_detection.stereo_format,
            )
            self.left_image, self.right_image = left, right
            viewer_image = left
        elif selected_format == PanoramaInputFormat.FISHEYE_STEREO:
            left, right = split_panorama_views(
                self.raw_image,
                PanoramaInputFormat.FISHEYE_STEREO,
                stereo_format=self.format_detection.stereo_format,
            )
            self.left_image = convert_fisheye_to_equirectangular(left, self.distortion_params)
            self.right_image = convert_fisheye_to_equirectangular(right, self.distortion_params)
            viewer_image = self.left_image
            self.applied_format = (
                PanoramaInputFormat.STEREO_SBS
                if (self.format_detection.stereo_format == StereoFormat.SIDE_BY_SIDE)
                else PanoramaInputFormat.STEREO_TB
            )
            self.distortion_corrected = True
        elif selected_format == PanoramaInputFormat.FISHEYE_MONO:
            self.left_image = convert_fisheye_to_equirectangular(self.raw_image, self.distortion_params)
            self.right_image = None
            viewer_image = self.left_image
            self.distortion_corrected = True
            self.applied_format = PanoramaInputFormat.MONO_EQUI
        else:
            self.left_image = np.ascontiguousarray(self.raw_image)
            self.right_image = None

        if selected_format not in {PanoramaInputFormat.FISHEYE_MONO, PanoramaInputFormat.FISHEYE_STEREO}:
            self.applied_format = selected_format

        self.image = np.ascontiguousarray(viewer_image)

        if source_path is not None:
            self.metadata.source_path = str(source_path)
            self.metadata.depth_path = None

    # ------------------------------------------------------------------
    def set_depth(
        self,
        depth: np.ndarray,
        depth_path: Optional[Path] = None,
        source: Optional[str] = None,
    ) -> None:
        depth = depth.astype("float32", copy=False)
        self.depth = depth
        self.depth_source = source
        if depth_path is not None:
            self.metadata.depth_path = str(depth_path)

    def ensure_depth(self) -> np.ndarray:
        if self.depth is None:
            raise ValueError("Depth map missing. Load or generate a depth map to continue.")
        return self.depth

    def ensure_image(self) -> np.ndarray:
        if self.image is None:
            raise ValueError("Panorama image missing. Load an equirectangular image first.")
        return self.image

    # ------------------------------------------------------------------
    @property
    def current_stereo_format(self) -> Optional[StereoFormat]:
        if self.right_image is not None:
            if self.stereo_detection is not None:
                return self.stereo_detection.format
            if self.format_detection and self.format_detection.stereo_format:
                return self.format_detection.stereo_format
        return None

    def set_render_eye(self, eye: str) -> np.ndarray:
        eye = eye.lower()
        if eye not in {"left", "right", "anaglyph"}:
            raise ValueError(f"Unsupported render eye: {eye}")
        if eye in {"right", "anaglyph"} and self.right_image is None:
            raise ValueError("Right eye data unavailable for this panorama")

        self.render_eye = eye
        self.anaglyph_image = None
        self._apply_render_eye()
        return self.ensure_image()

    def available_render_eyes(self) -> set[str]:
        options = {"left"}
        if self.right_image is not None:
            options.add("right")
            options.add("anaglyph")
        return options

    def _apply_render_eye(self) -> None:
        if self.render_eye == "right" and self.right_image is not None:
            self.image = np.ascontiguousarray(self.right_image)
        elif self.render_eye == "anaglyph" and self.left_image is not None and self.right_image is not None:
            self.image = self._compose_anaglyph()
        else:
            self.image = np.ascontiguousarray(self.left_image or self.raw_image)

    def _compose_anaglyph(self) -> np.ndarray:
        if self.anaglyph_image is not None:
            return self.anaglyph_image
        if self.left_image is None or self.right_image is None:
            return np.ascontiguousarray(self.left_image or self.raw_image)

        left = self.left_image.astype(np.float32)
        right = self.right_image.astype(np.float32)
        anaglyph = np.zeros_like(left, dtype=np.uint8)
        anaglyph[..., 0] = np.clip(left[..., 0], 0, 255).astype(np.uint8)
        green = 0.7 * right[..., 1] + 0.3 * right[..., 0]
        blue = 0.7 * right[..., 2] + 0.3 * right[..., 0]
        anaglyph[..., 1] = np.clip(green, 0, 255).astype(np.uint8)
        anaglyph[..., 2] = np.clip(blue, 0, 255).astype(np.uint8)
        self.anaglyph_image = anaglyph
        return anaglyph

    # ------------------------------------------------------------------
    def set_format_override(self, fmt: Optional[PanoramaInputFormat]) -> None:
        self.format_override = fmt if fmt not in {None, PanoramaInputFormat.AUTO} else None
        if self.raw_image is not None:
            source_path = Path(self.metadata.source_path) if self.metadata.source_path else None
            self.set_image(self.raw_image, source_path, detection=None, override=self.format_override)

    def apply_distortion_correction(self, params: FisheyeConversionParams) -> None:
        if self.raw_image is None:
            raise ValueError("No panorama loaded to correct")
        self.distortion_params = params
        source_format = self.format_detection.format if self.format_detection else PanoramaInputFormat.MONO_EQUI

        if source_format == PanoramaInputFormat.FISHEYE_MONO:
            corrected = convert_fisheye_to_equirectangular(self.raw_image, params)
            self.left_image = corrected
            self.right_image = None
            self.image = np.ascontiguousarray(corrected)
            self.applied_format = PanoramaInputFormat.MONO_EQUI
            self.distortion_corrected = True
        elif source_format == PanoramaInputFormat.FISHEYE_STEREO:
            left, right = split_panorama_views(
                self.raw_image,
                PanoramaInputFormat.FISHEYE_STEREO,
                stereo_format=self.format_detection.stereo_format if self.format_detection else None,
            )
            self.left_image = convert_fisheye_to_equirectangular(left, params)
            self.right_image = convert_fisheye_to_equirectangular(right, params)
            self.image = np.ascontiguousarray(self.left_image)
            self.applied_format = (
                PanoramaInputFormat.STEREO_SBS
                if (self.format_detection and self.format_detection.stereo_format == StereoFormat.SIDE_BY_SIDE)
                else PanoramaInputFormat.STEREO_TB
            )
            self.distortion_corrected = True
        else:
            corrected = convert_fisheye_to_equirectangular(self.image or self.raw_image, params)
            self.left_image = corrected
            self.right_image = None
            self.image = np.ascontiguousarray(corrected)
            self.applied_format = PanoramaInputFormat.MONO_EQUI
            self.distortion_corrected = True

        self.anaglyph_image = None
        self.render_eye = "left"
        self._apply_render_eye()

    # ------------------------------------------------------------------
    def format_summary(self) -> str:
        applied = self.applied_format.value
        if self.render_eye == "right":
            applied += " (right eye)"
        elif self.render_eye == "anaglyph":
            applied += " (anaglyph)"
        return applied
