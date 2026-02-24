"""Stereo panorama detection and depth computation helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from .loader import load_equirectangular_image, StereoDepthConfig


class StereoFormat(Enum):
    """Supported stereo panorama encodings."""

    MONO = "Mono"
    TOP_BOTTOM = "Top-Bottom"
    SIDE_BY_SIDE = "Side-by-Side"

    def __str__(self) -> str:  # pragma: no cover - convenience for UI display
        return self.value


@dataclass(slots=True)
class StereoDetectionResult:
    """Outcome of auto-detecting stereo layout."""

    format: StereoFormat
    confidence: float
    message: str

    @property
    def is_stereo(self) -> bool:
        return self.format != StereoFormat.MONO


DEFAULT_BASELINE_M = 0.06  # metres, Insta360 Pro2 baseline


def detect_stereo_format(image: np.ndarray, tolerance: float = 0.03) -> StereoDetectionResult:
    """Infer stereo arrangement from the panorama dimensions."""
    height, width = image.shape[:2]
    ratio_wh = width / float(height)
    ratio_hw = height / float(width)

    if abs(ratio_wh - 2.0) <= tolerance:
        message = "Width is approximately twice the height; treating panorama as side-by-side stereo."
        confidence = 1.0 - abs(ratio_wh - 2.0)
        return StereoDetectionResult(StereoFormat.SIDE_BY_SIDE, confidence, message)

    if abs(ratio_hw - 1.0) <= tolerance and height % 2 == 0:
        mid = height // 2
        upper = image[:mid]
        lower = image[mid:]
        upper_mean = float(np.mean(cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)))
        lower_mean = float(np.mean(cv2.cvtColor(lower, cv2.COLOR_RGB2GRAY)))
        lum_delta = abs(upper_mean - lower_mean)
        confidence = (1.0 - abs(ratio_hw - 1.0)) * 0.6 + min(lum_delta / 255.0, 0.4)
        message = "Height is approximately equal to the width; treating panorama as top-bottom stereo (upper=left eye)."
        return StereoDetectionResult(StereoFormat.TOP_BOTTOM, confidence, message)

    message = "Dimensions consistent with monoscopic panorama."
    return StereoDetectionResult(StereoFormat.MONO, 0.0, message)





def split_stereo_views(image: np.ndarray, fmt: StereoFormat) -> Tuple[np.ndarray, np.ndarray]:
    """Split stereo panorama into left/right eye views."""
    if fmt == StereoFormat.MONO:
        raise ValueError("Mono panorama cannot be split into stereo views")

    height, width = image.shape[:2]
    if fmt == StereoFormat.TOP_BOTTOM:
        mid = height // 2
        left = image[:mid].copy()
        right = image[mid:].copy()
    elif fmt == StereoFormat.SIDE_BY_SIDE:
        mid = width // 2
        left = image[:, :mid].copy()
        right = image[:, mid:].copy()
    else:  # pragma: no cover - safety guard
        raise ValueError(f"Unsupported stereo format: {fmt}")

    if left.shape[:2] != right.shape[:2]:
        raise ValueError("Stereo views have mismatched dimensions after split")
    return left, right


def estimate_default_focal(width: int) -> float:
    """Estimate focal length (pixels) for an equirectangular panorama."""
    return width / (2.0 * math.pi)


def compute_depth_from_panorama(
    panorama_path: Path,
    fmt: StereoFormat,
    baseline_m: float = DEFAULT_BASELINE_M,
    focal_length_px: Optional[float] = None,
    rectify: bool = False,
    downsample_factor: int = 1,
) -> np.ndarray:
    """Compute a depth map directly from a stereo panorama file."""
    image = load_equirectangular_image(panorama_path)
    return compute_depth_from_panorama_array(
        image,
        fmt=fmt,
        baseline_m=baseline_m,
        focal_length_px=focal_length_px,
        rectify=rectify,
        downsample_factor=downsample_factor,
    )


def compute_depth_from_panorama_array(
    image: np.ndarray,
    fmt: StereoFormat,
    baseline_m: float = DEFAULT_BASELINE_M,
    focal_length_px: Optional[float] = None,
    rectify: bool = False,
    downsample_factor: int = 1,
) -> np.ndarray:
    detection = detect_stereo_format(image)
    if fmt == StereoFormat.MONO:
        raise ValueError("Cannot compute stereo depth from mono panorama")

    if detection.format != fmt:
        logger.info("Using manual stereo format %s (detected %s)", fmt, detection.format)

    left_rgb, right_rgb = split_stereo_views(image, fmt)

    target_height, target_width = left_rgb.shape[:2]

    if downsample_factor > 1:
        ds_size = (target_width // downsample_factor, target_height // downsample_factor)
        left_proc = cv2.resize(left_rgb, ds_size, interpolation=cv2.INTER_AREA)
        right_proc = cv2.resize(right_rgb, ds_size, interpolation=cv2.INTER_AREA)
    else:
        left_proc = left_rgb
        right_proc = right_rgb

    left_gray = cv2.cvtColor(left_proc, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_proc, cv2.COLOR_RGB2GRAY)

    if focal_length_px is None:
        focal_length_px = estimate_default_focal(target_width)
        logger.debug("Estimated focal length %.2f px for width %s", focal_length_px, target_width)

    if rectify:
        logger.warning(
            "Rectification requested; equirectangular rectification not implemented. Proceeding without modifications."
        )

    config = StereoDepthConfig(
        baseline_m=baseline_m,
        focal_length_px=focal_length_px,
        block_size=5,
        num_disparities=128,
        speckle_range=32,
        speckle_window_size=100,
    )

    depth = compute_depth_from_stereo_images(left_gray, right_gray, config)

    if downsample_factor > 1:
        depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    return depth.astype(np.float32)


def compute_depth_from_stereo_images(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    config: StereoDepthConfig,
) -> np.ndarray:
    """Run Semi-Global Block Matching on grayscale stereo views.

    OpenCV requires ``numDisparities`` to be a positive multiple of 16 and
    practically bounded by image width. We clamp it defensively so small test
    images and reduced-resolution frames do not trigger allocation failures.
    """
    image_width = int(left_gray.shape[1])
    if image_width < 32:
        raise ValueError(
            f"Stereo input width {image_width} is too small for SGBM. Minimum supported width is 32 px."
        )

    # SGBM expects minDisparity + numDisparities < image width.
    max_disparities = ((image_width - 16) // 16) * 16
    max_disparities = max(16, max_disparities)
    num_disparities = max(16, (int(config.num_disparities) // 16) * 16)
    num_disparities = min(num_disparities, max_disparities)

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=config.block_size,
        P1=8 * 3 * config.block_size**2,
        P2=32 * 3 * config.block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan

    epsilon = 1e-6
    depth = (config.focal_length_px * config.baseline_m) / (disparity + epsilon)
    depth = np.where(np.isfinite(depth), depth, np.nan)

    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size:
        range_min = float(np.min(finite_depth))
        range_max = float(np.max(finite_depth))
    else:
        range_min = float("nan")
        range_max = float("nan")

    logger.info(
        "Stereo depth computed: baseline=%.3f m, focal=%.2f px, disparities=%d, range=[%.2f, %.2f] m",
        config.baseline_m,
        config.focal_length_px,
        num_disparities,
        range_min,
        range_max,
    )
    return depth.astype(np.float32)
