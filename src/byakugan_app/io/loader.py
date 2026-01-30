"""File loading and preprocessing utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


@dataclass(slots=True)
class StereoDepthConfig:
    """Configuration for stereo depth estimation."""

    baseline_m: float
    focal_length_px: float
    block_size: int = 11
    num_disparities: int = 128
    speckle_range: int = 16
    speckle_window_size: int = 200


def load_equirectangular_image(path: Path) -> np.ndarray:
    """Load an equirectangular panorama as an RGB float32 array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read panorama image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.debug("Loaded panorama image {} with shape {}", path, image.shape)
    return image


def load_depth_map(path: Path) -> np.ndarray:
    """Load a depth map, coercing to float32 meters."""
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Unable to read depth map: {path}")

    if depth_raw.dtype == np.uint16:
        logger.debug("Depth map {} detected as uint16; assuming millimetres", path)
        depth_m = depth_raw.astype(np.float32) / 1000.0
    elif depth_raw.dtype == np.uint8:
        logger.debug(
            "Depth map {} detected as uint8; scaling to meters via 8-bit normalisation", path
        )
        depth_m = (depth_raw.astype(np.float32) / 255.0) * depth_raw.max()
    else:
        depth_m = depth_raw.astype(np.float32)

    if depth_m.ndim == 3:
        depth_m = cv2.cvtColor(depth_m, cv2.COLOR_BGR2GRAY)

    logger.debug(
        "Loaded depth map {} with min {:.3f} m, max {:.3f} m",
        path,
        float(np.nanmin(depth_m)),
        float(np.nanmax(depth_m)),
    )
    return depth_m


def compute_depth_from_stereo(
    left_path: Path,
    right_path: Path,
    config: StereoDepthConfig,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a depth map from stereo input using Semi-Global Block Matching.

    Parameters
    ----------
    left_path, right_path: pathlib.Path
        File paths to the rectified left/right images.
    config: StereoDepthConfig
        Holds baseline and focal length information required to convert disparity to depth.
    mask: np.ndarray, optional
        Optional mask to invalidate noisy regions.
    """
    left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None:
        raise FileNotFoundError("Stereo images could not be loaded")

    logger.info("Starting StereoSGBM depth reconstruction")
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=config.num_disparities,
        blockSize=config.block_size,
        speckleRange=config.speckle_range,
        speckleWindowSize=config.speckle_window_size,
        uniquenessRatio=10,
        disp12MaxDiff=1,
    )
    disparity = matcher.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan

    depth = (config.focal_length_px * config.baseline_m) / disparity
    if mask is not None:
        depth[mask == 0] = np.nan

    logger.info("Stereo depth reconstruction complete")
    return depth.astype(np.float32)
