import numpy as np
import pytest

from byakugan_app.io.depth_utils import (
    StereoDetectionResult,
    StereoFormat,
    compute_depth_from_panorama_array,
    detect_stereo_format,
    split_stereo_views,
)


def make_rgb(image: np.ndarray) -> np.ndarray:
    return np.stack([image] * 3, axis=2).astype(np.uint8)


def test_detect_stereo_format_side_by_side():
    panorama = np.zeros((512, 1024, 3), dtype=np.uint8)
    result = detect_stereo_format(panorama)
    assert isinstance(result, StereoDetectionResult)
    assert result.format == StereoFormat.SIDE_BY_SIDE
    assert result.is_stereo


def test_detect_stereo_format_top_bottom():
    panorama = np.zeros((640, 640, 3), dtype=np.uint8)
    result = detect_stereo_format(panorama)
    assert result.format == StereoFormat.TOP_BOTTOM


def test_split_stereo_views_top_bottom():
    top = np.full((64, 128, 3), 50, dtype=np.uint8)
    bottom = np.full((64, 128, 3), 200, dtype=np.uint8)
    panorama = np.vstack([top, bottom])
    left, right = split_stereo_views(panorama, StereoFormat.TOP_BOTTOM)
    assert left.shape == right.shape == (64, 128, 3)
    assert np.all(left == 50)
    assert np.all(right == 200)


def test_compute_depth_from_panorama_array_top_bottom():
    width = 128
    height_eye = 64
    disparity = 6

    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height_eye, 1))
    left_rgb = make_rgb(gradient)
    right_rgb = make_rgb(np.roll(gradient, -disparity, axis=1))
    panorama = np.vstack([left_rgb, right_rgb])

    depth = compute_depth_from_panorama_array(
        panorama,
        fmt=StereoFormat.TOP_BOTTOM,
        baseline_m=0.06,
        focal_length_px=120.0,
        rectify=False,
        downsample_factor=1,
    )
    assert depth.shape == (height_eye, width)
    assert depth.dtype == np.float32
    median_depth = float(np.nanmedian(depth))
    assert np.isfinite(median_depth)
    expected = (120.0 * 0.06) / disparity
    assert pytest.approx(median_depth, rel=0.3) == expected


def test_compute_depth_from_panorama_array_requires_stereo():
    mono = np.zeros((256, 512, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_depth_from_panorama_array(mono, fmt=StereoFormat.MONO)
