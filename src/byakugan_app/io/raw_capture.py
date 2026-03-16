"""Raw multi-sensor capture ingestion and calibrated ray construction.

This module supports i-Pulse style raw folders that contain:
- ``*-calibration.yaml``
- ``*-gps.txt``
- ``*-imu.csv`` (optional)
- ``*-sensor-{1..4}.mkv``
- ``*-timestamps-{1..4}.txt``

The output is a typed representation that downstream geometry code can use to
build calibrated sensor rays directly, before stitched panorama warping.
"""
from __future__ import annotations

from dataclasses import dataclass
import bisect
import csv
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..math import geodesy


@dataclass(slots=True, frozen=True)
class RawSensorCalibration:
    """Intrinsic + extrinsic calibration for one physical sensor."""

    sensor_index: int
    focal_px: float
    k1: float
    k2: float
    k3: float
    center_x: float
    center_y: float
    tm_b_to_c: np.ndarray
    tm_c_to_b: np.ndarray


@dataclass(slots=True, frozen=True)
class RawGpsSample:
    """One raw GNSS sample aligned by timestamp."""

    timestamp_sec: float
    latitude: float
    longitude: float
    altitude_m: float
    heading_deg: float
    horizontal_accuracy_m: Optional[float]
    vertical_accuracy_m: Optional[float]
    heading_accuracy_deg: Optional[float]


@dataclass(slots=True, frozen=True)
class RawCapture:
    """Parsed raw multi-sensor capture folder."""

    capture_root: Path
    calibration_path: Path
    gps_path: Path
    imu_path: Optional[Path]
    sensor_video_paths: dict[int, Path]
    timestamp_paths: dict[int, Path]
    calibrations: tuple[RawSensorCalibration, ...]
    gps_samples: tuple[RawGpsSample, ...]
    sensor_timestamps_ns: dict[int, tuple[int, ...]]


@dataclass(slots=True, frozen=True)
class RawRayObservation:
    """Calibrated world-space ray observation from one raw sensor frame."""

    sensor_index: int
    frame_index: int
    timestamp_sec: float
    origin_ecef: np.ndarray
    direction_ecef: np.ndarray
    horizontal_accuracy_m: Optional[float]
    heading_accuracy_deg: Optional[float]


def discover_raw_capture(path: Path) -> RawCapture:
    """Discover and parse a raw multi-sensor capture folder."""
    root = path.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Raw capture path does not exist: {root}")

    calibration_path = _find_single(root, "*-calibration.yaml")
    gps_path = _find_single(root, "*-gps.txt")
    imu_path = _find_optional(root, "*-imu.csv")
    sensor_paths = {idx: _find_single(root, f"*-sensor-{idx}.mkv") for idx in range(1, 5)}
    timestamp_paths = {idx: _find_single(root, f"*-timestamps-{idx}.txt") for idx in range(1, 5)}

    calibrations = _parse_calibration(calibration_path)
    gps_samples = _parse_raw_gps(gps_path)
    sensor_timestamps = {
        idx: tuple(_parse_sensor_timestamps(timestamp_paths[idx]))
        for idx in range(1, 5)
    }

    return RawCapture(
        capture_root=root,
        calibration_path=calibration_path,
        gps_path=gps_path,
        imu_path=imu_path,
        sensor_video_paths=sensor_paths,
        timestamp_paths=timestamp_paths,
        calibrations=tuple(calibrations),
        gps_samples=tuple(gps_samples),
        sensor_timestamps_ns=sensor_timestamps,
    )


def build_raw_observation(
    capture: RawCapture,
    *,
    sensor_index: int,
    frame_index: int,
    pixel_u: float,
    pixel_v: float,
) -> RawRayObservation:
    """Build one calibrated raw-sensor ray observation in ECEF coordinates."""
    if sensor_index not in capture.sensor_timestamps_ns:
        raise ValueError(f"Unknown sensor index {sensor_index}.")
    timestamps = capture.sensor_timestamps_ns[sensor_index]
    if frame_index < 0 or frame_index >= len(timestamps):
        raise ValueError(f"Frame index {frame_index} out of range for sensor {sensor_index}.")
    timestamp_sec = timestamps[frame_index] / 1_000_000_000.0
    gps_sample = _nearest_gps_sample(capture.gps_samples, timestamp_sec)
    calibration = capture.calibrations[sensor_index - 1]

    direction_cam = _fisheye_direction_from_pixel(
        pixel_u,
        pixel_v,
        calibration,
    )
    direction_body = calibration.tm_c_to_b[:3, :3] @ direction_cam
    direction_body = direction_body / max(np.linalg.norm(direction_body), 1e-12)
    direction_enu = _body_direction_to_enu(direction_body, gps_sample.heading_deg)
    direction_ecef = np.asarray(
        geodesy.enu_to_ecef(
            float(direction_enu[0]),
            float(direction_enu[1]),
            float(direction_enu[2]),
            gps_sample.latitude,
            gps_sample.longitude,
        ),
        dtype=np.float64,
    )
    direction_ecef /= max(np.linalg.norm(direction_ecef), 1e-12)
    origin_ecef = np.asarray(
        geodesy.geodetic_to_ecef(
            gps_sample.longitude,
            gps_sample.latitude,
            gps_sample.altitude_m,
        ),
        dtype=np.float64,
    )
    return RawRayObservation(
        sensor_index=sensor_index,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        origin_ecef=origin_ecef,
        direction_ecef=direction_ecef,
        horizontal_accuracy_m=gps_sample.horizontal_accuracy_m,
        heading_accuracy_deg=gps_sample.heading_accuracy_deg,
    )


def _parse_calibration(path: Path) -> list[RawSensorCalibration]:
    """Read OpenCV calibration YAML with per-sensor camera models."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Unable to open calibration file: {path}")
    cameras = storage.getNode("cameras")
    if cameras.empty():
        storage.release()
        raise ValueError(f"Calibration file missing 'cameras' node: {path}")

    parsed: list[RawSensorCalibration] = []
    for idx in range(cameras.size()):
        camera = cameras.at(idx)
        lens = camera.getNode("lensCoeffs").mat()
        center = camera.getNode("centre").mat()
        tm_b_to_c = camera.getNode("tmBToC").mat()
        if lens is None or center is None or tm_b_to_c is None:
            storage.release()
            raise ValueError(f"Calibration camera {idx + 1} missing required fields.")
        lens_values = lens.reshape(-1).astype(np.float64)
        if lens_values.size < 4:
            storage.release()
            raise ValueError(f"Calibration camera {idx + 1} lensCoeffs must contain at least 4 values.")
        center_values = center.reshape(-1).astype(np.float64)
        tm_b_to_c = np.asarray(tm_b_to_c, dtype=np.float64).reshape(4, 4)
        tm_c_to_b = np.linalg.inv(tm_b_to_c)
        parsed.append(
            RawSensorCalibration(
                sensor_index=idx + 1,
                focal_px=float(lens_values[0]),
                k1=float(lens_values[1]),
                k2=float(lens_values[2]),
                k3=float(lens_values[3]),
                center_x=float(center_values[0]),
                center_y=float(center_values[1]),
                tm_b_to_c=tm_b_to_c,
                tm_c_to_b=tm_c_to_b,
            )
        )
    storage.release()
    if len(parsed) < 4:
        raise ValueError("Calibration must include 4 camera entries.")
    return parsed


def _parse_raw_gps(path: Path) -> list[RawGpsSample]:
    """Parse raw GNSS stream from ``*-gps.txt``."""
    rows: list[RawGpsSample] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            sec_raw = (row.get("systemtime_sec") or "").strip()
            if not sec_raw:
                continue
            usec_raw = (row.get("systemtime_usec") or "0").strip()
            try:
                timestamp = float(sec_raw) + (float(usec_raw) / 1_000_000.0)
                lat = float(row["lat"]) / 10_000_000.0
                lon = float(row["lon"]) / 10_000_000.0
                altitude = float(row["height"]) / 1000.0
            except (KeyError, ValueError):
                continue

            heading = _parse_heading_deg(row)
            rows.append(
                RawGpsSample(
                    timestamp_sec=timestamp,
                    latitude=lat,
                    longitude=lon,
                    altitude_m=altitude,
                    heading_deg=heading,
                    horizontal_accuracy_m=_optional_scaled_float(row.get("hAcc"), 1000.0),
                    vertical_accuracy_m=_optional_scaled_float(row.get("vAcc"), 1000.0),
                    heading_accuracy_deg=_optional_scaled_float(row.get("headAcc"), 100000.0),
                )
            )
    if not rows:
        raise ValueError(f"No valid GNSS rows parsed from {path}")
    rows.sort(key=lambda item: item.timestamp_sec)
    return rows


def _parse_heading_deg(row: dict[str, str]) -> float:
    """Resolve heading in degrees from raw GNSS row."""
    for key in ("headVeh", "headMot"):
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw) / 100000.0
        except ValueError:
            continue
        if math.isfinite(value):
            return value % 360.0
    return 0.0


def _parse_sensor_timestamps(path: Path) -> list[int]:
    """Parse DTS nanosecond timestamps from ``*-timestamps-*.txt``."""
    values: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("DTS"):
                continue
            token = stripped.split()[0]
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no} has invalid timestamp '{token}'") from exc
    if not values:
        raise ValueError(f"No timestamps parsed from {path}")
    return values


def _nearest_gps_sample(samples: tuple[RawGpsSample, ...], timestamp_sec: float) -> RawGpsSample:
    """Return nearest GNSS sample by timestamp."""
    if not samples:
        raise ValueError("No GNSS samples available for raw observation construction.")
    times = [sample.timestamp_sec for sample in samples]
    idx = bisect.bisect_left(times, timestamp_sec)
    candidates: list[RawGpsSample] = []
    if idx < len(samples):
        candidates.append(samples[idx])
    if idx > 0:
        candidates.append(samples[idx - 1])
    if not candidates:
        return samples[0]
    return min(candidates, key=lambda item: abs(item.timestamp_sec - timestamp_sec))


def _fisheye_direction_from_pixel(
    pixel_u: float,
    pixel_v: float,
    calibration: RawSensorCalibration,
) -> np.ndarray:
    """Convert fisheye pixel coordinate into a unit camera-frame ray."""
    dx = float(pixel_u) - calibration.center_x
    dy = float(pixel_v) - calibration.center_y
    radius_distorted = math.hypot(dx, dy)
    if radius_distorted <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    theta = _invert_fisheye_radius(
        radius_distorted,
        focal_px=max(1e-6, calibration.focal_px),
        k1=calibration.k1,
        k2=calibration.k2,
        k3=calibration.k3,
    )
    scale = math.sin(theta) / radius_distorted
    direction = np.array(
        [
            dx * scale,
            dy * scale,
            math.cos(theta),
        ],
        dtype=np.float64,
    )
    direction /= max(np.linalg.norm(direction), 1e-12)
    return direction


def _invert_fisheye_radius(
    radius_distorted: float,
    *,
    focal_px: float,
    k1: float,
    k2: float,
    k3: float,
    max_iters: int = 8,
) -> float:
    """Invert polynomial fisheye radius model using Newton iterations."""
    theta = max(0.0, radius_distorted / focal_px)
    for _ in range(max_iters):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        distortion = 1.0 + (k1 * theta2) + (k2 * theta4) + (k3 * theta6)
        model = focal_px * theta * distortion
        error = model - radius_distorted
        if abs(error) <= 1e-9:
            break
        derivative = focal_px * (
            distortion
            + theta * ((2.0 * k1 * theta) + (4.0 * k2 * theta**3) + (6.0 * k3 * theta**5))
        )
        if abs(derivative) <= 1e-12:
            break
        theta -= error / derivative
        theta = max(0.0, theta)
    return theta


def _body_direction_to_enu(direction_body: np.ndarray, heading_deg: float) -> np.ndarray:
    """Rotate body-frame direction into ENU using heading only.

    Assumes body frame convention:
    - +X forward
    - +Y right
    - +Z up
    """
    bearing = math.radians(heading_deg % 360.0)
    x_fwd, y_right, z_up = [float(value) for value in direction_body.reshape(3)]
    east = (x_fwd * math.sin(bearing)) + (y_right * math.cos(bearing))
    north = (x_fwd * math.cos(bearing)) - (y_right * math.sin(bearing))
    up = z_up
    direction = np.array([east, north, up], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-12)
    return direction


def _find_single(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Expected file matching {pattern} in {directory}")
    if len(candidates) > 1:
        raise ValueError(f"Expected one file matching {pattern} in {directory}, found {len(candidates)}")
    return candidates[0]


def _find_optional(directory: Path, pattern: str) -> Optional[Path]:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        return None
    return candidates[0]


def _optional_scaled_float(raw_value: Optional[str], scale: float) -> Optional[float]:
    """Parse optional numeric text and divide by ``scale``."""
    raw = (raw_value or "").strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed / scale
