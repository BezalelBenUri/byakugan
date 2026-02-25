"""Generic stitched-capture discovery and parsing utilities.

This module targets exports that already contain stitched equirectangular
frames and a ``*_framepos.txt`` pose map. The parser is intentionally strict:
it validates required columns and frame references so downstream measurement
logic can rely on a complete, deterministic frame-to-pose mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import json
import math
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger


@dataclass(slots=True, frozen=True)
class CaptureFramePose:
    """Pose and file metadata for one stitched panorama frame.

    Attributes
    ----------
    frame_index:
        Zero-based frame index from ``*_framepos.txt``.
    timestamp_sec:
        Capture timestamp in UNIX seconds.
    latitude, longitude, altitude_m:
        Geodetic pose for the frame. Values are read directly from the export.
    heading_deg, pitch_deg, roll_deg:
        Orientation angles from the export in degrees.
    track_deg:
        Optional course-over-ground style heading from the export.
    distance_m:
        Cumulative route distance in metres, if present.
    horizontal_accuracy_m, vertical_accuracy_m:
        Optional GNSS accuracy from ``*-gps.txt`` (metres) matched by timestamp.
    heading_accuracy_deg:
        Optional heading accuracy (degrees) from ``*-gps.txt``.
    altitude_msl_m:
        Optional MSL altitude from ``*-gps.txt`` (metres).
    image_path:
        Absolute path to the stitched panorama frame image.
    """

    frame_index: int
    timestamp_sec: float
    latitude: float
    longitude: float
    altitude_m: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    track_deg: float
    distance_m: float
    image_path: Path
    horizontal_accuracy_m: Optional[float] = None
    vertical_accuracy_m: Optional[float] = None
    heading_accuracy_deg: Optional[float] = None
    altitude_msl_m: Optional[float] = None


@dataclass(slots=True, frozen=True)
class _GpsSample:
    """Single GNSS sample used to enrich stitched frame metadata."""

    timestamp_sec: float
    horizontal_accuracy_m: Optional[float]
    vertical_accuracy_m: Optional[float]
    heading_accuracy_deg: Optional[float]
    altitude_msl_m: Optional[float]


@dataclass(slots=True, frozen=True)
class StitchedCapture:
    """Representation of a stitched capture folder."""

    capture_root: Path
    output_dir: Path
    framepos_path: Path
    gps_path: Optional[Path]
    imu_path: Optional[Path]
    capture_id: Optional[str]
    frame_width: int
    frame_height: int
    frames: tuple[CaptureFramePose, ...]

    @property
    def frame_count(self) -> int:
        """Number of available stitched frames in the sequence."""
        return len(self.frames)


def discover_stitched_capture(path: Path) -> StitchedCapture:
    """Discover and parse a stitched capture export.

    Parameters
    ----------
    path:
        Either the capture root (containing ``Output``) or the ``Output``
        directory itself.

    Returns
    -------
    StitchedCapture
        Parsed capture object with validated frame pose records.

    Raises
    ------
    FileNotFoundError
        If required output assets are missing.
    ValueError
        If metadata exists but has invalid schema or inconsistent values.
    """

    resolved = path.expanduser().resolve()
    output_dir = _resolve_output_dir(resolved)
    framepos_path = _find_single(output_dir, "*_framepos.txt")
    gps_path = _find_optional(output_dir, "*-gps.txt")
    imu_path = _find_optional(output_dir, "*-imu.csv")

    frames = _parse_framepos(framepos_path, output_dir)
    gps_samples = _parse_gps_samples(gps_path) if gps_path is not None else []
    if gps_samples:
        frames = _attach_gps_metadata_to_frames(frames, gps_samples)
    if not frames:
        raise ValueError(f"No frame rows were parsed from {framepos_path}")

    first_image = cv2.imread(str(frames[0].image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Unable to read first stitched frame {frames[0].image_path}")
    frame_height, frame_width = first_image.shape[:2]

    capture_root = output_dir.parent
    capture_id = _read_capture_id(capture_root)

    logger.info(
        "Discovered stitched capture: root={}, frames={}, frame_size={}x{}",
        capture_root,
        len(frames),
        frame_width,
        frame_height,
    )

    return StitchedCapture(
        capture_root=capture_root,
        output_dir=output_dir,
        framepos_path=framepos_path,
        gps_path=gps_path,
        imu_path=imu_path,
        capture_id=capture_id,
        frame_width=frame_width,
        frame_height=frame_height,
        frames=tuple(frames),
    )


def _resolve_output_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Capture path does not exist: {path}")
    if path.is_dir() and path.name.lower() == "output":
        return path
    output_dir = path / "Output"
    if output_dir.is_dir():
        return output_dir
    raise FileNotFoundError(f"Could not locate output directory under {path}")


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


def _parse_framepos(framepos_path: Path, output_dir: Path) -> list[CaptureFramePose]:
    required_columns = {
        "systemtime_sec",
        "frame_index",
        "lat",
        "lon",
        "altitude",
        "distance",
        "heading",
        "pitch",
        "roll",
        "track",
        "jpeg_filename",
    }
    with framepos_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        header = set(reader.fieldnames or [])
        missing = sorted(required_columns - header)
        if missing:
            raise ValueError(f"{framepos_path.name} is missing required columns: {', '.join(missing)}")

        rows: list[CaptureFramePose] = []
        for line_no, row in enumerate(reader, start=2):
            image_name = (row.get("jpeg_filename") or "").strip()
            if not image_name:
                raise ValueError(f"{framepos_path.name}:{line_no} has empty jpeg_filename")
            image_path = output_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"{framepos_path.name}:{line_no} references missing image {image_path}")

            rows.append(
                CaptureFramePose(
                    frame_index=_to_int(row, "frame_index", framepos_path, line_no),
                    timestamp_sec=_to_float(row, "systemtime_sec", framepos_path, line_no),
                    latitude=_to_float(row, "lat", framepos_path, line_no),
                    longitude=_to_float(row, "lon", framepos_path, line_no),
                    altitude_m=_to_float(row, "altitude", framepos_path, line_no),
                    distance_m=_to_float(row, "distance", framepos_path, line_no),
                    heading_deg=_to_float(row, "heading", framepos_path, line_no),
                    pitch_deg=_to_float(row, "pitch", framepos_path, line_no),
                    roll_deg=_to_float(row, "roll", framepos_path, line_no),
                    track_deg=_to_float(row, "track", framepos_path, line_no),
                    image_path=image_path.resolve(),
                )
            )

    rows.sort(key=lambda item: item.frame_index)
    for expected_index, frame in enumerate(rows):
        if frame.frame_index != expected_index:
            raise ValueError(
                f"{framepos_path.name} has non-sequential frame_index values. "
                f"Expected {expected_index}, found {frame.frame_index}."
            )
    return rows


def _parse_gps_samples(gps_path: Path) -> list[_GpsSample]:
    """Parse optional GNSS stream exported by the capture workflow.

    The format mirrors UBX NAV-PVT style units:
    - ``hAcc``/``vAcc`` in millimetres
    - ``headAcc`` in 1e-5 degrees
    - ``hMSL`` in millimetres
    """
    samples: list[_GpsSample] = []
    try:
        with gps_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            header = set(reader.fieldnames or [])
            if not {"systemtime_sec", "systemtime_usec"} <= header:
                logger.warning("GPS stream missing timestamp columns: {}", gps_path)
                return []

            for row in reader:
                timestamp = _timestamp_from_row(row)
                if timestamp is None:
                    continue
                samples.append(
                    _GpsSample(
                        timestamp_sec=timestamp,
                        horizontal_accuracy_m=_optional_scaled_float(row.get("hAcc"), 1000.0),
                        vertical_accuracy_m=_optional_scaled_float(row.get("vAcc"), 1000.0),
                        heading_accuracy_deg=_optional_scaled_float(row.get("headAcc"), 100000.0),
                        altitude_msl_m=_optional_scaled_float(row.get("hMSL"), 1000.0),
                    )
                )
    except OSError:
        logger.warning("Unable to read GPS stream {}", gps_path)
        return []

    samples.sort(key=lambda item: item.timestamp_sec)
    return samples


def _attach_gps_metadata_to_frames(
    frames: list[CaptureFramePose],
    gps_samples: list[_GpsSample],
    *,
    max_time_delta_s: float = 1.0,
) -> list[CaptureFramePose]:
    """Attach nearest GNSS accuracy/altitude samples to frame pose rows."""
    if not gps_samples:
        return frames

    gps_times = [sample.timestamp_sec for sample in gps_samples]
    enriched: list[CaptureFramePose] = []

    import bisect

    for frame in frames:
        insert_at = bisect.bisect_left(gps_times, frame.timestamp_sec)
        candidates: list[_GpsSample] = []
        if 0 <= insert_at < len(gps_samples):
            candidates.append(gps_samples[insert_at])
        if insert_at > 0:
            candidates.append(gps_samples[insert_at - 1])
        if not candidates:
            enriched.append(frame)
            continue

        nearest = min(candidates, key=lambda item: abs(item.timestamp_sec - frame.timestamp_sec))
        if abs(nearest.timestamp_sec - frame.timestamp_sec) > max_time_delta_s:
            enriched.append(frame)
            continue

        enriched.append(
            replace(
                frame,
                horizontal_accuracy_m=nearest.horizontal_accuracy_m,
                vertical_accuracy_m=nearest.vertical_accuracy_m,
                heading_accuracy_deg=nearest.heading_accuracy_deg,
                altitude_msl_m=nearest.altitude_msl_m,
            )
        )
    return enriched


def _to_float(row: dict[str, str], key: str, source: Path, line_no: int) -> float:
    raw = (row.get(key) or "").strip()
    if not raw:
        raise ValueError(f"{source.name}:{line_no} has empty value for {key}")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{source.name}:{line_no} has invalid float for {key}: {raw}") from exc


def _optional_scaled_float(raw_value: Optional[str], scale: float) -> Optional[float]:
    """Parse optional numeric text and divide by `scale`.

    Returns ``None`` for empty, non-finite, non-positive, or invalid values.
    """
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


def _timestamp_from_row(row: dict[str, str]) -> Optional[float]:
    """Return GNSS timestamp in seconds, or ``None`` when unavailable."""
    sec_raw = (row.get("systemtime_sec") or "").strip()
    usec_raw = (row.get("systemtime_usec") or "").strip()
    if not sec_raw:
        return None
    try:
        sec = float(sec_raw)
        usec = float(usec_raw) if usec_raw else 0.0
    except ValueError:
        return None
    if not math.isfinite(sec) or not math.isfinite(usec):
        return None
    return sec + (usec / 1_000_000.0)


def _to_int(row: dict[str, str], key: str, source: Path, line_no: int) -> int:
    raw = (row.get(key) or "").strip()
    if not raw:
        raise ValueError(f"{source.name}:{line_no} has empty value for {key}")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{source.name}:{line_no} has invalid integer for {key}: {raw}") from exc


def _read_capture_id(capture_root: Path) -> Optional[str]:
    info_files = sorted(capture_root.glob("*-info.txt"))
    if not info_files:
        return None
    info_file = info_files[0]
    try:
        payload = json.loads(info_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Unable to parse capture metadata file {}", info_file)
        return None
    capture_id = payload.get("captureId")
    return str(capture_id) if capture_id is not None else None
