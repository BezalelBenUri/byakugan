"""NCTech iSTAR/iPulse stitched-capture discovery and parsing utilities.

This module targets exports that already contain stitched equirectangular
frames and a ``*_framepos.txt`` pose map. The parser is intentionally strict:
it validates required columns and frame references so downstream measurement
logic can rely on a complete, deterministic frame-to-pose mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger


@dataclass(slots=True, frozen=True)
class NCTechFramePose:
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


@dataclass(slots=True, frozen=True)
class NCTechStitchedCapture:
    """Representation of a stitched NCTech capture folder."""

    capture_root: Path
    output_dir: Path
    framepos_path: Path
    gps_path: Optional[Path]
    imu_path: Optional[Path]
    capture_id: Optional[str]
    frame_width: int
    frame_height: int
    frames: tuple[NCTechFramePose, ...]

    @property
    def frame_count(self) -> int:
        """Number of available stitched frames in the sequence."""
        return len(self.frames)


def discover_stitched_capture(path: Path) -> NCTechStitchedCapture:
    """Discover and parse a stitched NCTech export.

    Parameters
    ----------
    path:
        Either the capture root (containing ``Output``) or the ``Output``
        directory itself.

    Returns
    -------
    NCTechStitchedCapture
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
    if not frames:
        raise ValueError(f"No frame rows were parsed from {framepos_path}")

    first_image = cv2.imread(str(frames[0].image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Unable to read first stitched frame {frames[0].image_path}")
    frame_height, frame_width = first_image.shape[:2]

    capture_root = output_dir.parent
    capture_id = _read_capture_id(capture_root)

    logger.info(
        "Discovered NCTech stitched capture: root={}, frames={}, frame_size={}x{}",
        capture_root,
        len(frames),
        frame_width,
        frame_height,
    )

    return NCTechStitchedCapture(
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
    raise FileNotFoundError(f"Could not locate NCTech output directory under {path}")


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


def _parse_framepos(framepos_path: Path, output_dir: Path) -> list[NCTechFramePose]:
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

        rows: list[NCTechFramePose] = []
        for line_no, row in enumerate(reader, start=2):
            image_name = (row.get("jpeg_filename") or "").strip()
            if not image_name:
                raise ValueError(f"{framepos_path.name}:{line_no} has empty jpeg_filename")
            image_path = output_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"{framepos_path.name}:{line_no} references missing image {image_path}")

            rows.append(
                NCTechFramePose(
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


def _to_float(row: dict[str, str], key: str, source: Path, line_no: int) -> float:
    raw = (row.get(key) or "").strip()
    if not raw:
        raise ValueError(f"{source.name}:{line_no} has empty value for {key}")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{source.name}:{line_no} has invalid float for {key}: {raw}") from exc


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
