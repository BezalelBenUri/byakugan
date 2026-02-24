from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from byakugan_app.io.nctech_capture import discover_stitched_capture


def _write_test_frame(path: Path, width: int = 320, height: int = 160) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 127
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to create test frame at {path}")


def _write_framepos(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_discover_stitched_capture_from_root(tmp_path: Path):
    root = tmp_path / "02146-1234567890"
    output = root / "Output"
    output.mkdir(parents=True)

    _write_test_frame(output / "0000000000.jpg")
    _write_test_frame(output / "0000000001.jpg")

    _write_framepos(
        output / "02146-1234567890_framepos.txt",
        [
            {
                "systemtime_sec": "123.0",
                "frame_index": "0",
                "lat": "6.0",
                "lon": "3.0",
                "altitude": "42.0",
                "distance": "0.0",
                "heading": "15.0",
                "pitch": "-1.0",
                "roll": "2.0",
                "track": "15.0",
                "jpeg_filename": "0000000000.jpg",
            },
            {
                "systemtime_sec": "124.0",
                "frame_index": "1",
                "lat": "6.1",
                "lon": "3.1",
                "altitude": "43.0",
                "distance": "4.0",
                "heading": "16.0",
                "pitch": "-0.8",
                "roll": "2.1",
                "track": "16.0",
                "jpeg_filename": "0000000001.jpg",
            },
        ],
    )

    (output / "02146-1234567890-gps.txt").write_text("header\n", encoding="utf-8")
    (output / "02146-1234567890-imu.csv").write_text("header\n", encoding="utf-8")
    (root / "02146-1234567890-info.txt").write_text(
        json.dumps({"captureId": "1234567890"}), encoding="utf-8"
    )

    capture = discover_stitched_capture(root)
    assert capture.capture_id == "1234567890"
    assert capture.frame_count == 2
    assert capture.frame_width == 320
    assert capture.frame_height == 160
    assert capture.frames[0].frame_index == 0
    assert capture.frames[1].heading_deg == pytest.approx(16.0)
    assert capture.gps_path is not None
    assert capture.imu_path is not None


def test_discover_stitched_capture_from_output_dir(tmp_path: Path):
    output = tmp_path / "Output"
    output.mkdir(parents=True)
    _write_test_frame(output / "0000000000.jpg", width=64, height=32)
    _write_framepos(
        output / "capture_framepos.txt",
        [
            {
                "systemtime_sec": "100.0",
                "frame_index": "0",
                "lat": "0.0",
                "lon": "0.0",
                "altitude": "1.0",
                "distance": "0.0",
                "heading": "0.0",
                "pitch": "0.0",
                "roll": "0.0",
                "track": "0.0",
                "jpeg_filename": "0000000000.jpg",
            }
        ],
    )

    capture = discover_stitched_capture(output)
    assert capture.output_dir == output.resolve()
    assert capture.frame_count == 1
    assert capture.frame_width == 64
    assert capture.frame_height == 32


def test_discover_stitched_capture_validates_schema(tmp_path: Path):
    output = tmp_path / "Output"
    output.mkdir(parents=True)
    _write_test_frame(output / "0000000000.jpg")
    bad_framepos = output / "broken_framepos.txt"
    bad_framepos.write_text(
        "systemtime_sec,frame_index,lat,lon,altitude,distance,heading,pitch,track,jpeg_filename\n"
        "100.0,0,0,0,0,0,0,0,0,0000000000.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        discover_stitched_capture(output)


def test_discover_stitched_capture_rejects_missing_frame(tmp_path: Path):
    output = tmp_path / "Output"
    output.mkdir(parents=True)
    _write_framepos(
        output / "missing_framepos.txt",
        [
            {
                "systemtime_sec": "100.0",
                "frame_index": "0",
                "lat": "0.0",
                "lon": "0.0",
                "altitude": "1.0",
                "distance": "0.0",
                "heading": "0.0",
                "pitch": "0.0",
                "roll": "0.0",
                "track": "0.0",
                "jpeg_filename": "0000000000.jpg",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="references missing image"):
        discover_stitched_capture(output)
