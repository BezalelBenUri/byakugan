from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from byakugan_app.io.raw_capture import build_raw_observation, discover_raw_capture


def _write_calibration(path: Path) -> None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.startWriteStruct("cameras", cv2.FileNode_SEQ)
    for idx in range(4):
        storage.startWriteStruct("", cv2.FileNode_MAP)
        storage.write("lensCoeffs", np.array([[1700.0], [-0.05], [0.0], [0.0]], dtype=np.float64))
        storage.write("centre", np.array([[1500.0], [1000.0]], dtype=np.float64))
        yaw = (idx * np.pi) / 2.0
        rot = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        tm = np.eye(4, dtype=np.float64)
        tm[:3, :3] = rot
        storage.write("tmBToC", tm)
        storage.endWriteStruct()
    storage.endWriteStruct()
    storage.release()


def _write_gps(path: Path) -> None:
    fieldnames = [
        "systemtime_sec",
        "systemtime_usec",
        "lon",
        "lat",
        "height",
        "hAcc",
        "vAcc",
        "headAcc",
        "headVeh",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "systemtime_sec": "1770739062",
                "systemtime_usec": "900000",
                "lon": "33649698",
                "lat": "66208705",
                "height": "50000",
                "hAcc": "1200",
                "vAcc": "1500",
                "headAcc": "250000",
                "headVeh": "12321332",
            }
        )


def _write_timestamps(path: Path) -> None:
    path.write_text(
        "DTS,STS,PATTERN,SYSTIME\n"
        "1770739062900000000 0 0 0\n"
        "1770739063000000000 0 0 0\n",
        encoding="utf-8",
    )


def test_discover_raw_capture_and_build_observation(tmp_path: Path):
    root = tmp_path / "02168-raw"
    root.mkdir(parents=True)

    _write_calibration(root / "02168-test-calibration.yaml")
    _write_gps(root / "02168-test-gps.txt")
    (root / "02168-test-imu.csv").write_text("ts,ax,ay,az\n", encoding="utf-8")

    for idx in range(1, 5):
        (root / f"02168-test-sensor-{idx}.mkv").write_bytes(b"")
        _write_timestamps(root / f"02168-test-timestamps-{idx}.txt")

    capture = discover_raw_capture(root)
    assert len(capture.calibrations) == 4
    assert len(capture.gps_samples) == 1
    assert len(capture.sensor_timestamps_ns[1]) == 2

    observation = build_raw_observation(
        capture,
        sensor_index=1,
        frame_index=0,
        pixel_u=1500.0,
        pixel_v=1000.0,
    )
    assert observation.origin_ecef.shape == (3,)
    assert observation.direction_ecef.shape == (3,)
    assert np.isfinite(observation.direction_ecef).all()
    assert abs(float(np.linalg.norm(observation.direction_ecef)) - 1.0) < 1e-6
