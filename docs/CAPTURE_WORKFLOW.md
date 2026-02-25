# Capture Workflow

## Scope

This document defines the supported stitched-capture ingestion path in Byakugan and the
engineering constraints required for reliable geospatial measurements.

The implemented workflow targets **stitched exports** that contain:

- `Output/*.jpg` stitched equirectangular frames
- `Output/*_framepos.txt` per-frame pose map
- optional `Output/*-gps.txt` raw GNSS stream
- optional `Output/*-imu.csv` raw IMU stream

Raw unstitched sensor-video workflows remain a future phase.

## Input Contract

Byakugan accepts either:

- capture root directory containing an `Output` subdirectory
- the `Output` directory directly

`*_framepos.txt` must include these required columns:

- `systemtime_sec`
- `frame_index`
- `lat`
- `lon`
- `altitude`
- `distance`
- `heading`
- `pitch`
- `roll`
- `track`
- `jpeg_filename`

The parser validates that:

- all required columns exist
- every `jpeg_filename` exists on disk
- `frame_index` values are contiguous and zero-based

## Runtime Behavior

When a stitched capture is loaded:

1. Sequence metadata is parsed and validated.
2. Frame navigation controls are enabled.
3. Selecting a frame loads the corresponding panorama.
4. Camera pose fields are auto-filled from `framepos` (`lat/lon/alt/heading`).
5. Existing depth state is reset because each frame is a fresh panorama context.
6. In **Two-Frame Triangulation** mode, operators can click the same feature in
   two different frames to recover metric 3D coordinates without a depth map.
7. The UI selects this mode automatically for capture folders; no manual mode
   selection is required.

Frame `pitch` and `roll` are now applied in measurement geometry (depth, ground
plane, and two-frame triangulation rays) to reduce systematic orientation error.

When `*-gps.txt` is present, Byakugan timestamp-aligns GNSS records to frame
timestamps and enriches each frame with `hAcc`, `vAcc`, `headAcc`, and `hMSL`.
Two-frame triangulation then uses this metadata for quality-aware output
messages and uncertainty estimation.

## Engineering Notes

- Parsing is strict by design; partial or ambiguous capture folders are rejected.
- The loader is implemented in `src/byakugan_app/io/capture_sequence.py`.
- Parser behavior is covered by `tests/test_capture_sequence.py`.
- Existing geometry and depth tests should remain green to preserve baseline math
  correctness while sequence support evolves.

## Next Milestones

1. Add sequence-aware triangulation mode with persistent tracklets
   (multi-frame feature intersection, not only pairwise).
2. Add raw 4-sensor ingestion path from calibration + timestamps + videos.
