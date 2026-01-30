# Architecture Overview

Byakugan is organised into focused packages so that math, rendering, IO, and UI responsibilities stay isolated.

## High-Level Flow

1. **Bootstrap**: `byakugan_app.main` configures logging, high-DPI, theme, and instantiates `MainWindow`.
2. **State**: `PanoramaState` tracks the loaded panorama (raw + viewer image), depth map, camera pose metadata, and stereo detection metadata.
3. **Stereo Detection**: `io.depth_utils.detect_stereo_format` inspects panorama dimensions to classify mono versus top-bottom/side-by-side stereo; the raw image is split automatically when stereo is present.
4. **Rendering**: `PanoramaWidget` (PyQt6 + PyOpenGL) renders the equirectangular panorama mapped to the inside of a textured sphere, aligns the initial view with the camera bearing, and exposes full yaw/pitch navigation (mouse drag, wheel zoom, keyboard shortcuts).
5. **Computation**: `math.geometry` turns pixels into spherical angles and ENU vectors; `math.geodesy` converts ENU to geodetic coordinates with `pyproj`. Camera pose values (lat, lon, altitude, bearing) are applied immediately to every measurement, and stereo depth is generated through `depth_utils.compute_depth_from_panorama_array` or the existing external stereo pipeline.
6. **Concurrency**: Heavy IO (image/depth loading, stereo reconstruction) runs in background `FunctionTask`s executed by the global `QThreadPool`.
7. **Presentation**: `MainWindow` coordinates UI panels, updates measurement readouts, and persists a selectable history via `MeasurementTableModel`. The depth generation dialog surfaces stereo options, overrides, and export paths.

## Key Modules

| Module | Responsibility |
| ------ | -------------- |
| `byakugan_app/io/depth_utils.py` | Stereo format detection, panorama splitting, and depth computation via SGBM |
| `byakugan_app/viewer/panorama_widget.py` | OpenGL sphere rendering, camera navigation, and picking signals |
| `byakugan_app/ui/main_window.py` | UI composition, signal wiring, data validation, exports, and stereo prompts |
| `byakugan_app/ui/depth_generation_dialog.py` | Configurable dialog for generating depth from panoramas or external stereo pairs |
| `byakugan_app/math/geometry.py` | Pixel <-> spherical conversion, ENU vector math |
| `byakugan_app/math/geodesy.py` | ENU/ECEF/LLA transforms using WGS84 |
| `byakugan_app/io/loader.py` | Panorama/depth loading and legacy stereo reconstruction helpers |
| `byakugan_app/workers/task_runner.py` | Thread pool task wrapper for background jobs |
| `byakugan_app/models/*` | Dataclasses for pose, metadata, and measurement records |

## Threading Model

- UI runs on the main Qt thread.
- IO and stereo reconstruction functions are submitted to the global `QThreadPool` via `FunctionTask`.
- Results are marshalled back onto the UI thread through Qt signals.

## Coordinate Pipeline

```
Pixel (u, v)
  -> spherical angles (theta, phi)
  -> unit vector
  -> scale by depth -> local XYZ
  -> rotate by bearing -> ENU
  -> geodesy.enu_to_geodetic -> latitude, longitude, altitude
```

Coordinate calculations are unit tested (see `tests/test_geometry.py`). Stereo panorama detection and splitting are verified in `tests/test_depth_utils.py`.

## Packaging

- `pyproject.toml` defines runtime and dev dependencies.
- `byakugan.spec` produces a single-windowed executable using PyInstaller.

## Extensibility Notes

- Introduce additional sensors by extending `CameraPose` and augmenting the ENU rotation matrices.
- Additional export formats can be added by extending `MeasurementTableModel` consumers.
- Advanced depth filtering or smoothing can live inside `io/depth_utils.py` or a future `processing` package.
- The depth generation dialog is designed for additional modes (e.g., monocular depth estimation) without altering existing workflows.
