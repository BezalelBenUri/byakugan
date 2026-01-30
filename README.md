# Byakugan Panorama Geospatial Viewer

Byakugan is a native Python desktop application that lets Insta360 Pro2 users explore high-resolution equirectangular panoramas, pick pixels in an interactive 3D viewer, and recover accurate WGS84 coordinates for real-world objects. It is built with PyQt6 and PyOpenGL, runs fully offline, and can be packaged as a cross-platform executable via PyInstaller.

## Key Features
- GPU-accelerated panorama viewer with smooth pan, tilt, zoom, and keyboard-assisted navigation.
- Auto-detects stereo 360 formats (top-bottom or side-by-side) and offers one-click depth reconstruction.
- Bearing-aware initial orientation and reset view controls aligned with the camera compass bearing.
- Crosshair-based point picking that reports pixel indices, depth, ENU vectors, and lat/lon/alt in real time.
- Depth map ingestion (grayscale or float) plus optional stereo reconstruction using OpenCV's `StereoSGBM`.
- Configurable camera pose inputs (lat, lon, altitude, compass bearing) and precise ENU to ECEF conversions via `pyproj`, applied to every measurement in real time.
- Measurement history with CSV/JSON export for GIS pipelines and a depth-source readout for traceability.
- Dark, high-DPI-aware UI inspired by Fluent design.

## Getting Started

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
. .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -e .[dev]
```

### 3. Launch the app
```bash
python -m byakugan_app
```

### 4. Load your data
1. Click **Load Panorama...** and choose a stitched equirectangular RGB image (e.g., 7680x3840 or 7680x7680 for stereo top-bottom).
2. Provide the capture latitude, longitude, altitude (meters), and bearing (degrees clockwise from north).
3. If a stereo panorama is detected, Byakugan offers to split the image automatically and generate depth. You can override the detected format and optical parameters in the **Generate Depth Map...** dialog.
4. Alternatively, load a pre-computed depth map or supply external left/right images for stereo matching.
5. Click within the viewer to record measurements; results show in the sidebar and measurement table.
6. Use **Reset View (Front)** to re-center on the camera bearing whenever you update the pose.

A warning banner appears while the default pose (0,0,0) is active; update the spin boxes for accurate absolute coordinates.


## Viewer Controls
- Drag with the left mouse button to pan horizontally (yaw) and vertically (pitch).
- Scroll the mouse wheel to zoom the field of view; use the arrow or WASD keys for incremental panning.
- Press `R` (or click **Reset View (Front)**) to re-center on the camera front using the current bearing.
- The pose context panel reports the latitude, longitude, altitude, and bearing applied to each measurement.

## Stereo Depth from Panoramas
- On load, top-bottom (over-under) and side-by-side panoramas are detected automatically.
- The depth generation dialog lets you tweak baseline, focal length, rectification, downsampling, and optional export of the generated depth map.
- Depth maps produced from the panorama are stored in-memory, labelled as stereo sources, and immediately available for coordinate probing.

## Packaging an Executable

PyInstaller configuration is provided in `byakugan.spec`.

```bash
pyinstaller byakugan.spec
```

The resulting binary is written to `dist/byakugan/byakugan.exe` (Windows) or `dist/byakugan` (macOS/Linux). For best results ensure that OpenGL drivers are present on the target machine.

## Testing

The project ships with focused unit tests for the spherical geometry, geodesy math, and stereo panorama handling.

```bash
pytest
```

## Project Structure

```
src/byakugan_app/
    __main__.py         # Module entry point
    main.py             # QApplication bootstrapper
    logging.py          # Loguru configuration
    io/depth_utils.py   # Stereo detection and depth-from-panorama helpers
    math/               # Spherical + geodetic helpers
    viewer/             # OpenGL panorama widget
    ui/                 # MainWindow, dialogs, themed widgets
    io/loader.py        # Image/depth loading and stereo reconstruction
    workers/            # Thread-pool task helpers
    models/             # Dataclasses for pose and selections
```

## Depth Map Expectations
- Depth units: meters (float32 preferred). UInt16 maps are interpreted as millimetres by default.
- Values of zero or `NaN` are treated as invalid during selection.
- Stereo reconstruction expects rectified inputs; the panorama workflow assumes Insta360-style stitching with minimal parallax.

## Export Formats
- **CSV**: Pixel coordinates, depth, ENU vector components, and LLH.
- **JSON**: Array of measurement objects mirroring the CSV schema.

## Roadmap Ideas
- Integrate IMU pitch/roll compensation.
- Add VR-style navigation and measurement tools (rulers, area estimates).
- Expose configurable sampling (bilinear depth lookup, supersampling for sub-pixel accuracy).

## License
MIT License (pending confirmation).
