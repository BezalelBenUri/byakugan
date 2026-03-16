"""Main application window."""



from __future__ import annotations







import csv



import json



import math
from dataclasses import dataclass



from pathlib import Path



from typing import Optional







import cv2



import numpy as np
from scipy.signal import savgol_filter



from loguru import logger



from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QGuiApplication, QImage, QKeySequence, QPixmap, QShortcut



from PyQt6.QtWidgets import (



    QAbstractItemView,






    QFileDialog,



    QFormLayout,



    QGroupBox,



    QHBoxLayout,



    QCheckBox,




    QComboBox,



    QLabel,



    QLineEdit,



    QMainWindow,
    QDialog,



    QDialogButtonBox,



    QMessageBox,



    QPushButton,



    QSlider,



    QSpinBox,



    QSplitter,



    QStatusBar,



    QTableView,



    QVBoxLayout,



    QWidget,



    QDoubleSpinBox,
    QSpinBox,



)







from ..io.depth_utils import (



    StereoDetectionResult,



    StereoFormat,



    compute_depth_from_panorama_array,



    estimate_default_focal,



)



from ..io.loader import StereoDepthConfig, compute_depth_from_stereo, load_depth_map, load_equirectangular_image
from ..io.capture_sequence import StitchedCapture, discover_stitched_capture
from ..io.raw_capture import RawCapture, discover_raw_capture




from ..io.panorama_processing import FisheyeConversionParams, PanoramaInputFormat
from ..math import geodesy, geometry, multiview



from ..models.camera_pose import CameraPose



from ..models.panorama_state import PanoramaState



from ..models.selection import PixelSelection, PointMeasurement



from ..viewer.panorama_widget import PanoramaWidget



from ..workers.task_runner import FunctionTask, TaskRunner



from .depth_generation_dialog import DepthGenerationDialog, DepthGenerationMode



from .measurement_table_model import MeasurementTableModel


@dataclass(slots=True)
class TriangulationAnchor:
    """Stores the first ray/pose used for triangulation."""

    frame_index: int
    pose: CameraPose
    pixel_u: int
    pixel_v: int
    theta: float
    phi: float
    pitch_deg: float
    roll_deg: float


@dataclass(slots=True)
class TriangulationObservation:
    """One frame-level ray observation used for multi-view triangulation."""

    frame_index: int
    pixel_u: int
    pixel_v: int
    theta: float
    phi: float
    origin_ecef: np.ndarray
    direction_ecef: np.ndarray
    horizontal_accuracy_m: Optional[float]
    heading_accuracy_deg: Optional[float]











class MainWindow(QMainWindow):



    """Primary window containing the viewer and controls."""

    MEASUREMENT_MODE_AUTO = "auto"
    MEASUREMENT_MODE_DEPTH = "depth_map"
    MEASUREMENT_MODE_GROUND = "ground_plane"
    MEASUREMENT_MODE_TRIANGULATION = "triangulation_2frame"
    MIN_TRIANGULATION_BASELINE_M = 0.50
    MIN_TRIANGULATION_ANGLE_DEG = 2.00
    MAX_TRIANGULATION_RESIDUAL_M = 2.50
    MAX_TRIANGULATION_RAY_ERROR_DEG = 1.25
    MAX_TRIANGULATION_SIGMA_M = 3.50
    HARD_MAX_TRIANGULATION_RAY_ERROR_DEG = 6.00
    HARD_MAX_TRIANGULATION_SIGMA_M = 20.00
    TRIANGULATION_RANSAC_THRESHOLD_M = 2.0
    TRIANGULATION_HUBER_SCALE_M = 1.0
    TRIANGULATION_MAX_TRACK_OBSERVATIONS = 6
    TRIANGULATION_TEMPLATE_PATCH_RADIUS_PX = 14
    TRIANGULATION_TEMPLATE_SEARCH_RADIUS_PX = 96
    TRIANGULATION_MIN_TEMPLATE_SCORE = 0.28
    TRIANGULATION_POSE_CHECK_WINDOW_RADIUS_PX = 180
    TRIANGULATION_MIN_POSE_INLIER_RATIO = 0.20
    TRIANGULATION_MAX_HEADING_ACC_DEG = 20.0
    MAX_GROUND_PLANE_RANGE_M = 120.0
    CAPTURE_SMOOTHING_MIN_FRAMES = 11
    CAPTURE_SMOOTHING_MAX_WINDOW = 41
    CAPTURE_HORIZONTAL_FLIP = True
    MIN_TRAJECTORY_BEARING_BASELINE_M = 1.5
    MIN_BEARING_CALIBRATION_BASELINE_M = 1.0
    MIN_BEARING_CALIBRATION_SAMPLES = 24
    MIN_BEARING_CALIBRATION_CONCENTRATION = 0.60







    def __init__(self) -> None:



        super().__init__()



        self.setWindowTitle("Byakugan Panorama Geospatial Viewer")



        self.resize(1540, 920)







        self._state = PanoramaState()



        self._task_runner = TaskRunner()



        self._active_tasks: set[FunctionTask] = set()







        self.viewer = PanoramaWidget()
        self.viewer.set_instruction_text("Drag left/right to rotate 360 deg yaw. Drag up/down to tilt pitch. Scroll to zoom. Press R or Reset View to recenter.")
        self.viewer.set_instruction_visible(True)
        self.measurement_model = MeasurementTableModel(self)
        self._needs_reorient = False
        self._format_combo_block = False
        self._syncing_render_view = False
        self._active_capture: Optional[StitchedCapture] = None
        self._active_raw_capture: Optional[RawCapture] = None
        self._capture_spin_block = False
        self._capture_base_roll_offset_deg = 0.0
        self._pose_pitch_deg = 0.0
        self._pose_roll_deg = 0.0
        self._manual_roll_correction_deg = 0.0
        self._measurement_mode = self.MEASUREMENT_MODE_AUTO
        self._triangulation_anchor: Optional[TriangulationAnchor] = None
        self._triangulation_track_observations: list[TriangulationObservation] = []
        self._measurement_history: list[PointMeasurement] = []
        self._measurement_marker_angles: list[tuple[float, float]] = []
        self._measurement_marker_frames: list[Optional[int]] = []
        self._capture_horizontal_flip_enabled = self.CAPTURE_HORIZONTAL_FLIP
        self._capture_bearing_offset_deg = 0.0
        self._capture_resolved_bearings_deg: list[float] = []
        self._capture_resolved_positions: list[tuple[float, float, float]] = []
        self._capture_gray_cache: dict[int, np.ndarray] = {}

        self._build_ui()



        self._create_menu_bar()



        self._connect_signals()



        logger.info("UI initialised")







    # ------------------------------------------------------------------



    def _build_ui(self) -> None:



        splitter = QSplitter(Qt.Orientation.Horizontal, self)



        splitter.setChildrenCollapsible(False)



        splitter.setHandleWidth(8)



        splitter.addWidget(self.viewer)



        splitter.addWidget(self._build_sidebar())



        splitter.setStretchFactor(0, 4)



        splitter.setStretchFactor(1, 2)







        self.setCentralWidget(splitter)



        status = QStatusBar()



        self.setStatusBar(status)







    def _build_sidebar(self) -> QWidget:



        sidebar = QWidget()



        layout = QVBoxLayout(sidebar)



        layout.setContentsMargins(16, 12, 16, 12)



        layout.setSpacing(12)







        camera_group = self._build_camera_group()
        data_group = self._build_data_group()
        self.selection_group = self._build_selection_group()
        measurements_group = self._build_measurements_group()

        self.selection_toggle_btn = QPushButton("Show Selection Readout")
        self.selection_toggle_btn.setCheckable(True)
        self.selection_toggle_btn.setChecked(False)
        self.selection_toggle_btn.clicked.connect(self._toggle_selection_readout)
        self.selection_group.setVisible(False)

        layout.addWidget(camera_group)



        layout.addWidget(data_group)






        layout.addWidget(self.selection_toggle_btn)
        layout.addWidget(self.selection_group)



        layout.addWidget(measurements_group, stretch=1)







        return sidebar







    # Camera -----------------------------------------------------------------




    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Camera Pose")
        form = QFormLayout(group)
        form.setVerticalSpacing(6)

        self.lat_field = QDoubleSpinBox()
        self.lat_field.setRange(-90.0, 90.0)
        self.lat_field.setDecimals(6)
        self.lat_field.setKeyboardTracking(False)
        self.lat_field.setValue(0.0)

        self.lon_field = QDoubleSpinBox()
        self.lon_field.setRange(-180.0, 180.0)
        self.lon_field.setDecimals(6)
        self.lon_field.setKeyboardTracking(False)
        self.lon_field.setValue(0.0)

        self.alt_field = QDoubleSpinBox()
        self.alt_field.setRange(-500.0, 10000.0)
        self.alt_field.setDecimals(2)
        self.alt_field.setSuffix(" m")
        self.alt_field.setKeyboardTracking(False)
        self.alt_field.setValue(0.0)

        self.bearing_field = QDoubleSpinBox()
        self.bearing_field.setRange(0.0, 360.0)
        self.bearing_field.setDecimals(2)
        self.bearing_field.setSuffix(" deg")
        self.bearing_field.setKeyboardTracking(False)
        self.bearing_field.setValue(0.0)

        form.addRow("Latitude", self.lat_field)
        form.addRow("Longitude", self.lon_field)
        form.addRow("Altitude", self.alt_field)
        form.addRow("Bearing", self.bearing_field)

        self.pose_warning_label = QLabel(
            "Default camera pose used - enter real values for accurate geospatial coordinates."
        )
        self.pose_warning_label.setWordWrap(True)
        self.pose_warning_label.setStyleSheet("color: #d28a00; font-size: 11px;")

        self.pose_warning_label.hide()
        form.addRow("", self.pose_warning_label)

        return group



    # Data loading -----------------------------------------------------------




    def _build_data_group(self) -> QGroupBox:
        group = QGroupBox("Data Sources")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)
        show_advanced_controls = False

        self.image_path_display = QLineEdit()
        self.image_path_display.setReadOnly(True)
        self.image_path_display.setPlaceholderText("No panorama loaded")

        self.depth_path_display = QLineEdit()
        self.depth_path_display.setReadOnly(True)
        self.depth_path_display.setPlaceholderText("No depth map loaded")

        self.stereo_status_label = QLabel("Format status: -")
        self.stereo_status_label.setWordWrap(True)
        self.stereo_status_label.setObjectName("stereoStatusLabel")
        self.stereo_status_label.setStyleSheet("color: #8aa; font-size: 11px;")

        self.navigation_hint_label = QLabel(
            "Drag left/right to rotate yaw (full 360). Drag up/down to tilt pitch. Scroll to zoom. Press R or Reset View to recenter."
        )
        self.navigation_hint_label.setWordWrap(True)
        self.navigation_hint_label.setStyleSheet("color: #8aa; font-size: 11px;")

        load_image_btn = QPushButton("Load Panorama...")
        load_capture_btn = QPushButton("Load Capture Folder...")
        load_depth_btn = QPushButton("Load Depth Map...")
        generate_depth_btn = QPushButton("Generate Depth Map...")
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.setEnabled(False)
        self.reset_view_button.setToolTip("Snap back to the panorama front and horizon.")

        self.format_combo = QComboBox()
        self.format_combo.setEnabled(False)
        self.format_combo.currentIndexChanged.connect(self._on_format_override_changed)

        self.distortion_button = QPushButton("Correct Distortion...")
        self.distortion_button.setEnabled(False)
        self.distortion_button.clicked.connect(self._on_distortion_correction_clicked)

        self.render_view_combo = QComboBox()
        self.render_view_combo.setEnabled(False)
        self.render_view_combo.currentIndexChanged.connect(self._on_render_view_changed)

        load_image_btn.clicked.connect(self._on_load_panorama_clicked)
        load_capture_btn.clicked.connect(self._on_load_capture_folder_clicked)
        load_depth_btn.clicked.connect(self._on_load_depth_clicked)
        generate_depth_btn.clicked.connect(self._on_generate_depth_clicked)

        self.capture_frame_spin = QSpinBox()
        self.capture_frame_spin.setRange(0, 0)
        self.capture_frame_spin.setEnabled(False)
        self.capture_frame_spin.valueChanged.connect(self._on_capture_frame_changed)
        self.capture_frame_total_label = QLabel("/ 0")
        self.detected_frames_label = QLabel("Detected Frames: 0")
        self.detected_frames_label.setStyleSheet("color: #8aa; font-size: 11px;")
        self.capture_status_label = QLabel("Capture sequence: -")
        self.capture_status_label.setWordWrap(True)
        self.capture_status_label.setStyleSheet("color: #8aa; font-size: 11px;")

        def _reset_view_handler():
            self._perform_reset_view()

        self._reset_view_handler = _reset_view_handler
        self.reset_view_button.clicked.connect(_reset_view_handler)

        vbox.addWidget(load_image_btn)
        vbox.addWidget(load_capture_btn)
        vbox.addWidget(self.image_path_display)
        frame_row = QHBoxLayout()
        frame_row.setSpacing(6)
        frame_row.addWidget(QLabel("Capture Frame:"))
        frame_row.addWidget(self.capture_frame_spin, 1)
        frame_row.addWidget(self.capture_frame_total_label)
        vbox.addLayout(frame_row)
        vbox.addWidget(self.detected_frames_label)
        vbox.addWidget(self.capture_status_label)
        vbox.addWidget(self.reset_view_button)

        if show_advanced_controls:
            vbox.addSpacing(4)
            vbox.addWidget(load_depth_btn)
            vbox.addWidget(self.depth_path_display)
            vbox.addSpacing(4)
            vbox.addWidget(generate_depth_btn)

            format_row = QHBoxLayout()
            format_row.setSpacing(6)
            format_row.addWidget(QLabel("Image Format:"))
            format_row.addWidget(self.format_combo, 1)
            vbox.addLayout(format_row)
            vbox.addWidget(self.distortion_button)

            render_row = QHBoxLayout()
            render_row.setSpacing(6)
            render_row.addWidget(QLabel("Render View:"))
            render_row.addWidget(self.render_view_combo, 1)
            vbox.addLayout(render_row)
        else:
            load_depth_btn.hide()
            generate_depth_btn.hide()
            self.depth_path_display.hide()
            self.format_combo.hide()
            self.distortion_button.hide()
            self.render_view_combo.hide()
            self.stereo_status_label.hide()

        vbox.addSpacing(6)
        vbox.addWidget(self.navigation_hint_label)
        if show_advanced_controls:
            vbox.addWidget(self.stereo_status_label)

        self._populate_format_combo()
        self._update_reset_button_state()
        self._update_render_view_controls()
        return group

    def _build_viewer_controls_group(self) -> QGroupBox:
        group = QGroupBox("Viewer Controls")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        self.orientation_overlay_checkbox = QCheckBox("Show Orientation Overlay")
        self.orientation_overlay_checkbox.setChecked(False)
        self.orientation_overlay_checkbox.toggled.connect(
            self.viewer.set_orientation_overlay_enabled
        )
        vbox.addWidget(self.orientation_overlay_checkbox)

        self.invert_x_checkbox = QCheckBox("Invert Horizontal Drag")
        self.invert_x_checkbox.toggled.connect(self._on_invert_x_toggled)
        vbox.addWidget(self.invert_x_checkbox)

        self.invert_y_checkbox = QCheckBox("Invert Vertical Drag")
        self.invert_y_checkbox.toggled.connect(self._on_invert_y_toggled)
        vbox.addWidget(self.invert_y_checkbox)

        self.anaglyph_checkbox = QCheckBox("Stereo Mode (Anaglyph)")
        self.anaglyph_checkbox.setEnabled(False)
        self.anaglyph_checkbox.toggled.connect(self._on_anaglyph_toggled)
        vbox.addWidget(self.anaglyph_checkbox)

        sensitivity_row = QHBoxLayout()
        sensitivity_row.setSpacing(6)
        sensitivity_label = QLabel("Mouse Sensitivity:")
        self.mouse_sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mouse_sensitivity_slider.setRange(1, 10)
        self.mouse_sensitivity_slider.setSingleStep(1)
        default_value = int(round(self.viewer.mouse_sensitivity() * 1000))
        default_value = min(max(default_value, 1), 10)
        self.mouse_sensitivity_slider.setValue(default_value)
        self.mouse_sensitivity_slider.valueChanged.connect(self._on_mouse_sensitivity_changed)
        self.mouse_sensitivity_value_label = QLabel(f"{self.viewer.mouse_sensitivity():.3f}")
        sensitivity_row.addWidget(sensitivity_label)
        sensitivity_row.addWidget(self.mouse_sensitivity_slider, 1)
        sensitivity_row.addWidget(self.mouse_sensitivity_value_label)
        vbox.addLayout(sensitivity_row)

        roll_row = QHBoxLayout()
        roll_row.setSpacing(6)
        roll_row.addWidget(QLabel("Roll Offset:"))
        self.roll_correction_spin = QDoubleSpinBox()
        self.roll_correction_spin.setRange(-180.0, 180.0)
        self.roll_correction_spin.setDecimals(1)
        self.roll_correction_spin.setSingleStep(1.0)
        self.roll_correction_spin.setSuffix(" deg")
        self.roll_correction_spin.setValue(0.0)
        self.roll_correction_spin.valueChanged.connect(self._on_roll_correction_changed)
        roll_row.addWidget(self.roll_correction_spin, 1)
        vbox.addLayout(roll_row)

        hint_label = QLabel("If orientation feels tilted, adjust Roll Offset; otherwise tune sensitivity or invert axes.")
        hint_label.setStyleSheet("color: #8aa; font-size: 11px;")
        hint_label.setWordWrap(True)
        vbox.addWidget(hint_label)

        return group



    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("Selection Readout")
        form = QFormLayout(group)

        self.measurement_mode_combo = QComboBox()
        self.measurement_mode_combo.addItem("Automatic", self.MEASUREMENT_MODE_AUTO)
        self.measurement_mode_combo.setCurrentIndex(0)
        self.measurement_mode_combo.setEnabled(False)
        self.measurement_mode_combo.setVisible(False)
        self.measurement_mode_combo.setToolTip(
            "Byakugan automatically selects the best available measurement engine."
        )
        self.measurement_mode_combo.currentIndexChanged.connect(self._on_measurement_mode_changed)

        self.ground_height_spin = QDoubleSpinBox()
        self.ground_height_spin.setRange(0.10, 10.00)
        self.ground_height_spin.setDecimals(2)
        self.ground_height_spin.setSingleStep(0.10)
        self.ground_height_spin.setSuffix(" m")
        self.ground_height_spin.setValue(1.70)
        self.ground_height_spin.setEnabled(False)
        self.ground_height_spin.valueChanged.connect(self._on_ground_height_changed)
        self.triangulation_label = QLabel("Select anchor point")
        self.triangulation_label.setWordWrap(True)
        self.triangulation_label.setStyleSheet("color: #8aa; font-size: 11px;")

        self.hover_label = QLabel("-")
        self.pixel_label = QLabel("-")
        self.depth_label = QLabel("-")
        self.vector_label = QLabel("-")
        self.geo_label = QLabel("-")
        self.depth_source_label = QLabel("-")
        self.pose_summary_label = QLabel("-")
        self.pose_summary_label.setWordWrap(True)
        self.pose_summary_label.setStyleSheet("color: #8aa; font-size: 11px;")

        for label in (
            self.hover_label,
            self.pixel_label,
            self.depth_label,
            self.vector_label,
            self.geo_label,
            self.depth_source_label,
            self.pose_summary_label,
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form.addRow("Fallback Height", self.ground_height_spin)
        form.addRow("Triangulation", self.triangulation_label)
        form.addRow("Hover", self.hover_label)
        form.addRow("Pixel", self.pixel_label)
        form.addRow("Depth", self.depth_label)
        form.addRow("Local XYZ", self.vector_label)
        form.addRow("Lat/Lon/Alt", self.geo_label)
        form.addRow("Depth Source", self.depth_source_label)
        form.addRow("Pose Context", self.pose_summary_label)

        mode_hint = QLabel(
            "Automatic engine selection: Multi-Frame Triangulation (capture folders), "
            "Depth Map (if loaded), or Ground Plane fallback."
        )
        mode_hint.setWordWrap(True)
        mode_hint.setStyleSheet("color: #8aa; font-size: 11px;")
        form.addRow("", mode_hint)
        self._update_depth_source_label()
        self._on_measurement_mode_changed(self.measurement_mode_combo.currentIndex())
        return group

    def _toggle_selection_readout(self, checked: bool) -> None:
        """Show/hide the selection readout to preserve space on smaller screens."""
        if not hasattr(self, "selection_group") or not hasattr(self, "selection_toggle_btn"):
            return
        self.selection_group.setVisible(bool(checked))
        self.selection_toggle_btn.setText(
            "Hide Selection Readout" if checked else "Show Selection Readout"
        )



    # Measurements -----------------------------------------------------------



    def _build_measurements_group(self) -> QGroupBox:



        group = QGroupBox("Saved Measurements")



        vbox = QVBoxLayout(group)



        self.measurement_count_label = QLabel("Saved Points: 0")
        self.measurement_count_label.setStyleSheet("color: #8aa; font-size: 11px;")

        self.measurement_table = QTableView()
        self.measurement_table.setMinimumHeight(220)



        self.measurement_table.setModel(self.measurement_model)



        self.measurement_table.horizontalHeader().setStretchLastSection(True)



        self.measurement_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)



        self.measurement_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)



        self.measurement_table.setAlternatingRowColors(True)



        self.measurement_table.verticalHeader().setVisible(False)

        self._measurement_copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.measurement_table)
        self._measurement_copy_shortcut.activated.connect(self._copy_selected_measurement_cells)







        controls = QHBoxLayout()



        export_csv_btn = QPushButton("Export CSV...")



        export_json_btn = QPushButton("Export JSON...")



        clear_btn = QPushButton("Clear")







        export_csv_btn.clicked.connect(self._export_csv)



        export_json_btn.clicked.connect(self._export_json)



        clear_btn.clicked.connect(self._clear_measurements)







        controls.addWidget(export_csv_btn)



        controls.addWidget(export_json_btn)



        controls.addStretch(1)



        controls.addWidget(clear_btn)







        vbox.addWidget(self.measurement_count_label)
        vbox.addWidget(self.measurement_table, stretch=1)

        vbox.addLayout(controls)
        self._update_measurement_count_label()
        return group







    # Menu -------------------------------------------------------------------



    def _create_menu_bar(self) -> None:



        bar = self.menuBar()



        file_menu = bar.addMenu("File")







        load_action = QAction("Load Panorama...", self)



        load_action.triggered.connect(self._on_load_panorama_clicked)



        file_menu.addAction(load_action)

        load_capture_action = QAction("Load Capture Folder...", self)
        load_capture_action.triggered.connect(self._on_load_capture_folder_clicked)
        file_menu.addAction(load_capture_action)







        depth_action = QAction("Load Depth Map...", self)



        depth_action.triggered.connect(self._on_load_depth_clicked)



        file_menu.addAction(depth_action)







        generate_action = QAction("Generate Depth Map...", self)



        generate_action.triggered.connect(self._on_generate_depth_clicked)



        file_menu.addAction(generate_action)







        file_menu.addSeparator()







        export_csv_action = QAction("Export Measurements as CSV...", self)



        export_csv_action.triggered.connect(self._export_csv)



        file_menu.addAction(export_csv_action)







        export_json_action = QAction("Export Measurements as JSON...", self)



        export_json_action.triggered.connect(self._export_json)



        file_menu.addAction(export_json_action)







        file_menu.addSeparator()







        exit_action = QAction("Quit", self)



        exit_action.triggered.connect(self.close)



        file_menu.addAction(exit_action)







    def _connect_signals(self) -> None:



        self.viewer.pointSelected.connect(self._on_point_selected)



        self.viewer.pointHovered.connect(self._on_point_hovered)







        self.lat_field.valueChanged.connect(self._update_camera_pose)



        self.lon_field.valueChanged.connect(self._update_camera_pose)



        self.alt_field.valueChanged.connect(self._update_camera_pose)



        self.bearing_field.valueChanged.connect(self._update_camera_pose)







        self._update_camera_pose()







    # ------------------------------------------------------------------



    # Camera pose handling



    def _update_camera_pose(self) -> None:
        pose = CameraPose(
            latitude=self.lat_field.value(),
            longitude=self.lon_field.value(),
            altitude=self.alt_field.value(),
            bearing=self.bearing_field.value() % 360.0,
        )
        self._state.metadata.camera_pose = pose
        self.viewer.set_bearing_reference(pose.bearing_rad)
        self._update_pose_warning()
        self._update_pose_summary()
        self._update_reset_button_state()
        logger.debug("Camera pose updated: {}", pose)

    def _set_camera_pose_fields(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        bearing: float,
    ) -> None:
        """Update pose widgets atomically and refresh the derived camera pose state."""
        fields = (self.lat_field, self.lon_field, self.alt_field, self.bearing_field)
        for field in fields:
            field.blockSignals(True)
        self.lat_field.setValue(latitude)
        self.lon_field.setValue(longitude)
        self.alt_field.setValue(altitude)
        self.bearing_field.setValue(bearing % 360.0)
        for field in fields:
            field.blockSignals(False)
        self._update_camera_pose()

    @staticmethod
    def _wrap_signed_angle_deg(value_deg: float) -> float:
        """Wrap an angle difference into [-180, 180) degrees."""
        return ((float(value_deg) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _circular_weighted_mean_deg(values_deg: list[float], weights: list[float]) -> Optional[float]:
        """Return weighted circular mean in degrees, or None if degenerate."""
        if not values_deg or len(values_deg) != len(weights):
            return None
        w = np.asarray(weights, dtype=np.float64)
        if w.size == 0 or float(np.sum(w)) <= 1e-9:
            return None
        angles = np.radians(np.asarray(values_deg, dtype=np.float64))
        x = float(np.sum(w * np.cos(angles)))
        y = float(np.sum(w * np.sin(angles)))
        if abs(x) <= 1e-12 and abs(y) <= 1e-12:
            return None
        return math.degrees(math.atan2(y, x)) % 360.0

    def _resolve_raw_frame_bearing_deg(self, frame) -> float:
        """Resolve per-frame bearing from export metadata only.

        If heading accuracy is weak, course/track is usually the more stable
        estimate. When heading accuracy is available, blend heading+track on
        the unit circle to avoid wrap-around artifacts near 0/360 degrees.
        """
        heading_deg = float(frame.heading_deg) % 360.0
        track_deg = float(frame.track_deg) % 360.0

        heading_acc = frame.heading_accuracy_deg
        if heading_acc is None or heading_acc <= 0.0:
            return heading_deg
        if heading_acc >= 4.0:
            return track_deg

        # Inverse-variance weighting on circular components.
        heading_sigma = max(0.25, float(heading_acc))
        track_sigma = 2.0
        w_heading = 1.0 / (heading_sigma * heading_sigma)
        w_track = 1.0 / (track_sigma * track_sigma)
        heading_rad = math.radians(heading_deg)
        track_rad = math.radians(track_deg)
        x = (w_heading * math.cos(heading_rad)) + (w_track * math.cos(track_rad))
        y = (w_heading * math.sin(heading_rad)) + (w_track * math.sin(track_rad))
        if abs(x) <= 1e-9 and abs(y) <= 1e-9:
            return heading_deg
        return math.degrees(math.atan2(y, x)) % 360.0

    def _resolve_frame_bearing_deg(self, frame, frame_index: Optional[int] = None) -> float:
        """Resolve final frame bearing (raw blend + capture-level calibration)."""
        if frame_index is not None and 0 <= frame_index < len(self._capture_resolved_bearings_deg):
            return float(self._capture_resolved_bearings_deg[frame_index]) % 360.0
        return (self._resolve_raw_frame_bearing_deg(frame) + self._capture_bearing_offset_deg) % 360.0

    def _estimate_trajectory_bearing_for_frame(
        self,
        capture: StitchedCapture,
        frame_index: int,
    ) -> tuple[Optional[float], Optional[float], float]:
        """Estimate frame travel direction from neighboring geodetic positions."""
        if frame_index < 0 or frame_index >= capture.frame_count:
            return None, None, 0.0

        frame = capture.frames[frame_index]

        def _course_between(idx_a: int, idx_b: int) -> tuple[Optional[float], float]:
            if idx_a < 0 or idx_b < 0 or idx_a >= capture.frame_count or idx_b >= capture.frame_count:
                return None, 0.0
            fa = capture.frames[idx_a]
            fb = capture.frames[idx_b]
            course_deg, _, distance_m = geodesy.geodesic_inverse(
                fa.latitude,
                fa.longitude,
                fb.latitude,
                fb.longitude,
            )
            if distance_m < self.MIN_TRAJECTORY_BEARING_BASELINE_M:
                return None, distance_m
            return course_deg, distance_m

        central_course, central_baseline = _course_between(frame_index - 1, frame_index + 1)
        if central_course is not None:
            hacc = frame.horizontal_accuracy_m if frame.horizontal_accuracy_m is not None else 2.0
            hacc = float(np.clip(hacc, 0.3, 50.0))
            sigma_deg = math.degrees(math.atan2(math.sqrt(2.0) * hacc, max(central_baseline, 1e-3)))
            return central_course, max(0.5, sigma_deg), central_baseline

        forward_course, forward_baseline = _course_between(frame_index, frame_index + 1)
        backward_course, backward_baseline = _course_between(frame_index - 1, frame_index)
        if forward_course is None and backward_course is None:
            return None, None, max(forward_baseline, backward_baseline)

        candidates: list[float] = []
        candidate_weights: list[float] = []
        if forward_course is not None:
            candidates.append(forward_course)
            candidate_weights.append(max(forward_baseline, 1e-3))
        if backward_course is not None:
            candidates.append(backward_course)
            candidate_weights.append(max(backward_baseline, 1e-3))
        blended_course = self._circular_weighted_mean_deg(candidates, candidate_weights)
        if blended_course is None:
            return None, None, max(forward_baseline, backward_baseline)

        hacc = frame.horizontal_accuracy_m if frame.horizontal_accuracy_m is not None else 2.0
        hacc = float(np.clip(hacc, 0.3, 50.0))
        baseline_m = max(forward_baseline, backward_baseline)
        sigma_deg = math.degrees(math.atan2(math.sqrt(2.0) * hacc, max(baseline_m, 1e-3)))
        return blended_course, max(0.5, sigma_deg), baseline_m

    def _fuse_frame_heading_with_trajectory(self, capture: StitchedCapture, frame_index: int) -> float:
        """Fuse frame heading with trajectory-derived course direction."""
        frame = capture.frames[frame_index]
        heading_deg = self._resolve_raw_frame_bearing_deg(frame)
        heading_sigma_deg = frame.heading_accuracy_deg if frame.heading_accuracy_deg is not None else 3.0
        heading_sigma_deg = float(np.clip(heading_sigma_deg, 0.5, 45.0))

        course_deg, course_sigma_deg, _ = self._estimate_trajectory_bearing_for_frame(capture, frame_index)
        if course_deg is None or course_sigma_deg is None:
            return heading_deg

        w_heading = 1.0 / max(heading_sigma_deg * heading_sigma_deg, 1e-6)
        w_course = 1.0 / max(course_sigma_deg * course_sigma_deg, 1e-6)
        blended = self._circular_weighted_mean_deg([heading_deg, course_deg], [w_heading, w_course])
        return heading_deg if blended is None else blended

    def _build_capture_bearing_solution(self, capture: StitchedCapture) -> list[float]:
        """Build per-frame bearings after heading-course fusion and boresight calibration."""
        if capture.frame_count <= 0:
            self._capture_bearing_offset_deg = 0.0
            return []

        raw_bearings = [
            self._fuse_frame_heading_with_trajectory(capture, frame_index)
            for frame_index in range(capture.frame_count)
        ]
        offset_deg = self._estimate_capture_bearing_offset_deg(capture, raw_bearings)
        self._capture_bearing_offset_deg = offset_deg
        return [((bearing + offset_deg) % 360.0) for bearing in raw_bearings]

    def _estimate_capture_bearing_offset_deg(
        self,
        capture: StitchedCapture,
        raw_bearings_deg: list[float],
    ) -> float:
        """Estimate constant yaw boresight offset from trajectory course.

        The estimator compares each frame's orientation to the WGS84 forward
        azimuth between consecutive frame positions, then computes a weighted
        circular mean of the offsets.
        """
        if capture.frame_count < 2 or len(raw_bearings_deg) != capture.frame_count:
            return 0.0

        offsets_rad: list[float] = []
        weights: list[float] = []

        for idx in range(capture.frame_count - 1):
            frame_a = capture.frames[idx]
            frame_b = capture.frames[idx + 1]
            course_deg, _, distance_m = geodesy.geodesic_inverse(
                frame_a.latitude,
                frame_a.longitude,
                frame_b.latitude,
                frame_b.longitude,
            )
            if distance_m < self.MIN_BEARING_CALIBRATION_BASELINE_M:
                continue

            delta_deg = self._wrap_signed_angle_deg(course_deg - raw_bearings_deg[idx])

            heading_sigma_deg = frame_a.heading_accuracy_deg if frame_a.heading_accuracy_deg is not None else 3.0
            heading_sigma_deg = float(np.clip(heading_sigma_deg, 0.5, 45.0))
            # Convert GNSS horizontal uncertainty into an angular component.
            hacc_m = frame_a.horizontal_accuracy_m if frame_a.horizontal_accuracy_m is not None else 2.0
            hacc_m = float(np.clip(hacc_m, 0.3, 50.0))
            pose_sigma_deg = math.degrees(math.atan2(hacc_m, max(distance_m, 1e-3)))
            sigma_deg = max(0.5, math.hypot(heading_sigma_deg, pose_sigma_deg))
            weight = min(2.0, distance_m / 5.0) / (sigma_deg * sigma_deg)
            if weight <= 0.0:
                continue

            offsets_rad.append(math.radians(delta_deg))
            weights.append(weight)

        if len(offsets_rad) < self.MIN_BEARING_CALIBRATION_SAMPLES:
            return 0.0
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-9:
            return 0.0

        x = float(np.sum(np.asarray(weights) * np.cos(np.asarray(offsets_rad))))
        y = float(np.sum(np.asarray(weights) * np.sin(np.asarray(offsets_rad))))
        concentration = math.hypot(x, y) / total_weight
        if concentration < self.MIN_BEARING_CALIBRATION_CONCENTRATION:
            return 0.0

        return math.degrees(math.atan2(y, x))

    def _build_capture_position_solution(self, capture: StitchedCapture) -> list[tuple[float, float, float]]:
        """Build a smoothed per-frame geodetic trajectory for measurement math.

        The smoothing runs in local ENU space to preserve metric continuity and
        reduce frame-to-frame GNSS jitter that amplifies triangulation noise.
        """
        if capture.frame_count <= 0:
            return []

        raw_geodetic = [(f.latitude, f.longitude, f.altitude_m) for f in capture.frames]
        if capture.frame_count < self.CAPTURE_SMOOTHING_MIN_FRAMES:
            return raw_geodetic

        origin_lat, origin_lon, origin_alt = raw_geodetic[0]
        enu = np.array(
            [
                geodesy.geodetic_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt)
                for lat, lon, alt in raw_geodetic
            ],
            dtype=np.float64,
        )

        window = min(self.CAPTURE_SMOOTHING_MAX_WINDOW, capture.frame_count)
        if window % 2 == 0:
            window -= 1
        if window < 5:
            return raw_geodetic

        smooth_e = savgol_filter(enu[:, 0], window_length=window, polyorder=3, mode="interp")
        smooth_n = savgol_filter(enu[:, 1], window_length=window, polyorder=3, mode="interp")
        smooth_u = savgol_filter(enu[:, 2], window_length=window, polyorder=2, mode="interp")

        smoothed: list[tuple[float, float, float]] = []
        for east, north, up in zip(smooth_e, smooth_n, smooth_u):
            lat, lon, alt = geodesy.enu_to_geodetic(
                float(east),
                float(north),
                float(up),
                origin_lat,
                origin_lon,
                origin_alt,
            )
            smoothed.append((lat, lon, alt))
        return smoothed

    def _resolve_frame_position(self, frame_index: int) -> tuple[float, float, float]:
        """Return resolved frame geodetic position (smoothed when available)."""
        capture = self._active_capture
        if (
            capture is not None
            and 0 <= frame_index < len(self._capture_resolved_positions)
            and frame_index < capture.frame_count
        ):
            return self._capture_resolved_positions[frame_index]
        if capture is None or frame_index < 0 or frame_index >= capture.frame_count:
            pose = self._state.metadata.camera_pose
            return (pose.latitude, pose.longitude, pose.altitude)
        frame = capture.frames[frame_index]
        return (frame.latitude, frame.longitude, frame.altitude_m)

    def _capture_frame_gray(self, frame_index: int) -> Optional[np.ndarray]:
        """Load (or fetch cached) grayscale frame for local correspondence refinement."""
        if frame_index in self._capture_gray_cache:
            return self._capture_gray_cache[frame_index]
        if self._active_capture is None or frame_index < 0 or frame_index >= self._active_capture.frame_count:
            return None
        frame = self._active_capture.frames[frame_index]
        image = load_equirectangular_image(frame.image_path)
        image = self._prepare_capture_frame_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self._capture_gray_cache[frame_index] = gray
        return gray

    @staticmethod
    def _extract_wrapped_patch(gray: np.ndarray, center_u: int, center_v: int, radius_px: int) -> Optional[np.ndarray]:
        """Extract a square patch with horizontal wrap-around support."""
        height, width = gray.shape[:2]
        if radius_px <= 0 or height <= 0 or width <= 0:
            return None
        top = center_v - radius_px
        bottom = center_v + radius_px + 1
        if top < 0 or bottom > height:
            return None
        cols = (np.arange(center_u - radius_px, center_u + radius_px + 1, dtype=np.int64) % width).astype(np.int64)
        return np.ascontiguousarray(gray[top:bottom][:, cols])

    def _refine_click_with_template_match(
        self,
        *,
        anchor_frame_index: int,
        anchor_u: int,
        anchor_v: int,
        current_frame_index: int,
        current_u: int,
        current_v: int,
        width: int,
        height: int,
    ) -> tuple[int, int, float]:
        """Refine clicked correspondence in current frame using OpenCV template matching."""
        if anchor_frame_index == current_frame_index:
            return current_u, current_v, 1.0
        anchor_gray = self._capture_frame_gray(anchor_frame_index)
        current_gray = self._capture_frame_gray(current_frame_index)
        if anchor_gray is None or current_gray is None:
            return current_u, current_v, 0.0

        patch_radius = self.TRIANGULATION_TEMPLATE_PATCH_RADIUS_PX
        search_radius = self.TRIANGULATION_TEMPLATE_SEARCH_RADIUS_PX
        patch = self._extract_wrapped_patch(anchor_gray, anchor_u, anchor_v, patch_radius)
        if patch is None:
            return current_u, current_v, 0.0

        tile = np.concatenate((current_gray, current_gray, current_gray), axis=1)
        center_u = current_u + width
        x0 = center_u - search_radius - patch_radius
        x1 = center_u + search_radius + patch_radius + 1
        y0 = max(0, current_v - search_radius - patch_radius)
        y1 = min(height, current_v + search_radius + patch_radius + 1)
        if x0 < 0 or x1 > tile.shape[1] or y1 - y0 < patch.shape[0] or x1 - x0 < patch.shape[1]:
            return current_u, current_v, 0.0

        search = tile[y0:y1, x0:x1]
        response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        refined_u = int((x0 + max_loc[0] + patch_radius) % width)
        refined_v = int(np.clip(y0 + max_loc[1] + patch_radius, 0, height - 1))
        return refined_u, refined_v, float(max_val)

    def _build_triangulation_observation(
        self,
        *,
        frame_index: int,
        pixel_u: int,
        pixel_v: int,
        theta: float,
        phi: float,
        pitch_deg: float,
        roll_deg: float,
        heading_accuracy_deg: Optional[float],
        horizontal_accuracy_m: Optional[float],
    ) -> TriangulationObservation:
        """Build one triangulation ray observation in ECEF coordinates."""
        lat, lon, alt = self._resolve_frame_position(frame_index)
        bearing_deg = (
            self._capture_resolved_bearings_deg[frame_index]
            if 0 <= frame_index < len(self._capture_resolved_bearings_deg)
            else self._state.metadata.camera_pose.bearing
        )
        direction_enu = np.array(
            geometry.enu_vector_from_angles(
                theta,
                phi,
                1.0,
                math.radians(bearing_deg),
                pitch_rad=math.radians(pitch_deg),
                roll_rad=math.radians(roll_deg),
            ),
            dtype=np.float64,
        )
        direction_ecef = np.array(
            geodesy.enu_to_ecef(
                float(direction_enu[0]),
                float(direction_enu[1]),
                float(direction_enu[2]),
                lat,
                lon,
            ),
            dtype=np.float64,
        )
        origin_ecef = np.array(geodesy.geodetic_to_ecef(lon, lat, alt), dtype=np.float64)
        return TriangulationObservation(
            frame_index=frame_index,
            pixel_u=pixel_u,
            pixel_v=pixel_v,
            theta=theta,
            phi=phi,
            origin_ecef=origin_ecef,
            direction_ecef=direction_ecef,
            horizontal_accuracy_m=horizontal_accuracy_m,
            heading_accuracy_deg=heading_accuracy_deg,
        )

    @staticmethod
    def _upsert_triangulation_observation(
        observations: list[TriangulationObservation],
        observation: TriangulationObservation,
    ) -> list[TriangulationObservation]:
        """Insert or replace a per-frame observation in the active track."""
        next_obs = [obs for obs in observations if obs.frame_index != observation.frame_index]
        next_obs.append(observation)
        next_obs.sort(key=lambda item: item.frame_index)
        return next_obs

    def _recommend_triangulation_frames(self, anchor_frame_index: int, limit: int = 3) -> list[int]:
        """Suggest frame indices likely to provide stronger parallax than adjacent frames."""
        capture = self._active_capture
        if capture is None or capture.frame_count < 2:
            return []
        if anchor_frame_index < 0 or anchor_frame_index >= capture.frame_count:
            return []
        anchor_frame = capture.frames[anchor_frame_index]
        anchor_lat, anchor_lon, _ = self._resolve_frame_position(anchor_frame_index)
        candidates: list[tuple[float, int]] = []
        for frame_index, frame in enumerate(capture.frames):
            if frame_index == anchor_frame_index:
                continue
            if not self._frame_heading_usable(frame):
                continue
            lat, lon, _ = self._resolve_frame_position(frame_index)
            _, _, baseline_m = geodesy.geodesic_inverse(anchor_lat, anchor_lon, lat, lon)
            if baseline_m < self.MIN_TRIANGULATION_BASELINE_M:
                continue
            hacc = frame.horizontal_accuracy_m if frame.horizontal_accuracy_m is not None else 2.0
            headacc = frame.heading_accuracy_deg if frame.heading_accuracy_deg is not None else 3.0
            anchor_headacc = (
                anchor_frame.heading_accuracy_deg
                if anchor_frame.heading_accuracy_deg is not None
                else 3.0
            )
            sigma_deg = math.hypot(max(0.5, float(headacc)), max(0.5, float(anchor_headacc)))
            sigma_rad = math.radians(sigma_deg)
            # Predict pair angle from baseline and a conservative 25 m feature range.
            predicted_angle_rad = math.atan2(max(baseline_m, 1e-3), 25.0)
            predicted_sigma_m = (25.0 * sigma_rad) / max(math.sin(predicted_angle_rad), 1e-3)
            score = baseline_m / max(0.5, predicted_sigma_m + float(hacc))
            candidates.append((score, frame_index))
        candidates.sort(reverse=True)
        return [frame_idx for _, frame_idx in candidates[: max(1, limit)]]

    def _extract_wrapped_window(
        self,
        gray: np.ndarray,
        center_u: int,
        center_v: int,
        radius_px: int,
    ) -> tuple[Optional[np.ndarray], Optional[tuple[int, int]]]:
        """Extract a square window around a pixel with horizontal wrap-around."""
        height, width = gray.shape[:2]
        if radius_px <= 0:
            return None, None
        top = center_v - radius_px
        bottom = center_v + radius_px + 1
        if top < 0 or bottom > height:
            return None, None
        cols = (np.arange(center_u - radius_px, center_u + radius_px + 1, dtype=np.int64) % width).astype(np.int64)
        patch = np.ascontiguousarray(gray[top:bottom][:, cols])
        return patch, (top, int((center_u - radius_px) % width))

    def _frame_heading_usable(self, frame) -> bool:
        """Return whether frame heading accuracy is strong enough for triangulation."""
        if frame.heading_accuracy_deg is None:
            return True
        return float(frame.heading_accuracy_deg) <= self.TRIANGULATION_MAX_HEADING_ACC_DEG

    def _pair_pose_consistency_opencv(
        self,
        *,
        anchor_frame_index: int,
        anchor_u: int,
        anchor_v: int,
        current_frame_index: int,
        current_u: int,
        current_v: int,
        width: int,
        height: int,
    ) -> Optional[tuple[float, bool]]:
        """Estimate pairwise pose consistency via OpenCV essential geometry.

        Returns:
            (inlier_ratio, cheirality_ok) or ``None`` when insufficient matches.
        """
        if anchor_frame_index == current_frame_index:
            return None
        anchor_gray = self._capture_frame_gray(anchor_frame_index)
        current_gray = self._capture_frame_gray(current_frame_index)
        if anchor_gray is None or current_gray is None:
            return None

        radius = self.TRIANGULATION_POSE_CHECK_WINDOW_RADIUS_PX
        patch_a, _ = self._extract_wrapped_window(anchor_gray, anchor_u, anchor_v, radius)
        patch_b, _ = self._extract_wrapped_window(current_gray, current_u, current_v, radius)
        if patch_a is None or patch_b is None:
            return None

        orb = cv2.ORB_create(nfeatures=500, fastThreshold=12)
        kp_a, des_a = orb.detectAndCompute(patch_a, None)
        kp_b, des_b = orb.detectAndCompute(patch_b, None)
        if des_a is None or des_b is None or len(kp_a) < 24 or len(kp_b) < 24:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(des_a, des_b, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 20:
            return None

        def to_global(match_point: cv2.KeyPoint, center_u_px: int, center_v_px: int) -> tuple[float, float]:
            x_local = float(match_point.pt[0])
            y_local = float(match_point.pt[1])
            u_px = (center_u_px - radius + x_local) % width
            v_px = float(np.clip(center_v_px - radius + y_local, 0.0, float(height - 1)))
            return float(u_px), v_px

        pts1 = []
        pts2 = []
        for match in good:
            u1, v1 = to_global(kp_a[match.queryIdx], anchor_u, anchor_v)
            u2, v2 = to_global(kp_b[match.trainIdx], current_u, current_v)
            pts1.append((u1, v1))
            pts2.append((u2, v2))
        points1 = np.asarray(pts1, dtype=np.float64)
        points2 = np.asarray(pts2, dtype=np.float64)
        if points1.shape[0] < 16:
            return None

        fx = width / (2.0 * math.pi)
        fy = height / math.pi
        cx = width / 2.0
        cy = height / 2.0
        k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        essential, mask = cv2.findEssentialMat(
            points1,
            points2,
            cameraMatrix=k_mat,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=2.0,
        )
        if essential is None or mask is None:
            return None
        inlier_mask = mask.reshape(-1).astype(bool)
        if int(np.count_nonzero(inlier_mask)) < 12:
            return None
        inlier_ratio = float(np.count_nonzero(inlier_mask) / max(1, points1.shape[0]))

        _, r_mat, t_vec, pose_mask = cv2.recoverPose(
            essential,
            points1,
            points2,
            cameraMatrix=k_mat,
            mask=mask,
        )
        if pose_mask is None:
            return inlier_ratio, False

        clicked1 = np.array([[float(anchor_u)], [float(anchor_v)]], dtype=np.float64)
        clicked2 = np.array([[float(current_u)], [float(current_v)]], dtype=np.float64)
        p1 = k_mat @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
        p2 = k_mat @ np.hstack((r_mat, t_vec.reshape(3, 1)))
        triangulated = cv2.triangulatePoints(p1, p2, clicked1, clicked2)
        if triangulated.shape[1] < 1:
            return inlier_ratio, False
        point_cam1 = triangulated[:3, 0] / max(1e-12, triangulated[3, 0])
        point_cam2 = (r_mat @ point_cam1.reshape(3, 1)) + t_vec.reshape(3, 1)
        cheirality_ok = bool(point_cam1[2] > 0.0 and point_cam2[2, 0] > 0.0)
        return inlier_ratio, cheirality_ok

    def _apply_view_roll(self) -> None:
        """Apply pose-derived roll and manual operator offset to the viewer."""
        total_roll = self._capture_base_roll_offset_deg + self._pose_roll_deg + self._manual_roll_correction_deg
        self.viewer.set_view_roll_degrees(total_roll)

    def _prepare_capture_frame_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize stitched capture frame orientation before rendering/math.

        Some stitched capture exports encode panoramas with horizontal mirroring.
        Flipping once here aligns text/readouts and keeps pixel->ray geometry
        consistent with real-world left/right directions.
        """
        if not self._capture_horizontal_flip_enabled:
            return image
        return np.ascontiguousarray(np.flip(image, axis=1))


    # ------------------------------------------------------------------



    # Data loading



    def _on_load_capture_folder_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Capture Folder",
            "",
        )
        if not folder:
            return

        capture_path = Path(folder)
        self.statusBar().showMessage("Scanning capture folder...")
        task = FunctionTask(self._discover_capture_folder_payload, capture_path)
        self._bind_task(task, self._capture_folder_loaded_from_payload)

    @staticmethod
    def _discover_capture_folder_payload(path: Path) -> tuple[str, object]:
        """Discover stitched or raw capture folders with one unified task API."""
        try:
            stitched = discover_stitched_capture(path)
            return ("stitched", stitched)
        except Exception as stitched_error:
            try:
                raw = discover_raw_capture(path)
                return ("raw", raw)
            except Exception:
                raise stitched_error

    def _capture_folder_loaded_from_payload(self, payload: tuple[str, object]) -> None:
        """Route loaded folder payload to stitched or raw handlers."""
        kind, value = payload
        if kind == "stitched":
            self._capture_folder_loaded(value)  # type: ignore[arg-type]
            return
        if kind == "raw":
            self._raw_capture_loaded(value)  # type: ignore[arg-type]
            return
        raise ValueError(f"Unsupported capture payload kind: {kind}")

    def _raw_capture_loaded(self, capture: RawCapture) -> None:
        """Load raw capture metadata for calibrated-sensor backend workflows."""
        self._clear_active_capture()
        self._active_raw_capture = capture
        frame_count = min(len(capture.sensor_timestamps_ns[idx]) for idx in capture.sensor_timestamps_ns)
        self.capture_status_label.setText(
            "Raw capture loaded (4-sensor). "
            f"Frames/sensor: {frame_count} | Folder: {capture.capture_root}"
        )
        self.capture_status_label.setToolTip(str(capture.capture_root))
        self.statusBar().showMessage(
            "Raw capture parsed. Calibrated sensor-ray solver is available for pipeline integration.",
            5000,
        )
        QMessageBox.information(
            self,
            "Raw Capture Loaded",
            "Raw 4-sensor capture was parsed successfully.\n"
            "Viewer-based triangulation still uses stitched panoramas.\n"
            "Use the calibrated raw-sensor APIs for pre-stitch geospatial solving.",
        )

    def _capture_folder_loaded(self, capture: StitchedCapture) -> None:
        self._active_raw_capture = None
        self._active_capture = capture
        self._capture_base_roll_offset_deg = 0.0
        self._capture_resolved_bearings_deg = self._build_capture_bearing_solution(capture)
        self._capture_resolved_positions = self._build_capture_position_solution(capture)
        self._capture_gray_cache.clear()
        self._clear_measurement_history()
        self._clear_triangulation_anchor(update_label=True)
        self._on_measurement_mode_changed(self.measurement_mode_combo.currentIndex())
        self._capture_spin_block = True
        self.capture_frame_spin.setEnabled(capture.frame_count > 0)
        self.capture_frame_spin.setRange(0, max(0, capture.frame_count - 1))
        self.capture_frame_spin.setValue(0)
        self.capture_frame_total_label.setText(f"/ {capture.frame_count - 1}")
        self.detected_frames_label.setText(f"Detected Frames: {capture.frame_count}")
        self._capture_spin_block = False

        capture_label = capture.capture_id or capture.capture_root.name
        self.capture_status_label.setText(
            f"Capture sequence: {capture_label} | Frames: {capture.frame_count} | "
            f"Resolution: {capture.frame_width}x{capture.frame_height}\n"
            f"Folder: {capture.capture_root}"
        )
        self.capture_status_label.setToolTip(str(capture.capture_root))
        self.statusBar().showMessage(
            f"Capture folder validated: {capture.capture_root} "
            f"(bearing calibration {self._capture_bearing_offset_deg:+.2f} deg, "
            f"trajectory smoothing {'on' if len(self._capture_resolved_positions) == capture.frame_count else 'off'})",
            4000,
        )
        self._load_capture_frame(0, reset_orientation=True)

    def _on_capture_frame_changed(self, frame_index: int) -> None:
        if self._capture_spin_block or self._active_capture is None:
            return
        self._load_capture_frame(frame_index, reset_orientation=False)

    def _load_capture_frame(self, frame_index: int, reset_orientation: bool) -> None:
        capture = self._active_capture
        if capture is None:
            return
        if frame_index < 0 or frame_index >= capture.frame_count:
            return
        frame = capture.frames[frame_index]
        task = FunctionTask(load_equirectangular_image, frame.image_path)
        handler = lambda image, idx=frame_index, orient=reset_orientation: self._capture_frame_loaded(
            idx, image, orient
        )
        self._bind_task(task, handler)
        self.statusBar().showMessage(f"Loading capture frame {frame_index}...")

    def _capture_frame_loaded(self, frame_index: int, image: np.ndarray, reset_orientation: bool) -> None:
        capture = self._active_capture
        if capture is None or frame_index < 0 or frame_index >= capture.frame_count:
            return
        current_frame_index = self._current_capture_frame_index()
        if current_frame_index is not None and frame_index != current_frame_index:
            logger.debug(
                "Ignoring stale frame load: loaded={}, current={}",
                frame_index,
                current_frame_index,
            )
            return

        frame = capture.frames[frame_index]
        image = self._prepare_capture_frame_image(image)
        self._pose_pitch_deg = float(frame.pitch_deg)
        self._pose_roll_deg = float(frame.roll_deg)
        self._apply_view_roll()
        self._state.set_image(image, frame.image_path)
        self._capture_gray_cache[frame_index] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self._refresh_viewer_image(reset_orientation=reset_orientation)
        self.viewer.set_instruction_visible(frame_index == 0)
        resolved_bearing_deg = self._resolve_frame_bearing_deg(frame, frame_index)
        resolved_lat, resolved_lon, resolved_alt = self._resolve_frame_position(frame_index)
        self._set_camera_pose_fields(
            latitude=resolved_lat,
            longitude=resolved_lon,
            altitude=resolved_alt,
            bearing=resolved_bearing_deg,
        )

        self.image_path_display.setText(str(frame.image_path))
        self.depth_path_display.setText("[None]")
        self._sync_viewer_measurement_markers()
        self._needs_reorient = False

        self._update_format_status()
        self._update_depth_source_label()
        self._update_pose_warning()
        self._update_pose_summary()
        self._update_reset_button_state()
        self._update_render_view_controls()
        self._update_overlay_metadata()

        hacc_label = (
            f"{frame.horizontal_accuracy_m:.2f} m"
            if frame.horizontal_accuracy_m is not None
            else "n/a"
        )
        self.statusBar().showMessage(
            f"Capture frame {frame_index} loaded "
            f"(heading {resolved_bearing_deg:.2f} deg, pitch {frame.pitch_deg:.2f} deg, "
            f"roll {frame.roll_deg:.2f} deg, base offset {self._capture_base_roll_offset_deg:+.1f} deg, "
            f"hAcc {hacc_label}, bearing calib {self._capture_bearing_offset_deg:+.2f} deg, "
            f"mirror {'on' if self._capture_horizontal_flip_enabled else 'off'})",
            4000,
        )

    def _clear_active_capture(self) -> None:
        self._active_capture = None
        self._active_raw_capture = None
        self._capture_base_roll_offset_deg = 0.0
        self._capture_bearing_offset_deg = 0.0
        self._capture_resolved_bearings_deg = []
        self._capture_resolved_positions = []
        self._capture_gray_cache.clear()
        self._pose_pitch_deg = 0.0
        self._pose_roll_deg = 0.0
        self._apply_view_roll()
        self._sync_viewer_measurement_markers()
        self._clear_triangulation_anchor()
        self._on_measurement_mode_changed(self.measurement_mode_combo.currentIndex())
        self._capture_spin_block = True
        self.capture_frame_spin.setEnabled(False)
        self.capture_frame_spin.setRange(0, 0)
        self.capture_frame_spin.setValue(0)
        self.capture_frame_total_label.setText("/ 0")
        self.detected_frames_label.setText("Detected Frames: 0")
        self.capture_status_label.setText("Capture sequence: -")
        self._capture_spin_block = False

    def _on_load_panorama_clicked(self) -> None:



        path, _ = QFileDialog.getOpenFileName(



            self,



            "Select Panorama Image",



            "",



            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",



        )



        if not path:



            return



        file_path = Path(path)
        self._clear_active_capture()



        self.statusBar().showMessage("Loading panorama...")



        task = FunctionTask(load_equirectangular_image, file_path)



        self._bind_task(task, lambda image: self._panorama_loaded(image, file_path))







    def _on_load_depth_clicked(self) -> None:



        path, _ = QFileDialog.getOpenFileName(



            self,



            "Select Depth Map",



            "",



            "Depth Maps (*.png *.tif *.tiff *.exr)",



        )



        if not path:



            return



        file_path = Path(path)



        self.statusBar().showMessage("Loading depth map...")



        task = FunctionTask(load_depth_map, file_path)



        self._bind_task(task, lambda depth: self._depth_loaded(depth, file_path, source=f"Loaded from {file_path.name}"))







    def _on_generate_depth_clicked(self) -> None:



        if not self._state.has_image:



            QMessageBox.information(self, "No Panorama", "Load a panorama image before generating depth.")



            return







        detection = self._state.stereo_detection



        width = self._state.image.shape[1] if self._state.has_image else 4096



        default_focal = estimate_default_focal(width)







        dialog = DepthGenerationDialog(detection, default_focal, parent=self)



        if dialog.exec() != QDialog.DialogCode.Accepted:



            return



        request = dialog.request()



        self._start_depth_generation(request)







    def _start_depth_generation(self, request) -> None:



        if request.mode == DepthGenerationMode.PANORAMA_STEREO:



            raw_image = self._state.raw_image



            if raw_image is None:



                QMessageBox.warning(self, "Raw Image Missing", "Original panorama data unavailable for stereo computation.")



                return



            args = (



                raw_image,



                request.stereo_format,



                request.baseline_m,



                request.focal_length_px,



                request.rectify,



                request.downsample_factor,



            )



            task = FunctionTask(compute_depth_from_panorama_array, *args)



        else:



            config = StereoDepthConfig(



                baseline_m=request.baseline_m,



                focal_length_px=request.focal_length_px,



            )



            task = FunctionTask(compute_depth_from_stereo, request.left_path, request.right_path, config)







        handler = lambda depth, req=request: self._handle_depth_generation_success(depth, req)



        self._bind_task(task, handler)



        self.statusBar().showMessage("Computing depth map... this may take a few moments")











    def _handle_depth_generation_success(self, depth: np.ndarray, request) -> None:



        source = "Stereo"



        if request.mode == DepthGenerationMode.PANORAMA_STEREO:



            fmt = request.stereo_format.value if request.stereo_format else "Stereo"



            source = f"Stereo from panorama ({fmt}, close-range accuracy high)"



        else:



            source = "External stereo pair"







        self._state.set_depth(depth, depth_path=None, source=source)



        self.depth_path_display.setText("[Generated]")



        self._update_depth_source_label()



        self._clear_measurement_history()



        self.statusBar().showMessage("Depth map generated", 4000)







        if request.mode == DepthGenerationMode.PANORAMA_STEREO and request.output_path is not None:



            self._save_depth_to_file(depth, request.output_path)







    def _save_depth_to_file(self, depth: np.ndarray, path: Path) -> None:



        depth_mm = np.nan_to_num(depth, nan=0.0)



        depth_mm = np.clip(depth_mm * 1000.0, 0, 65535).astype(np.uint16)



        try:



            success = cv2.imwrite(str(path), depth_mm)



        except Exception as exc:  # noqa: BLE001



            QMessageBox.warning(self, "Save Failed", f"Unable to write depth map: {exc}")



        else:



            if success:



                self.statusBar().showMessage(f"Depth saved ... {path}", 4000)



            else:



                QMessageBox.warning(self, "Save Failed", "OpenCV could not write the depth image.")







    def _bind_task(self, task: FunctionTask, on_success) -> None:



        self._active_tasks.add(task)



        task.signals.finished.connect(lambda result, t=task: self._on_task_success(t, result, on_success))



        task.signals.failed.connect(lambda message, t=task: self._on_task_failure(t, message))



        self._task_runner.submit(task)







    def _on_task_success(self, task: FunctionTask, result, on_success) -> None:



        self._active_tasks.discard(task)



        try:



            on_success(result)



        finally:



            self.statusBar().clearMessage()







    def _on_task_failure(self, task: FunctionTask, message: str) -> None:



        self._active_tasks.discard(task)



        self.statusBar().clearMessage()



        logger.error("Background task failed: {}", message)



        QMessageBox.critical(self, "Error", f"Operation failed:\n{message}")







    def _panorama_loaded(self, image: np.ndarray, path: Path) -> None:
        logger.info("Panorama image loaded: {}", path)
        self._state.set_image(image, path)

        self._refresh_viewer_image(reset_orientation=True)
        self.viewer.set_instruction_visible(True)

        pose = self._state.metadata.camera_pose
        self.viewer.set_bearing_reference(pose.bearing_rad, recenter=True)

        self.image_path_display.setText(str(path))
        self.depth_path_display.setText("[None]")
        self._clear_measurement_history()
        self._needs_reorient = False

        self._update_format_status()
        self._update_depth_source_label()
        self._update_pose_warning()
        self._update_pose_summary()
        self._update_reset_button_state()
        self._update_render_view_controls()
        self._update_overlay_metadata()

        detection = self._state.format_detection
        if detection and detection.probable_fisheye and self._state.distortion_corrected:
            QMessageBox.information(
                self,
                "Fisheye Detected",
                "Image appeared distorted and was automatically remapped to equirectangular. Use Correct Distortion if further adjustments are needed.",
            )

        self.statusBar().showMessage("Panorama loaded", 3000)

        stereo_detection = self._state.stereo_detection
        if stereo_detection and stereo_detection.is_stereo:
            logger.info("Stereo panorama detected; continuing in panorama-first workflow.")

    def _depth_loaded(self, depth: np.ndarray, path: Optional[Path], source: str) -> None:



        if path:



            logger.info("Depth map loaded: {}", path)



        else:



            logger.info("Depth map generated from stereo pair")



        self._state.set_depth(depth, depth_path=path, source=source)



        if path:



            self.depth_path_display.setText(str(path))



        else:



            self.depth_path_display.setText("[Generated]")



        self._update_depth_source_label()



        self.statusBar().showMessage("Depth map ready", 3000)



        self._validate_depth_alignment()







    def _update_stereo_status(self, detection: Optional[StereoDetectionResult] = None) -> None:
        # Backwards-compat helper retained for older call sites.
        self._update_format_status()







    def _populate_format_combo(self) -> None:
        if not hasattr(self, 'format_combo'):
            return
        self._format_combo_block = True
        self.format_combo.clear()
        for fmt in (
            PanoramaInputFormat.AUTO,
            PanoramaInputFormat.MONO_EQUI,
            PanoramaInputFormat.STEREO_TB,
            PanoramaInputFormat.STEREO_SBS,
            PanoramaInputFormat.FISHEYE_MONO,
            PanoramaInputFormat.FISHEYE_STEREO,
        ):
            self.format_combo.addItem(fmt.value, fmt)
        self._format_combo_block = False

    def _sync_format_selection(self) -> None:
        if not hasattr(self, 'format_combo'):
            return
        target = self._state.format_override or PanoramaInputFormat.AUTO
        index = self.format_combo.findData(target)
        if index < 0:
            index = self.format_combo.findData(PanoramaInputFormat.AUTO)
        self._format_combo_block = True
        if index >= 0:
            self.format_combo.setCurrentIndex(index)
        self.format_combo.setEnabled(self._state.has_image)
        self._format_combo_block = False

    def _resolve_measurement_mode(self) -> str:
        """Resolve the effective measurement mode from current available inputs."""
        if self._measurement_mode != self.MEASUREMENT_MODE_AUTO:
            return self._measurement_mode
        if self._active_capture is not None and self._active_capture.frame_count >= 2:
            return self.MEASUREMENT_MODE_TRIANGULATION
        if self._state.has_depth:
            return self.MEASUREMENT_MODE_DEPTH
        return self.MEASUREMENT_MODE_GROUND

    def _update_depth_source_label(self) -> None:
        effective_mode = self._resolve_measurement_mode()
        if effective_mode == self.MEASUREMENT_MODE_GROUND:
            camera_height = self.ground_height_spin.value() if hasattr(self, "ground_height_spin") else 1.7
            source = f"Ground plane estimate (camera height {camera_height:.2f} m)"
        elif effective_mode == self.MEASUREMENT_MODE_TRIANGULATION:
            source = "Multi-frame triangulation"
        else:
            source = self._state.depth_source or "-"
        self.depth_source_label.setText(source)

    def _update_render_view_controls(self) -> None:
        if not hasattr(self, 'render_view_combo'):
            return
        self._sync_format_selection()
        if hasattr(self, 'distortion_button'):
            self.distortion_button.setEnabled(self._state.has_image)
        combo = self.render_view_combo
        self._syncing_render_view = True
        combo.blockSignals(True)
        combo.clear()
        available = self._state.available_render_eyes() if hasattr(self._state, 'available_render_eyes') else {"left"}
        items = [("Left", "left"), ("Right", "right"), ("Anaglyph", "anaglyph")]
        for label, eye in items:
            if eye == "left" or eye in available:
                combo.addItem(label, eye)
        current_eye = getattr(self._state, 'render_eye', 'left')
        if current_eye not in {combo.itemData(i) for i in range(combo.count())}:
            current_eye = "left"
        index = combo.findData(current_eye)
        if combo.count() > 0:
            combo.setCurrentIndex(max(0, index))
        combo.setEnabled(combo.count() > 1)
        combo.blockSignals(False)
        if hasattr(self, 'anaglyph_checkbox'):
            self.anaglyph_checkbox.blockSignals(True)
            self.anaglyph_checkbox.setEnabled("anaglyph" in available)
            self.anaglyph_checkbox.setChecked(current_eye == "anaglyph")
            self.anaglyph_checkbox.blockSignals(False)
        self._syncing_render_view = False

    def _on_render_view_changed(self, index: int) -> None:
        if not hasattr(self, 'render_view_combo') or self._syncing_render_view:
            return
        eye = self.render_view_combo.currentData()
        if eye is None:
            return
        try:
            self._state.set_render_eye(eye)
        except ValueError as exc:
            QMessageBox.warning(self, "Render View", str(exc))
            self._update_render_view_controls()
            return
        self._refresh_viewer_image(reset_orientation=False)
        self.viewer.set_instruction_visible(False)
        if hasattr(self, 'anaglyph_checkbox'):
            self.anaglyph_checkbox.blockSignals(True)
            self.anaglyph_checkbox.setChecked(eye == 'anaglyph')
            self.anaglyph_checkbox.blockSignals(False)
        human_eye = eye.capitalize()
        self.statusBar().showMessage(f"Rendering {human_eye} view", 2000)
        self._update_render_view_controls()
        self._update_overlay_metadata()

    def _on_mouse_sensitivity_changed(self, value: int) -> None:
        value = max(1, min(int(value), 10))
        sensitivity = value / 1000.0
        if hasattr(self, 'mouse_sensitivity_value_label'):
            self.mouse_sensitivity_value_label.setText(f'{sensitivity:.3f}')
        self.viewer.set_mouse_sensitivity(sensitivity)

    def _on_roll_correction_changed(self, value: float) -> None:
        self._manual_roll_correction_deg = float(value)
        self._apply_view_roll()

    def _on_measurement_mode_changed(self, index: int) -> None:
        if not hasattr(self, "measurement_mode_combo"):
            return
        mode = self.measurement_mode_combo.currentData()
        if mode not in {
            self.MEASUREMENT_MODE_AUTO,
            self.MEASUREMENT_MODE_DEPTH,
            self.MEASUREMENT_MODE_GROUND,
            self.MEASUREMENT_MODE_TRIANGULATION,
        }:
            mode = self.MEASUREMENT_MODE_AUTO
        self._measurement_mode = mode

        effective_mode = self._resolve_measurement_mode()
        is_ground_mode = effective_mode == self.MEASUREMENT_MODE_GROUND
        if effective_mode != self.MEASUREMENT_MODE_TRIANGULATION and self._triangulation_anchor is not None:
            self._clear_triangulation_anchor(update_label=False)
        if hasattr(self, "ground_height_spin"):
            self.ground_height_spin.setEnabled(is_ground_mode)
        if hasattr(self, "triangulation_label"):
            if effective_mode == self.MEASUREMENT_MODE_TRIANGULATION:
                if self._triangulation_anchor is None:
                    self.triangulation_label.setText("Step 1: click feature in frame A")
                else:
                    self.triangulation_label.setText(
                        f"Anchor frame {self._triangulation_anchor.frame_index} set. "
                        "Step 2: switch frame and click same feature (extra frames improve accuracy)."
                    )
            else:
                self.triangulation_label.setText("Not active")
        self._update_depth_source_label()

        if effective_mode == self.MEASUREMENT_MODE_TRIANGULATION:
            self.statusBar().showMessage("Measurement engine: Multi-frame triangulation", 2500)
        elif effective_mode == self.MEASUREMENT_MODE_DEPTH:
            self.statusBar().showMessage("Measurement engine: Depth map", 2500)
        else:
            self.statusBar().showMessage("Measurement engine: Ground plane fallback", 2500)

    def _on_ground_height_changed(self, value: float) -> None:
        if self._resolve_measurement_mode() == self.MEASUREMENT_MODE_GROUND:
            self._update_depth_source_label()

    def _current_capture_frame_index(self) -> Optional[int]:
        if self._active_capture is None or not hasattr(self, "capture_frame_spin"):
            return None
        return int(self.capture_frame_spin.value())

    def _clear_triangulation_anchor(self, *, update_label: bool = True) -> None:
        self._triangulation_anchor = None
        self._triangulation_track_observations.clear()
        self._sync_viewer_measurement_markers()
        if update_label and hasattr(self, "triangulation_label"):
            if self._resolve_measurement_mode() == self.MEASUREMENT_MODE_TRIANGULATION:
                self.triangulation_label.setText("Step 1: click feature in frame A")
            else:
                self.triangulation_label.setText("Not active")

    def _append_measurement(
        self,
        measurement: PointMeasurement,
        *,
        marker_theta: float,
        marker_phi: float,
        marker_frame_index: Optional[int] = None,
    ) -> None:
        """Append one measured point and keep marker/table state in sync."""
        self._measurement_history.append(measurement)
        self._measurement_marker_angles.append((marker_theta, marker_phi))
        self._measurement_marker_frames.append(marker_frame_index)
        self.measurement_model.add_measurement(measurement)
        self._update_measurement_count_label()
        self._sync_viewer_measurement_markers()
        if hasattr(self, "measurement_table"):
            self.measurement_table.resizeColumnsToContents()
            self.measurement_table.scrollToBottom()

    def _clear_measurement_history(self) -> None:
        """Clear all persisted measurements and viewer markers."""
        self._measurement_history.clear()
        self._measurement_marker_angles.clear()
        self._measurement_marker_frames.clear()
        self.measurement_model.clear()
        self._update_measurement_count_label()
        self._sync_viewer_measurement_markers()

    def _rebuild_measurement_table(self) -> None:
        """Rebuild the table model from persisted measurement history.

        This guarantees deterministic multi-point rendering even after view/layout
        changes or repeated frame navigation.
        """
        self.measurement_model = MeasurementTableModel(self)
        if hasattr(self, "measurement_table"):
            self.measurement_table.setModel(self.measurement_model)
        for measurement in self._measurement_history:
            self.measurement_model.add_measurement(measurement)
        self._update_measurement_count_label()
        self._sync_viewer_measurement_markers()
        if hasattr(self, "measurement_table"):
            self.measurement_table.resizeColumnsToContents()
            self.measurement_table.scrollToBottom()

    def _update_measurement_count_label(self) -> None:
        if hasattr(self, "measurement_count_label"):
            self.measurement_count_label.setText(f"Saved Points: {len(self._measurement_history)}")

    def _sync_viewer_measurement_markers(self) -> None:
        if not hasattr(self, "viewer"):
            return
        current_frame_index = self._current_capture_frame_index() if self._active_capture is not None else None
        markers: list[tuple[float, float, str]] = []
        for point_index, (theta_phi, frame_index) in enumerate(
            zip(self._measurement_marker_angles, self._measurement_marker_frames),
            start=1,
        ):
            if frame_index is not None and frame_index != current_frame_index:
                continue
            theta, phi = theta_phi
            # Keep global point numbering stable across frame filters.
            markers.append((theta, phi, f"pt{point_index}"))
        anchor = self._triangulation_anchor
        if anchor is not None and (current_frame_index is None or anchor.frame_index == current_frame_index):
            markers.append((anchor.theta, anchor.phi, "A"))
        for observation in self._triangulation_track_observations:
            if current_frame_index is None or observation.frame_index != current_frame_index:
                continue
            if anchor is not None and observation.frame_index == anchor.frame_index:
                continue
            markers.append((observation.theta, observation.phi, "T"))
        self.viewer.set_measurement_markers(markers)

    def _copy_selected_measurement_cells(self) -> None:
        """Copy selected table cells as TSV for spreadsheet-friendly paste."""
        if not hasattr(self, "measurement_table"):
            return
        indexes = self.measurement_table.selectedIndexes()
        if not indexes:
            return
        indexes.sort(key=lambda idx: (idx.row(), idx.column()))

        by_row: dict[int, dict[int, str]] = {}
        min_col = indexes[0].column()
        max_col = indexes[0].column()
        for index in indexes:
            row = index.row()
            col = index.column()
            min_col = min(min_col, col)
            max_col = max(max_col, col)
            value = index.data(Qt.ItemDataRole.DisplayRole)
            by_row.setdefault(row, {})[col] = "" if value is None else str(value)

        lines: list[str] = []
        for row in sorted(by_row):
            cols = by_row[row]
            line = [cols.get(col, "") for col in range(min_col, max_col + 1)]
            lines.append("\t".join(line))

        QGuiApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage(f"Copied {len(indexes)} cell(s)", 1500)

    def _build_ground_plane_measurement(
        self,
        theta: float,
        phi: float,
        u: int,
        v: int,
        width: int,
        height: int,
    ) -> tuple[PointMeasurement, np.ndarray, str]:
        """Compute a fallback ground-plane coordinate for a selected panorama ray."""
        pose = self._state.metadata.camera_pose
        camera_height_m = max(0.10, float(self.ground_height_spin.value()))
        target_alt_m = pose.altitude - camera_height_m
        pitch_rad = math.radians(self._pose_pitch_deg)
        roll_rad = math.radians(
            self._capture_base_roll_offset_deg + self._pose_roll_deg + self._manual_roll_correction_deg
        )
        east, north, up, depth_val = geometry.intersect_ray_with_altitude_plane(
            theta=theta,
            phi=phi,
            bearing_rad=pose.bearing_rad,
            camera_alt_m=pose.altitude,
            target_alt_m=target_alt_m,
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
        )
        if depth_val > self.MAX_GROUND_PLANE_RANGE_M:
            raise ValueError(
                "Selected point is too close to the horizon for reliable ground-plane intersection."
            )
        lat, lon, alt = geodesy.enu_to_geodetic(
            east,
            north,
            up,
            pose.latitude,
            pose.longitude,
            pose.altitude,
        )
        pixel = PixelSelection(u=u, v=v, width=width, height=height)
        measurement = PointMeasurement(
            pixel=pixel,
            depth_m=depth_val,
            enu_vector=(east, north, up),
            geodetic=(lat, lon, alt),
            quality_score=40.0,
            quality_label=self._quality_label_from_score(40.0),
        )
        local_vec = geometry.spherical_direction(theta, phi) * depth_val
        source = f"Ground plane estimate (camera height {camera_height_m:.2f} m)"
        return measurement, local_vec, source

    def _triangulate_track_observations(
        self,
        observations: list[TriangulationObservation],
    ) -> tuple[multiview.RobustTriangulationResult, float, float]:
        """Triangulate an active feature track using robust multi-view estimation.

        Returns:
            result: Robust point estimate and residual diagnostics.
            median_angle_rad: Median pairwise intersection angle across inlier rays.
            median_baseline_m: Median baseline between inlier camera origins.
        """
        if len(observations) < 2:
            raise ValueError("At least two observations are required for triangulation.")

        origins = np.stack([obs.origin_ecef for obs in observations], axis=0)
        directions = np.stack([obs.direction_ecef for obs in observations], axis=0)
        weights = np.array(
            [
                self._triangulation_pose_weight(
                    horizontal_accuracy_m=obs.horizontal_accuracy_m,
                    heading_accuracy_deg=obs.heading_accuracy_deg,
                )
                for obs in observations
            ],
            dtype=np.float64,
        )
        result = multiview.triangulate_observations_robust(
            origins=origins,
            directions=directions,
            weights=weights,
            ransac_threshold_m=self.TRIANGULATION_RANSAC_THRESHOLD_M,
            huber_scale_m=self.TRIANGULATION_HUBER_SCALE_M,
            min_inliers=2,
        )
        inlier_indices = np.flatnonzero(result.inlier_mask)
        if inlier_indices.size < 2:
            inlier_indices = np.arange(len(observations), dtype=np.int64)

        pair_angles: list[float] = []
        baselines: list[float] = []
        for idx, i in enumerate(inlier_indices):
            for j in inlier_indices[idx + 1 :]:
                pair_angles.append(
                    geometry.acute_angle_between_vectors(
                        observations[int(i)].direction_ecef,
                        observations[int(j)].direction_ecef,
                    )
                )
                baselines.append(
                    float(np.linalg.norm(observations[int(i)].origin_ecef - observations[int(j)].origin_ecef))
                )
        median_angle_rad = float(np.median(pair_angles)) if pair_angles else 0.0
        median_baseline_m = float(np.median(baselines)) if baselines else 0.0
        return result, median_angle_rad, median_baseline_m

    def _triangulation_pose_weight(
        self,
        *,
        horizontal_accuracy_m: Optional[float],
        heading_accuracy_deg: Optional[float],
    ) -> float:
        """Return inverse-variance style confidence weight for one frame pose."""
        hacc = float(horizontal_accuracy_m) if horizontal_accuracy_m is not None else 2.0
        hacc = float(np.clip(hacc, 0.3, 50.0))
        heading_acc = float(heading_accuracy_deg) if heading_accuracy_deg is not None else 3.0
        heading_acc = float(np.clip(heading_acc, 0.5, 45.0))
        sigma_equiv = math.hypot(hacc, heading_acc * 0.1)
        return 1.0 / max(sigma_equiv * sigma_equiv, 1e-6)

    @staticmethod
    def _quality_label_from_score(score: Optional[float]) -> str:
        """Map numeric quality score to label."""
        if score is None or not math.isfinite(score):
            return "N/A"
        if score >= 85.0:
            return "Excellent"
        if score >= 70.0:
            return "Good"
        if score >= 50.0:
            return "Fair"
        return "Poor"

    def _triangulation_quality_score(
        self,
        *,
        baseline_m: float,
        angle_deg: float,
        residual_m: float,
        uncertainty_m: float,
        ray_error_deg: float,
    ) -> float:
        """Compute bounded quality score for triangulated measurement."""
        angle_term = min(1.0, max(0.0, angle_deg / 10.0))
        baseline_term = min(1.0, max(0.0, baseline_m / 15.0))
        residual_term = max(0.0, 1.0 - (residual_m / max(0.5, self.MAX_TRIANGULATION_RESIDUAL_M)))
        sigma_term = max(0.0, 1.0 - (uncertainty_m / max(0.5, self.MAX_TRIANGULATION_SIGMA_M)))
        ray_term = max(0.0, 1.0 - (ray_error_deg / max(0.25, self.MAX_TRIANGULATION_RAY_ERROR_DEG)))
        score = 100.0 * (
            (0.28 * angle_term)
            + (0.20 * baseline_term)
            + (0.22 * residual_term)
            + (0.22 * sigma_term)
            + (0.08 * ray_term)
        )
        return float(np.clip(score, 0.0, 100.0))

    def _estimate_triangulation_uncertainty_m(
        self,
        *,
        observations: list[TriangulationObservation],
        ranges_m: np.ndarray,
        baseline_m: float,
        angle_rad: float,
        image_width: int,
        image_height: int,
    ) -> float:
        """Approximate 1-sigma uncertainty for robust multi-view triangulation.

        This combines three dominant terms:
        1) pixel click quantization on the panorama ray,
        2) heading uncertainty from GNSS/INS metadata (if available),
        3) camera position uncertainty from GNSS horizontal accuracy.
        """
        if baseline_m <= 1e-9 or angle_rad <= 1e-9:
            return float("inf")

        pixel_sigma_theta = 0.5 * (2.0 * math.pi / max(1, image_width))
        pixel_sigma_phi = 0.5 * (math.pi / max(1, image_height))
        pixel_sigma_rad = math.hypot(pixel_sigma_theta, pixel_sigma_phi)

        heading_sigmas = [
            math.radians(obs.heading_accuracy_deg)
            for obs in observations
            if obs.heading_accuracy_deg is not None and obs.heading_accuracy_deg > 0.0
        ]
        heading_sigma_rad = max(heading_sigmas) if heading_sigmas else 0.0
        angular_sigma_rad = math.hypot(pixel_sigma_rad, heading_sigma_rad)

        positive_ranges = [float(value) for value in np.asarray(ranges_m).tolist() if float(value) > 0.0]
        representative_range = float(np.median(positive_ranges)) if positive_ranges else 0.0
        geometry_gain = 1.0 / max(math.sin(angle_rad), 1e-6)
        angular_component_m = representative_range * angular_sigma_rad * geometry_gain

        horizontal_sigmas = [
            obs.horizontal_accuracy_m
            for obs in observations
            if obs.horizontal_accuracy_m is not None and obs.horizontal_accuracy_m > 0.0
        ]
        if horizontal_sigmas:
            pose_component_m = float(np.sqrt(np.mean(np.square(horizontal_sigmas))))
        else:
            pose_component_m = 0.0

        return float(math.hypot(angular_component_m, pose_component_m))

    def _refresh_viewer_image(self, reset_orientation: bool = True) -> None:


        if not self._state.has_image:
            return
        image = self._state.ensure_image()
        self.viewer.set_panorama(image, reset_orientation=reset_orientation)
        self._sync_viewer_measurement_markers()
        self.viewer.update()

    def _update_format_status(self) -> None:
        if not hasattr(self, 'stereo_status_label'):
            return
        detection = self._state.format_detection
        if detection is None:
            self.stereo_status_label.setText('Format status: -')
            return
        status = f"Detected: {detection.format.value} (confidence {detection.confidence:.2f})."
        if detection.notes:
            status += f" {detection.notes}"
        if self._state.distortion_corrected:
            status += ' Distortion corrected.'
        self.stereo_status_label.setText(status)

    def _update_overlay_metadata(self) -> None:
        if not self._state.has_image:
            self.viewer.set_overlay_metadata('')
            return
        detection = self._state.format_detection
        detected = detection.format.value if detection else 'Unknown'
        applied = self._state.format_summary()
        corrected = 'Yes' if self._state.distortion_corrected else 'No'
        overlay = f"Format: {applied} | Detected: {detected} | Distortion corrected: {corrected}"
        self.viewer.set_overlay_metadata(overlay)

    def _on_format_override_changed(self) -> None:
        if not hasattr(self, 'format_combo') or self._format_combo_block:
            return
        fmt = self.format_combo.currentData() or PanoramaInputFormat.AUTO
        try:
            self._state.set_format_override(fmt)
        except ValueError as exc:
            QMessageBox.warning(self, 'Format Override Failed', str(exc))
            self._sync_format_selection()
            return
        self._refresh_viewer_image(reset_orientation=True)
        self.viewer.set_instruction_visible(True)
        self._update_format_status()
        self._update_render_view_controls()
        self._update_overlay_metadata()
        self.statusBar().showMessage(f'Applied format override: {fmt.value}', 2500)

    def _on_distortion_correction_clicked(self) -> None:
        if not self._state.has_image:
            return
        dialog = DistortionCorrectionDialog(self, self._state)
        dialog.previewRequested.connect(self._on_preview_split_views)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog.parameters()
        try:
            self._state.apply_distortion_correction(params)
        except ValueError as exc:
            QMessageBox.warning(self, 'Distortion Correction Failed', str(exc))
            return
        self._refresh_viewer_image(reset_orientation=False)
        self.viewer.set_instruction_visible(False)
        self._update_format_status()
        self._update_render_view_controls()
        self._update_overlay_metadata()
        self.statusBar().showMessage('Distortion correction applied', 3000)

    def _on_preview_split_views(self) -> None:
        left = getattr(self._state, 'left_image', None)
        right = getattr(self._state, 'right_image', None)
        if left is None and right is None:
            QMessageBox.information(self, 'Preview Split Views', 'Split views are not available for this panorama.')
            return
        preview = QDialog(self)
        preview.setWindowTitle('Split View Preview')
        layout = QHBoxLayout(preview)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        for label_text, image in (('Left Eye', left), ('Right Eye', right)):
            if image is None:
                continue
            pixmap = self._numpy_to_pixmap(image)
            image_label = QLabel()
            image_label.setPixmap(
                pixmap.scaled(
                    400,
                    200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            column = QVBoxLayout()
            column.addWidget(QLabel(label_text))
            column.addWidget(image_label)
            container = QWidget()
            container.setLayout(column)
            layout.addWidget(container)
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=preview)
        close_box.rejected.connect(preview.reject)
        layout.addWidget(close_box)
        preview.resize(900, 260)
        preview.exec()

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        if image is None:
            raise ValueError('Image data missing')
        if image.dtype != np.uint8:
            image = image.astype('uint8')
        height, width = image.shape[:2]
        qimage = QImage(image.data, width, height, width * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())

    def _on_invert_x_toggled(self, checked: bool) -> None:
        self.viewer.set_invert_x(checked)

    def _on_invert_y_toggled(self, checked: bool) -> None:
        self.viewer.set_invert_y(checked)

    def _on_anaglyph_toggled(self, checked: bool) -> None:
        if self._syncing_render_view:
            return
        target = 'anaglyph' if checked else 'left'
        try:
            self._state.set_render_eye(target)
        except ValueError as exc:
            QMessageBox.warning(self, "Stereo Mode", str(exc))
            self._syncing_render_view = True
            if hasattr(self, 'anaglyph_checkbox'):
                self.anaglyph_checkbox.blockSignals(True)
                self.anaglyph_checkbox.setChecked(False)
                self.anaglyph_checkbox.blockSignals(False)
            self._syncing_render_view = False
            return
        self._refresh_viewer_image(reset_orientation=False)
        self.viewer.set_instruction_visible(False)
        self._syncing_render_view = True
        if hasattr(self, 'render_view_combo'):
            index = self.render_view_combo.findData(target)
            if index >= 0:
                self.render_view_combo.blockSignals(True)
                self.render_view_combo.setCurrentIndex(index)
                self.render_view_combo.blockSignals(False)
        self._syncing_render_view = False
        self._update_render_view_controls()
        self._update_overlay_metadata()
        message = "Stereo anaglyph preview enabled" if checked else "Stereo anaglyph preview disabled"
        self.statusBar().showMessage(message, 2000)




    def _update_pose_warning(self) -> None:
        if not hasattr(self, "pose_warning_label"):
            return
        pose = self._state.metadata.camera_pose
        defaults = (
            abs(pose.latitude) < 1e-6
            and abs(pose.longitude) < 1e-6
            and abs(pose.altitude) < 1e-3
        )
        if defaults:
            self.pose_warning_label.show()
        else:
            self.pose_warning_label.hide()

    def _update_pose_summary(self) -> None:
        if not hasattr(self, "pose_summary_label"):
            return
        pose = self._state.metadata.camera_pose
        summary = (
            f"Absolute coordinates include lat {pose.latitude:.6f} deg, "
            f"lon {pose.longitude:.6f} deg, alt {pose.altitude:.2f} m, "
            f"bearing {pose.bearing:.2f} deg."
        )
        defaults = (
            abs(pose.latitude) < 1e-6
            and abs(pose.longitude) < 1e-6
            and abs(pose.altitude) < 1e-3
        )
        if defaults:
            summary += " Update the camera pose for real-world accuracy."
        self.pose_summary_label.setText(summary)

    def _update_reset_button_state(self) -> None:
        if not hasattr(self, "reset_view_button"):
            return
        has_image = self._state.has_image
        self.reset_view_button.setEnabled(has_image)
        if not has_image:
            self.reset_view_button.setText("Reset View")
            self.reset_view_button.setToolTip(
                "Snap back to the panorama front and horizon."
            )
            return
        label = "Reset View"
        if self._needs_reorient:
            label += " *"
        self.reset_view_button.setText(label)
        self.reset_view_button.setToolTip(
            "Reset to the panorama front and horizon."
        )

    def _perform_reset_view(self) -> None:
        if not self._state.has_image:
            return
        pose = self._state.metadata.camera_pose
        self.viewer.set_bearing_reference(pose.bearing_rad, recenter=True)
        self.viewer.set_instruction_visible(True)
        self._needs_reorient = False
        self._update_reset_button_state()
        self.statusBar().showMessage("View reset to panorama front/horizon", 2500)

    def _validate_depth_alignment(self) -> None:



        if not (self._state.has_image and self._state.has_depth):



            return



        img_w, img_h = self._state.image_size or (0, 0)



        depth_h, depth_w = self._state.depth.shape[:2]



        if depth_w != img_w or depth_h != img_h:



            QMessageBox.warning(



                self,



                "Dimension Mismatch",



                "Depth map size does not match the panorama view. Results may be inaccurate.",



            )







    # ------------------------------------------------------------------



    # Interaction handling



    def _on_point_hovered(self, theta: float, phi: float) -> None:



        theta_deg = math.degrees(theta)



        phi_deg = math.degrees(phi)



        self.hover_label.setText(f"...={theta_deg:.2f}deg, f={phi_deg:.2f}deg")

    def _on_point_selected(self, theta: float, phi: float) -> None:
        if not self._state.has_image:
            QMessageBox.information(self, "No Panorama", "Load a panorama image before selecting points.")
            return

        image = self._state.ensure_image()
        height, width, _ = image.shape
        u, v = geometry.angles_to_pixel(theta, phi, width, height)
        marker_frame_index = self._current_capture_frame_index()

        effective_mode = self._resolve_measurement_mode()

        if effective_mode == self.MEASUREMENT_MODE_TRIANGULATION:
            self._handle_two_frame_triangulation_selection(theta, phi, u, v, width, height)
            return

        pose = self._state.metadata.camera_pose
        pitch_rad = math.radians(self._pose_pitch_deg)
        roll_rad = math.radians(
            self._capture_base_roll_offset_deg + self._pose_roll_deg + self._manual_roll_correction_deg
        )

        depth_val: float
        east: float
        north: float
        up: float

        if effective_mode == self.MEASUREMENT_MODE_GROUND:
            try:
                measurement, local_vec, source = self._build_ground_plane_measurement(
                    theta,
                    phi,
                    u,
                    v,
                    width,
                    height,
                )
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    "Ground Plane Intersection",
                    f"{exc}\nTry a point likely on the ground, or move to another frame for triangulation.",
                )
                return
            self._append_measurement(
                measurement,
                marker_theta=theta,
                marker_phi=phi,
                marker_frame_index=marker_frame_index,
            )
            self._update_selection_labels(measurement, local_vec)
            self.depth_source_label.setText(source)
            message = "Point measured (ground-plane estimate)"
            duration = 2500
            if self.pose_warning_label.isVisible():
                message += " (default camera pose - update lat/lon/alt for accuracy)"
                duration = 5000
            self.statusBar().showMessage(message, duration)
            return
        else:
            if not self._state.has_depth:
                QMessageBox.warning(
                    self,
                    "No Depth Map",
                    "Depth map unavailable. Automatic mode will use triangulation if capture frames are available.",
                )
                return
            depth = self._state.ensure_depth()
            depth_val = float(depth[v, u])
            if not math.isfinite(depth_val) or depth_val <= 0.0:
                QMessageBox.warning(
                    self,
                    "Invalid Depth",
                    "Selected pixel does not have a valid depth value.",
                )
                return
            east, north, up, _, _ = geometry.enu_vector_from_pixel(
                u,
                v,
                width,
                height,
                depth_val,
                pose.bearing_rad,
                pitch_rad=pitch_rad,
                roll_rad=roll_rad,
            )
            source = self._state.depth_source or "Depth map"

        local_vec = geometry.spherical_direction(theta, phi) * depth_val
        lat, lon, alt = geodesy.enu_to_geodetic(
            east,
            north,
            up,
            pose.latitude,
            pose.longitude,
            pose.altitude,
        )

        pixel = PixelSelection(u=u, v=v, width=width, height=height)
        depth_quality_score = 70.0 if "stereo" in source.lower() else 60.0
        measurement = PointMeasurement(
            pixel=pixel,
            depth_m=depth_val,
            enu_vector=(east, north, up),
            geodetic=(lat, lon, alt),
            quality_score=depth_quality_score,
            quality_label=self._quality_label_from_score(depth_quality_score),
        )

        self._append_measurement(
            measurement,
            marker_theta=theta,
            marker_phi=phi,
            marker_frame_index=marker_frame_index,
        )
        self._update_selection_labels(measurement, local_vec)
        self.depth_source_label.setText(source)

        message = "Point measured"
        duration = 2500
        if effective_mode == self.MEASUREMENT_MODE_GROUND:
            message += " (ground-plane estimate)"
        if self.pose_warning_label.isVisible():
            message += " (default camera pose - update lat/lon/alt for accuracy)"
            duration = 5000
        self.statusBar().showMessage(message, duration)

    def _handle_two_frame_triangulation_selection(
        self,
        theta: float,
        phi: float,
        u: int,
        v: int,
        width: int,
        height: int,
    ) -> None:
        """Handle one click in triangulation mode (multi-view capable)."""
        capture = self._active_capture
        if capture is None or capture.frame_count < 2:
            QMessageBox.warning(
                self,
                "Triangulation",
                "Triangulation requires a loaded capture folder with at least 2 frames.",
            )
            return

        frame_index = self._current_capture_frame_index()
        if frame_index is None:
            QMessageBox.warning(self, "Triangulation", "Unable to resolve active frame index.")
            return

        pose = self._state.metadata.camera_pose
        pose_copy = CameraPose(
            latitude=pose.latitude,
            longitude=pose.longitude,
            altitude=pose.altitude,
            bearing=pose.bearing,
        )
        frame = capture.frames[frame_index]
        pitch_deg = float(self._pose_pitch_deg)
        roll_deg = float(self._capture_base_roll_offset_deg + self._pose_roll_deg + self._manual_roll_correction_deg)

        if not self._frame_heading_usable(frame):
            suggested = self._recommend_triangulation_frames(frame_index, limit=3)
            hint = f" Recommended frames: {', '.join(str(idx) for idx in suggested)}." if suggested else ""
            self.statusBar().showMessage(
                "Current frame has weak heading accuracy for triangulation."
                f"{hint}",
                5000,
            )
            return

        if self._triangulation_anchor is None or frame_index == self._triangulation_anchor.frame_index:
            self._triangulation_anchor = TriangulationAnchor(
                frame_index=frame_index,
                pose=pose_copy,
                pixel_u=u,
                pixel_v=v,
                theta=theta,
                phi=phi,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
            )
            anchor_observation = self._build_triangulation_observation(
                frame_index=frame_index,
                pixel_u=u,
                pixel_v=v,
                theta=theta,
                phi=phi,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                heading_accuracy_deg=frame.heading_accuracy_deg,
                horizontal_accuracy_m=frame.horizontal_accuracy_m,
            )
            self._triangulation_track_observations = [anchor_observation]
            if hasattr(self, "triangulation_label"):
                suggested_frames = self._recommend_triangulation_frames(frame_index, limit=3)
                suggestion_text = (
                    f" Suggested frames: {', '.join(str(idx) for idx in suggested_frames)}."
                    if suggested_frames
                    else ""
                )
                self.triangulation_label.setText(
                    f"Anchor frame {frame_index} set. Step 2: switch frame and click the same feature."
                    f"{suggestion_text}"
                )
            provisional_note = "Coordinate pending second-frame click."
            try:
                provisional_measurement, provisional_vec, _ = self._build_ground_plane_measurement(
                    theta,
                    phi,
                    u,
                    v,
                    width,
                    height,
                )
                provisional_measurement.quality_score = 25.0
                provisional_measurement.quality_label = "Provisional"
                self._append_measurement(
                    provisional_measurement,
                    marker_theta=theta,
                    marker_phi=phi,
                    marker_frame_index=frame_index,
                )
                self._update_selection_labels(provisional_measurement, provisional_vec)
                self.depth_source_label.setText(
                    "Triangulation anchor (provisional estimate; refine with second frame)"
                )
                provisional_note = "Provisional point saved. Switch frame and click same feature to refine."
            except ValueError:
                pending_measurement = PointMeasurement(
                    pixel=PixelSelection(u=u, v=v, width=width, height=height),
                    depth_m=float("nan"),
                    enu_vector=(float("nan"), float("nan"), float("nan")),
                    geodetic=(float("nan"), float("nan"), float("nan")),
                    quality_score=15.0,
                    quality_label="Anchor",
                )
                self._append_measurement(
                    pending_measurement,
                    marker_theta=theta,
                    marker_phi=phi,
                    marker_frame_index=frame_index,
                )
                provisional_note = "Anchor saved. Switch frame and click same feature to compute coordinates."
            self._sync_viewer_measurement_markers()
            suggested_frames = self._recommend_triangulation_frames(frame_index, limit=2)
            suffix = (
                f" Suggested frame(s): {', '.join(str(idx) for idx in suggested_frames)}."
                if suggested_frames
                else ""
            )
            self.statusBar().showMessage(
                f"Triangulation anchor set at frame {frame_index}. {provisional_note}{suffix}",
                4500,
            )
            return

        anchor = self._triangulation_anchor
        assert anchor is not None

        refined_u, refined_v, template_score = self._refine_click_with_template_match(
            anchor_frame_index=anchor.frame_index,
            anchor_u=anchor.pixel_u,
            anchor_v=anchor.pixel_v,
            current_frame_index=frame_index,
            current_u=u,
            current_v=v,
            width=width,
            height=height,
        )
        use_refined = template_score >= self.TRIANGULATION_MIN_TEMPLATE_SCORE
        if use_refined:
            u = refined_u
            v = refined_v
            theta, phi = geometry.pixel_to_angles(u, v, width, height)

        pose_consistency = self._pair_pose_consistency_opencv(
            anchor_frame_index=anchor.frame_index,
            anchor_u=anchor.pixel_u,
            anchor_v=anchor.pixel_v,
            current_frame_index=frame_index,
            current_u=u,
            current_v=v,
            width=width,
            height=height,
        )
        if pose_consistency is not None:
            inlier_ratio, cheirality_ok = pose_consistency
            if inlier_ratio < self.TRIANGULATION_MIN_POSE_INLIER_RATIO or not cheirality_ok:
                self.statusBar().showMessage(
                    "OpenCV pose consistency check failed for this frame pair. "
                    "Choose a farther frame or re-click the same feature.",
                    5000,
                )
                return
        else:
            inlier_ratio = None

        current_observation = self._build_triangulation_observation(
            frame_index=frame_index,
            pixel_u=u,
            pixel_v=v,
            theta=theta,
            phi=phi,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            heading_accuracy_deg=frame.heading_accuracy_deg,
            horizontal_accuracy_m=frame.horizontal_accuracy_m,
        )
        candidate_observations = self._upsert_triangulation_observation(
            self._triangulation_track_observations,
            current_observation,
        )
        if len(candidate_observations) > self.TRIANGULATION_MAX_TRACK_OBSERVATIONS:
            anchor_idx = anchor.frame_index
            anchor_obs = [obs for obs in candidate_observations if obs.frame_index == anchor_idx]
            trailing = [obs for obs in candidate_observations if obs.frame_index != anchor_idx]
            trailing = trailing[-(self.TRIANGULATION_MAX_TRACK_OBSERVATIONS - len(anchor_obs)) :]
            candidate_observations = anchor_obs + trailing

        anchor_obs = next((obs for obs in candidate_observations if obs.frame_index == anchor.frame_index), None)
        current_obs = next((obs for obs in candidate_observations if obs.frame_index == frame_index), None)
        if anchor_obs is None or current_obs is None:
            QMessageBox.warning(self, "Triangulation", "Unable to build triangulation observations.")
            return

        baseline_m = float(np.linalg.norm(current_obs.origin_ecef - anchor_obs.origin_ecef))
        if baseline_m < self.MIN_TRIANGULATION_BASELINE_M:
            self.statusBar().showMessage(
                "Triangulation baseline too small. Move to a farther frame for accurate coordinates.",
                4000,
            )
            return

        try:
            triangulation, intersection_angle_rad, median_baseline_m = self._triangulate_track_observations(
                candidate_observations
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Triangulation", str(exc))
            return

        intersection_angle_deg = math.degrees(intersection_angle_rad)
        if intersection_angle_deg < self.MIN_TRIANGULATION_ANGLE_DEG:
            self.statusBar().showMessage(
                "Triangulation angle is too small for high-accuracy output. "
                "Choose a farther frame with more parallax.",
                4500,
            )
            return

        point_ecef = triangulation.point_ecef
        residual_m = float(triangulation.residual_rms_m)
        ranges_m = triangulation.ranges_m
        positive_count = int(np.count_nonzero(ranges_m > 0.0))
        current_obs_idx = next(
            (idx for idx, obs in enumerate(candidate_observations) if obs.frame_index == frame_index),
            None,
        )
        if current_obs_idx is None:
            QMessageBox.warning(self, "Triangulation", "Triangulation observation index mismatch.")
            return
        range_current_m = float(ranges_m[current_obs_idx])
        if range_current_m <= 0.0:
            self.statusBar().showMessage(
                "Current-frame ray intersects behind the camera. Re-click the same feature in this frame.",
                4500,
            )
            return
        ray_errors_deg = [
            math.degrees(
                geometry.acute_angle_between_vectors(
                    obs.direction_ecef,
                    point_ecef - obs.origin_ecef,
                )
            )
            for obs in candidate_observations
        ]
        max_ray_error_deg = max(ray_errors_deg) if ray_errors_deg else float("inf")
        if max_ray_error_deg > self.HARD_MAX_TRIANGULATION_RAY_ERROR_DEG:
            self.statusBar().showMessage(
                "Feature correspondence is highly inconsistent between frames. "
                "Re-click the same landmark more precisely.",
                5000,
            )
            return

        residual_limit_m = max(self.MAX_TRIANGULATION_RESIDUAL_M, median_baseline_m * 0.35)
        if residual_m > (residual_limit_m * 3.0):
            self.statusBar().showMessage(
                "Triangulation residual is extremely high. "
                "Re-click using a clearer feature and a wider baseline.",
                5000,
            )
            return

        if positive_count < 2:
            try:
                fallback_measurement, fallback_vec, _ = self._build_ground_plane_measurement(
                    theta,
                    phi,
                    u,
                    v,
                    width,
                    height,
                )
                fallback_measurement.quality_score = 30.0
                fallback_measurement.quality_label = "Fallback"
                self._append_measurement(
                    fallback_measurement,
                    marker_theta=theta,
                    marker_phi=phi,
                    marker_frame_index=frame_index,
                )
                self._update_selection_labels(fallback_measurement, fallback_vec)
                self.depth_source_label.setText("Triangulation unstable (fallback ground estimate)")
                self.statusBar().showMessage(
                    "Triangulation unstable for this pair. Showing fallback estimate; choose a farther frame.",
                    4500,
                )
                return
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Triangulation",
                    "Triangulation is unstable for this point. Try a farther frame or click a stronger feature.",
                )
                return

        uncertainty_m = self._estimate_triangulation_uncertainty_m(
            observations=candidate_observations,
            ranges_m=ranges_m,
            baseline_m=baseline_m,
            angle_rad=intersection_angle_rad,
            image_width=width,
            image_height=height,
        )
        if uncertainty_m > self.HARD_MAX_TRIANGULATION_SIGMA_M:
            self.statusBar().showMessage(
                "Triangulation uncertainty is extremely high for this pair. "
                "Use a farther frame and re-click the same feature.",
                5000,
            )
            return

        heading_sigma = np.array(
            [
                obs.heading_accuracy_deg if obs.heading_accuracy_deg is not None else 3.0
                for obs in candidate_observations
            ],
            dtype=np.float64,
        )
        position_sigma = np.array(
            [
                obs.horizontal_accuracy_m if obs.horizontal_accuracy_m is not None else 2.0
                for obs in candidate_observations
            ],
            dtype=np.float64,
        )
        ba_inputs = dict(
            point_seed=point_ecef,
            origins=np.stack([obs.origin_ecef for obs in candidate_observations], axis=0),
            directions=np.stack([obs.direction_ecef for obs in candidate_observations], axis=0),
            weights=np.array(
                [
                    self._triangulation_pose_weight(
                        horizontal_accuracy_m=obs.horizontal_accuracy_m,
                        heading_accuracy_deg=obs.heading_accuracy_deg,
                    )
                    for obs in candidate_observations
                ],
                dtype=np.float64,
            ),
            heading_sigma_deg=heading_sigma,
            position_sigma_m=position_sigma,
        )
        try:
            ba_result = multiview.bundle_adjust_point_with_pose_priors(
                **ba_inputs,
                solver_backend="ceres",
            )
        except Exception as exc:
            logger.warning("Ceres BA failed; falling back to SciPy: {}", exc)
            ba_result = multiview.bundle_adjust_point_with_pose_priors(
                **ba_inputs,
                solver_backend="scipy",
            )
        point_ecef = ba_result.point_ecef
        residual_m = min(residual_m, float(ba_result.residual_rms_m))
        ranges_m = np.sum(
            ba_result.adjusted_directions
            * (point_ecef[None, :] - ba_result.adjusted_origins),
            axis=1,
        )
        range_current_m = float(ranges_m[current_obs_idx])
        if range_current_m <= 0.0:
            self.statusBar().showMessage(
                "Bundle-adjusted solution places the point behind current camera. Re-click the feature.",
                4500,
            )
            return

        lat, lon, alt = geodesy.ecef_to_geodetic(
            float(point_ecef[0]),
            float(point_ecef[1]),
            float(point_ecef[2]),
        )
        current_lat, current_lon, current_alt = self._resolve_frame_position(frame_index)
        east_cur, north_cur, up_cur = geodesy.geodetic_to_enu(
            lat,
            lon,
            alt,
            current_lat,
            current_lon,
            current_alt,
        )

        self._triangulation_track_observations = candidate_observations
        pixel = PixelSelection(u=u, v=v, width=width, height=height)
        triang_quality_score = self._triangulation_quality_score(
            baseline_m=baseline_m,
            angle_deg=intersection_angle_deg,
            residual_m=residual_m,
            uncertainty_m=uncertainty_m,
            ray_error_deg=max_ray_error_deg,
        )
        if use_refined:
            triang_quality_score = float(np.clip(triang_quality_score + (template_score * 4.0), 0.0, 100.0))
        if inlier_ratio is not None:
            triang_quality_score = float(np.clip(triang_quality_score + (12.0 * inlier_ratio), 0.0, 100.0))
        measurement = PointMeasurement(
            pixel=pixel,
            depth_m=range_current_m,
            enu_vector=(east_cur, north_cur, up_cur),
            geodetic=(lat, lon, alt),
            quality_score=triang_quality_score,
            quality_label=self._quality_label_from_score(triang_quality_score),
        )
        local_vec = geometry.spherical_direction(theta, phi) * range_current_m

        self._append_measurement(
            measurement,
            marker_theta=theta,
            marker_phi=phi,
            marker_frame_index=frame_index,
        )
        self._update_selection_labels(measurement, local_vec)
        quality_warnings: list[str] = []
        if residual_m > residual_limit_m:
            quality_warnings.append("high residual")
        if max_ray_error_deg > self.MAX_TRIANGULATION_RAY_ERROR_DEG:
            quality_warnings.append("ray mismatch")
        if uncertainty_m > self.MAX_TRIANGULATION_SIGMA_M:
            quality_warnings.append("high uncertainty")
        if use_refined:
            quality_warnings.append(f"template {template_score:.2f}")
        if inlier_ratio is not None:
            quality_warnings.append(f"pose inliers {inlier_ratio:.2f}")
        quality_warnings.append(f"BA {ba_result.solver_backend}")
        self.depth_source_label.setText(
            "Multi-frame triangulation "
            f"(angle {intersection_angle_deg:.2f} deg, residual {residual_m:.2f} m, "
            f"est. sigma {uncertainty_m:.2f} m, inliers {int(np.count_nonzero(triangulation.inlier_mask))}/"
            f"{len(candidate_observations)}, BA {ba_result.solver_backend}, quality {triang_quality_score:.0f}/100)"
        )
        if hasattr(self, "triangulation_label"):
            self.triangulation_label.setText(
                f"Anchor {anchor.frame_index} active with {len(candidate_observations)} observations. "
                "Click another frame to refine, or click anchor frame to start a new point."
            )

        message = (
            f"Triangulated point from frames {anchor.frame_index}->{frame_index} "
            f"(angle {intersection_angle_deg:.2f} deg, residual {residual_m:.2f} m, sigma~{uncertainty_m:.2f} m)"
        )
        if quality_warnings:
            message += f" [{', '.join(quality_warnings)}]"
        if uncertainty_m > 3.0:
            message += " [high uncertainty]"
        if self.pose_warning_label.isVisible():
            message += " (default camera pose - update lat/lon/alt for accuracy)"
        self.statusBar().showMessage(message, 5000)

    def _update_selection_labels(self, measurement: PointMeasurement, local_vec: np.ndarray) -> None:
        self.pixel_label.setText(f"({measurement.pixel.u}, {measurement.pixel.v})")
        self.depth_label.setText(f"{measurement.depth_m:.2f} m")
        self.vector_label.setText(
            f"X={local_vec[0]:.2f} m, Y={local_vec[1]:.2f} m, Z={local_vec[2]:.2f} m"
        )
        pose = self._state.metadata.camera_pose
        self.geo_label.setText(
            f"{measurement.latitude:.6f}deg, {measurement.longitude:.6f}deg, {measurement.altitude:.2f} m"
            f" (pose lat {pose.latitude:.6f}deg, lon {pose.longitude:.6f}deg, alt {pose.altitude:.2f} m,"
            f" bearing {pose.bearing:.2f}deg)"
        )
        self._update_pose_summary()

        # ------------------------------------------------------------------



        # Exporting and measurements



    def _export_csv(self) -> None:



        if not self.measurement_model.rowCount():



            QMessageBox.information(



                self,



                "No Data",



                "Add at least one measurement before exporting.",



            )



            return



        path, _ = QFileDialog.getSaveFileName(



            self,



            "Export Measurements",



            "measurements.csv",



            "CSV Files (*.csv)",



        )



        if not path:



            return



        try:



            with open(path, "w", newline="", encoding="utf-8") as fh:



                writer = csv.DictWriter(



                    fh,



                    fieldnames=[



                        "pixel_u",



                        "pixel_v",



                        "depth_m",



                        "enu_e",



                        "enu_n",



                        "enu_u",



                        "latitude",



                        "longitude",



                        "altitude",
                        "quality_score",
                        "quality_label",



                    ],



                )



                writer.writeheader()



                for measurement in self.measurement_model.measurements():



                    writer.writerow(measurement.serialize())



            self.statusBar().showMessage(f"Exported CSV ... {path}", 4000)



        except Exception as exc:  # noqa: BLE001



            QMessageBox.critical(self, "Export Failed", f"Could not write CSV: {exc}")







    def _export_json(self) -> None:



        if not self.measurement_model.rowCount():



            QMessageBox.information(



                self,



                "No Data",



                "Add at least one measurement before exporting.",



            )



            return



        path, _ = QFileDialog.getSaveFileName(



            self,



            "Export Measurements",



            "measurements.json",



            "JSON Files (*.json)",



        )



        if not path:



            return



        data = [m.serialize() for m in self.measurement_model.measurements()]



        try:



            with open(path, "w", encoding="utf-8") as fh:



                json.dump(data, fh, indent=2)



            self.statusBar().showMessage(f"Exported JSON ... {path}", 4000)



        except Exception as exc:  # noqa: BLE001



            QMessageBox.critical(self, "Export Failed", f"Could not write JSON: {exc}")







    def _clear_measurements(self) -> None:



        if self.measurement_model.rowCount() == 0:



            return



        reply = QMessageBox.question(



            self,



            "Clear Measurements",



            "Remove all recorded measurements...",



        )



        if reply == QMessageBox.StandardButton.Yes:



            self._clear_measurement_history()
            self._clear_triangulation_anchor(update_label=True)
            if hasattr(self, "viewer"):
                self.viewer.clear_selection_marker()



            self.pixel_label.setText("-")



            self.depth_label.setText("-")



            self.vector_label.setText("-")



            self.geo_label.setText("-")



            self.statusBar().showMessage("Measurements cleared", 2000)




class DistortionCorrectionDialog(QDialog):
    """Dialog that configures fisheye-to-equirectangular conversion parameters."""

    previewRequested = pyqtSignal()

    def __init__(self, parent, state: PanoramaState) -> None:
        super().__init__(parent)
        self.setWindowTitle("Correct Distortion")
        self._state = state
        self._params = FisheyeConversionParams()
        self._build_ui()
        self._apply_defaults()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(6)

        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(90.0, 220.0)
        self.fov_spin.setDecimals(1)
        self.fov_spin.setSuffix(" deg")
        form.addRow("Field of View", self.fov_spin)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1024, 8192)
        self.width_spin.setSingleStep(256)
        form.addRow("Output Width", self.width_spin)

        self.center_x_spin = QDoubleSpinBox()
        self.center_x_spin.setRange(0.0, 1.0)
        self.center_x_spin.setDecimals(4)
        form.addRow("Center X", self.center_x_spin)

        self.center_y_spin = QDoubleSpinBox()
        self.center_y_spin.setRange(0.0, 1.0)
        self.center_y_spin.setDecimals(4)
        form.addRow("Center Y", self.center_y_spin)

        layout.addLayout(form)

        hint = QLabel(
            "Adjust the fisheye parameters, then apply. Use Preview Split Views to inspect the current left/right images." 
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8aa; font-size: 11px;")
        layout.addWidget(hint)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        self.preview_button = QPushButton("Preview Split Views")
        self.preview_button.clicked.connect(self.previewRequested.emit)
        button_row.addWidget(self.preview_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _apply_defaults(self) -> None:
        params = self._state.distortion_params
        self.fov_spin.setValue(params.fov_deg)
        image_size = self._state.image_size
        default_width = params.output_width or (image_size[0] * 2 if image_size else 4096)
        self.width_spin.setValue(int(default_width))
        cx = params.center_x if params.center_x is not None else 0.5
        cy = params.center_y if params.center_y is not None else 0.5
        self.center_x_spin.setValue(cx)
        self.center_y_spin.setValue(cy)
        has_split_views = self._state.left_image is not None or self._state.right_image is not None
        self.preview_button.setEnabled(has_split_views)

    def parameters(self) -> FisheyeConversionParams:
        return FisheyeConversionParams(
            fov_deg=self.fov_spin.value(),
            output_width=self.width_spin.value(),
            center_x=self.center_x_spin.value(),
            center_y=self.center_y_spin.value(),
        )



