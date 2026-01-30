"""Main application window."""



from __future__ import annotations







import csv



import json



import math



from pathlib import Path



from typing import Optional







import cv2



import numpy as np



from loguru import logger



from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QImage, QPixmap



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




from ..io.panorama_processing import FisheyeConversionParams, PanoramaInputFormat
from ..math import geodesy, geometry



from ..models.camera_pose import CameraPose



from ..models.panorama_state import PanoramaState



from ..models.selection import PixelSelection, PointMeasurement



from ..viewer.panorama_widget import PanoramaWidget



from ..workers.task_runner import FunctionTask, TaskRunner



from .depth_generation_dialog import DepthGenerationDialog, DepthGenerationMode



from .measurement_table_model import MeasurementTableModel











class MainWindow(QMainWindow):



    """Primary window containing the viewer and controls."""







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







        layout.addWidget(self._build_camera_group())



        layout.addWidget(self._build_data_group())



        layout.addWidget(self._build_viewer_controls_group())



        layout.addWidget(self._build_selection_group())



        layout.addWidget(self._build_measurements_group(), stretch=1)







        layout.addStretch(1)



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
        load_depth_btn = QPushButton("Load Depth Map...")
        generate_depth_btn = QPushButton("Generate Depth Map...")
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.setEnabled(False)
        self.reset_view_button.setToolTip("Snap back to the horizon/front using the current camera bearing.")

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
        load_depth_btn.clicked.connect(self._on_load_depth_clicked)
        generate_depth_btn.clicked.connect(self._on_generate_depth_clicked)

        def _reset_view_handler():
            self._perform_reset_view()

        self._reset_view_handler = _reset_view_handler
        self.reset_view_button.clicked.connect(_reset_view_handler)

        vbox.addWidget(load_image_btn)
        vbox.addWidget(self.image_path_display)
        vbox.addSpacing(4)
        vbox.addWidget(load_depth_btn)
        vbox.addWidget(self.depth_path_display)
        vbox.addSpacing(4)
        vbox.addWidget(generate_depth_btn)
        vbox.addWidget(self.reset_view_button)

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

        vbox.addSpacing(6)
        vbox.addWidget(self.navigation_hint_label)
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

        hint_label = QLabel("If rotation feels wrong, adjust sensitivity or invert axes.")
        hint_label.setStyleSheet("color: #8aa; font-size: 11px;")
        hint_label.setWordWrap(True)
        vbox.addWidget(hint_label)

        return group



    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("Selection Readout")
        form = QFormLayout(group)

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

        form.addRow("Hover", self.hover_label)
        form.addRow("Pixel", self.pixel_label)
        form.addRow("Depth", self.depth_label)
        form.addRow("Local XYZ", self.vector_label)
        form.addRow("Lat/Lon/Alt", self.geo_label)
        form.addRow("Depth Source", self.depth_source_label)
        form.addRow("Pose Context", self.pose_summary_label)
        return group



    # Measurements -----------------------------------------------------------



    def _build_measurements_group(self) -> QGroupBox:



        group = QGroupBox("Saved Measurements")



        vbox = QVBoxLayout(group)



        self.measurement_table = QTableView()



        self.measurement_table.setModel(self.measurement_model)



        self.measurement_table.horizontalHeader().setStretchLastSection(True)



        self.measurement_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)



        self.measurement_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)



        self.measurement_table.setAlternatingRowColors(True)



        self.measurement_table.verticalHeader().setVisible(False)







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







        vbox.addWidget(self.measurement_table, stretch=1)



        vbox.addLayout(controls)



        return group







    # Menu -------------------------------------------------------------------



    def _create_menu_bar(self) -> None:



        bar = self.menuBar()



        file_menu = bar.addMenu("File")







        load_action = QAction("Load Panorama...", self)



        load_action.triggered.connect(self._on_load_panorama_clicked)



        file_menu.addAction(load_action)







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
        prev_pose = self._state.metadata.camera_pose
        prev_bearing = prev_pose.bearing if prev_pose is not None else 0.0
        pose = CameraPose(
            latitude=self.lat_field.value(),
            longitude=self.lon_field.value(),
            altitude=self.alt_field.value(),
            bearing=self.bearing_field.value() % 360.0,
        )
        self._state.metadata.camera_pose = pose
        self.viewer.set_bearing_reference(pose.bearing_rad)
        if self._state.has_image:
            bearing_delta = abs(((pose.bearing - prev_bearing + 180.0) % 360.0) - 180.0)
            if bearing_delta > 1e-6:
                self._needs_reorient = True
        self._update_pose_warning()
        self._update_pose_summary()
        self._update_reset_button_state()
        logger.debug("Camera pose updated: {}", pose)


    # ------------------------------------------------------------------



    # Data loading



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



        self.measurement_model.clear()



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
        self.measurement_model.clear()
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
            QTimer.singleShot(250, self._prompt_stereo_depth)

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

    def _update_depth_source_label(self) -> None:
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
    def _refresh_viewer_image(self, reset_orientation: bool = True) -> None:


        if not self._state.has_image:
            return
        image = self._state.ensure_image()
        self.viewer.set_panorama(image, reset_orientation=reset_orientation)
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
        value = max(1, min(int(value), 10))
        sensitivity = value / 1000.0
        if hasattr(self, 'mouse_sensitivity_value_label'):
            self.mouse_sensitivity_value_label.setText(f"{sensitivity:.3f}")
        self.viewer.set_mouse_sensitivity(sensitivity)

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
                "Snap back to the horizon/front using the current camera bearing."
            )
            return
        label = "Reset View"
        if self._needs_reorient:
            label += " *"
        self.reset_view_button.setText(label)
        pose = self._state.metadata.camera_pose
        self.reset_view_button.setToolTip(
            f"Reset to the camera front/horizon (bearing {pose.bearing:.2f} deg)."
        )

    def _perform_reset_view(self) -> None:
        if not self._state.has_image:
            return
        pose = self._state.metadata.camera_pose
        self.viewer.set_bearing_reference(pose.bearing_rad, recenter=True)
        self.viewer.set_instruction_visible(True)
        self._needs_reorient = False
        self._update_reset_button_state()
        self.statusBar().showMessage("View reset to front/horizon", 2500)

    def _prompt_stereo_depth(self) -> None:



        detection = self._state.stereo_detection



        if not (detection and detection.is_stereo) or self._state.has_depth:



            return



        reply = QMessageBox.question(



            self,



            "Stereo Panorama Detected",



            "A stereo panorama was detected. Would you like to generate a depth map now...",



        )



        if reply == QMessageBox.StandardButton.Yes:



            self._on_generate_depth_clicked()







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



        if not self._state.has_depth:



            QMessageBox.warning(self, "No Depth Map", "Load or generate a depth map to enable measurements.")



            return







        image = self._state.ensure_image()



        depth = self._state.ensure_depth()



        height, width, _ = image.shape



        u, v = geometry.angles_to_pixel(theta, phi, width, height)







        depth_val = float(depth[v, u])



        if not math.isfinite(depth_val) or depth_val <= 0.0:



            QMessageBox.warning(self, "Invalid Depth", "Selected pixel does not have a valid depth value.")



            return







        pose = self._state.metadata.camera_pose



        east, north, up, _, _ = geometry.enu_vector_from_pixel(



            u,



            v,



            width,



            height,



            depth_val,



            pose.bearing_rad,



        )



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



        measurement = PointMeasurement(



            pixel=pixel,



            depth_m=depth_val,



            enu_vector=(east, north, up),



            geodetic=(lat, lon, alt),



        )







        self.measurement_model.add_measurement(measurement)



        self._update_selection_labels(measurement, local_vec)



        message = "Point measured"
        duration = 2500
        if self.pose_warning_label.isVisible():
            message += " (default camera pose - update lat/lon/alt for accuracy)"
            duration = 5000
        self.statusBar().showMessage(message, duration)







    def _update_selection_labels(self, measurement: PointMeasurement, local_vec: np.ndarray) -> None:
        self.pixel_label.setText(f"({measurement.pixel.u}, {measurement.pixel.v})")
        self.depth_label.setText(f"{measurement.depth_m:.2f} m")
        self.vector_label.setText(
            f"X={local_vec[0]:.2f} m, Y={local_vec[1]:.2f} m, Z={local_vec[2]:.2f} m"
        )
        pose = self._state.metadata.camera_pose
        self.geo_label.setText(
            f"{measurement.latitude:.7f}deg, {measurement.longitude:.7f}deg, {measurement.altitude:.2f} m"
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



            self.measurement_model.clear()



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



