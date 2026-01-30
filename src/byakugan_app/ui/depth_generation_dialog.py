"""Dialog for configuring depth generation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QDoubleSpinBox,
)

from ..io.depth_utils import StereoDetectionResult, StereoFormat


class DepthGenerationMode(Enum):
    PANORAMA_STEREO = "panorama_stereo"
    EXTERNAL_STEREO = "external_stereo"


@dataclass(slots=True)
class DepthGenerationRequest:
    """Container for dialog selections."""

    mode: DepthGenerationMode
    stereo_format: Optional[StereoFormat]
    baseline_m: float
    focal_length_px: float
    rectify: bool
    downsample_factor: int
    output_path: Optional[Path]
    left_path: Optional[Path] = None
    right_path: Optional[Path] = None


class DepthGenerationDialog(QDialog):
    """Dialog allowing the user to configure stereo depth generation."""

    def __init__(
        self,
        detection: Optional[StereoDetectionResult],
        default_focal_px: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Depth Map")
        self.detection = detection
        self.default_focal_px = default_focal_px

        self._build_ui()
        self._configure_defaults()
        self._wire_events()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Mode selection
        self.mode_group = QButtonGroup(self)
        self.panorama_radio = QRadioButton("Stereo from Panorama (auto-split if detected)")
        self.external_radio = QRadioButton("Stereo from External Left/Right Files")
        self.mode_group.addButton(self.panorama_radio)
        self.mode_group.addButton(self.external_radio)

        layout.addWidget(self.panorama_radio)
        layout.addWidget(self.external_radio)

        # Panorama options -------------------------------------------------
        pano_group = QGroupBox("Panorama Stereo Options")
        pano_layout = QFormLayout(pano_group)
        pano_layout.setSpacing(6)

        self.format_combo = QComboBox()
        self.format_combo.setToolTip(
            "Override detected stereo format. Top-Bottom: upper half left eye, lower half right eye."
        )
        pano_layout.addRow("Format", self.format_combo)

        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setDecimals(3)
        self.baseline_spin.setRange(0.001, 0.5)
        self.baseline_spin.setSuffix(" m")
        pano_layout.addRow("Baseline", self.baseline_spin)

        self.focal_spin = QDoubleSpinBox()
        self.focal_spin.setDecimals(1)
        self.focal_spin.setRange(10.0, 5000.0)
        self.focal_spin.setSuffix(" px")
        pano_layout.addRow("Focal Length", self.focal_spin)

        self.rectify_checkbox = QCheckBox("Rectify views before matching")
        self.rectify_checkbox.setToolTip("Experimental: attempts to smooth disparity seams for better consistency.")
        pano_layout.addRow(self.rectify_checkbox)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 4)
        self.downsample_spin.setToolTip("Downsample before matching to speed up processing, then upscale the depth map.")
        pano_layout.addRow("Downsample", self.downsample_spin)

        self.save_checkbox = QCheckBox("Save generated depth to file")
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setEnabled(False)
        self.save_browse_btn = QPushButton("Browse...")
        self.save_browse_btn.setEnabled(False)

        save_layout = QHBoxLayout()
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(self.save_browse_btn)
        pano_layout.addRow(self.save_checkbox)
        pano_layout.addRow("Save Path", save_layout)

        layout.addWidget(pano_group)

        # External stereo --------------------------------------------------
        external_group = QGroupBox("External Stereo Files")
        ext_layout = QGridLayout(external_group)
        ext_layout.setContentsMargins(6, 10, 6, 10)
        ext_layout.setHorizontalSpacing(6)
        ext_layout.setVerticalSpacing(6)

        self.left_path_edit = QLineEdit()
        self.right_path_edit = QLineEdit()
        self.left_browse_btn = QPushButton("Browse...")
        self.right_browse_btn = QPushButton("Browse...")

        ext_layout.addWidget(QLabel("Left Image"), 0, 0)
        ext_layout.addWidget(self.left_path_edit, 0, 1)
        ext_layout.addWidget(self.left_browse_btn, 0, 2)
        ext_layout.addWidget(QLabel("Right Image"), 1, 0)
        ext_layout.addWidget(self.right_path_edit, 1, 1)
        ext_layout.addWidget(self.right_browse_btn, 1, 2)
        ext_layout.addWidget(QLabel("Baseline"), 2, 0)
        self.external_baseline_spin = QDoubleSpinBox()
        self.external_baseline_spin.setDecimals(3)
        self.external_baseline_spin.setRange(0.001, 0.5)
        self.external_baseline_spin.setSuffix(" m")
        ext_layout.addWidget(self.external_baseline_spin, 2, 1)
        ext_layout.addWidget(QLabel("Focal Length"), 3, 0)
        self.external_focal_spin = QDoubleSpinBox()
        self.external_focal_spin.setDecimals(1)
        self.external_focal_spin.setRange(10.0, 5000.0)
        self.external_focal_spin.setSuffix(" px")
        ext_layout.addWidget(self.external_focal_spin, 3, 1)

        layout.addWidget(external_group)

        # Buttons ----------------------------------------------------------
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.button_box)

        self.pano_group = pano_group
        self.external_group = external_group

    # ------------------------------------------------------------------
    def _configure_defaults(self) -> None:
        detection = self.detection
        if detection and detection.is_stereo:
            self.status_label.setText(
                f"Stereo panorama detected: {detection.format.value} (confidence {detection.confidence:.2f}).\n{detection.message}"
            )
            self.panorama_radio.setChecked(True)
        else:
            self.status_label.setText(
                "Stereo format not confirmed. Select a layout manually or provide external stereo files."
            )
            self.external_radio.setChecked(True)
        self.panorama_radio.setEnabled(True)

        self.format_combo.addItem("Auto (use detection)", userData=None)
        self.format_combo.addItem(StereoFormat.TOP_BOTTOM.value, StereoFormat.TOP_BOTTOM)
        self.format_combo.addItem(StereoFormat.SIDE_BY_SIDE.value, StereoFormat.SIDE_BY_SIDE)
        self.format_combo.addItem("Mono", StereoFormat.MONO)

        self.baseline_spin.setValue(0.06)
        self.external_baseline_spin.setValue(0.06)

        self.focal_spin.setValue(self.default_focal_px)
        self.external_focal_spin.setValue(self.default_focal_px)

        self.downsample_spin.setValue(1)

    def _wire_events(self) -> None:
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        self.mode_group.buttonToggled.connect(self._refresh_mode_state)
        self.save_checkbox.toggled.connect(self._toggle_save_controls)
        self.save_browse_btn.clicked.connect(self._on_browse_save)
        self.left_browse_btn.clicked.connect(lambda: self._browse_path(self.left_path_edit))
        self.right_browse_btn.clicked.connect(lambda: self._browse_path(self.right_path_edit))

        self._refresh_mode_state()

    # ------------------------------------------------------------------
    def _refresh_mode_state(self) -> None:
        use_panorama = self.panorama_radio.isChecked()
        self.pano_group.setEnabled(use_panorama)
        self.external_group.setEnabled(not use_panorama)

    def _toggle_save_controls(self, checked: bool) -> None:
        self.save_path_edit.setEnabled(checked)
        self.save_browse_btn.setEnabled(checked)

    def _on_browse_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Depth Map", "depth.png", "Depth Map (*.png *.tif *.tiff)")
        if path:
            self.save_path_edit.setText(path)

    def _browse_path(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            target.setText(path)

    # ------------------------------------------------------------------
    def _on_accept(self) -> None:
        request = self.build_request()
        if request is None:
            return
        self._request = request
        self.accept()

    def build_request(self) -> Optional[DepthGenerationRequest]:
        if self.panorama_radio.isChecked():
            fmt = self.format_combo.currentData()
            if fmt is None and self.detection:
                fmt = self.detection.format
            if fmt is None or fmt == StereoFormat.MONO:
                QMessageBox.warning(self, "Stereo Required", "Select a stereo format before generating depth from the panorama.")
                return None
            output_path = Path(self.save_path_edit.text()) if (self.save_checkbox.isChecked() and self.save_path_edit.text()) else None
            return DepthGenerationRequest(
                mode=DepthGenerationMode.PANORAMA_STEREO,
                stereo_format=fmt,
                baseline_m=self.baseline_spin.value(),
                focal_length_px=self.focal_spin.value(),
                rectify=self.rectify_checkbox.isChecked(),
                downsample_factor=self.downsample_spin.value(),
                output_path=output_path,
            )

        left_path = self.left_path_edit.text().strip()
        right_path = self.right_path_edit.text().strip()
        if not left_path or not right_path:
            QMessageBox.warning(self, "Missing Files", "Select both left and right images for external stereo matching.")
            return None

        return DepthGenerationRequest(
            mode=DepthGenerationMode.EXTERNAL_STEREO,
            stereo_format=None,
            baseline_m=self.external_baseline_spin.value(),
            focal_length_px=self.external_focal_spin.value(),
            rectify=False,
            downsample_factor=1,
            output_path=None,
            left_path=Path(left_path),
            right_path=Path(right_path),
        )

    # ------------------------------------------------------------------
    def request(self) -> DepthGenerationRequest:
        return getattr(self, "_request")
