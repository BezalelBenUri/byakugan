"""OpenGL-powered panorama viewer widget."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glClear,
    glClearColor,
    glDeleteTextures,
    glDisable,
    glEnable,
    glGenTextures,
    glLoadIdentity,
    glMatrixMode,
    glTexImage2D,
    glTexParameteri,
    glViewport,
)
from OpenGL.GLU import (
    GLU_FILL,
    GLU_INSIDE,
    GLU_SMOOTH,
    gluDeleteQuadric,
    gluLookAt,
    gluNewQuadric,
    gluPerspective,
    gluQuadricDrawStyle,
    gluQuadricNormals,
    gluQuadricOrientation,
    gluQuadricTexture,
    gluSphere,
)
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget


class PanoramaWidget(QOpenGLWidget):
    """Interactive panorama viewer backed by OpenGL."""

    pointSelected = pyqtSignal(float, float)  # theta, phi
    pointHovered = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._image: Optional[np.ndarray] = None
        self._texture_id: Optional[int] = None
        self._pending_upload = False
        self._quadric = None

        self._yaw = 0.0
        self._pitch = 0.0
        self._initial_yaw = 0.0
        self._bearing_reference = 0.0
        self._yaw_offset = math.pi / 2.0
        self._fov_y = math.radians(75.0)
        self._min_fov = math.radians(20.0)
        self._max_fov = math.radians(120.0)
        self._min_pitch = -math.pi / 2 + 1e-3
        self._max_pitch = math.pi / 2 - 1e-3

        self._last_pos = QPointF()
        self._press_pos = QPointF()
        self._dragging = False

        self._mouse_sensitivity = 0.005
        self._min_sensitivity = 0.001
        self._max_sensitivity = 0.01
        self._keyboard_step = math.radians(4.0)

        self._invert_x = False
        self._invert_y = False
        self._last_drag_dx = 0.0
        self._last_drag_dy = 0.0

        self._overlay_metadata = ""

        self._selection_point: Optional[QPointF] = None
        self._instructions_visible = True
        self._instruction_text = (
            "Drag left/right to rotate 360 deg yaw. Drag up/down to tilt. "
            "Scroll to zoom. Press R to reset view. If controls feel unresponsive, check console logs."
        )
        self._show_orientation_overlay = False

        self._update_initial_orientation()

    # ------------------------------------------------------------------
    def set_panorama(self, image: np.ndarray, reset_orientation: bool = True) -> None:
        if image.dtype != np.uint8:
            raise ValueError("Panorama image must be uint8 RGB data")
        if image.shape[2] != 3:
            raise ValueError("Panorama image must be an RGB image")
        self._image = np.ascontiguousarray(image)
        self._pending_upload = True
        if reset_orientation:
            self.reset_view()
        else:
            self.update()

    def clear_panorama(self) -> None:
        self._image = None
        self._delete_texture()
        self.update()

    def set_instruction_text(self, text: str) -> None:
        self._instruction_text = text
        self.update()

    def set_instruction_visible(self, visible: bool) -> None:
        self._instructions_visible = visible
        self.update()

    def set_bearing_reference(self, bearing_rad: float, recenter: bool = False) -> None:
        self._bearing_reference = bearing_rad % (2.0 * math.pi)
        self._update_initial_orientation()
        if recenter:
            self.reset_view()
        else:
            self.update()

    def reset_view(self) -> None:
        self._yaw = self._normalize_angle(self._initial_yaw)
        self._pitch = 0.0
        self._selection_point = None
        self._instructions_visible = True
        self._last_drag_dx = 0.0
        self._last_drag_dy = 0.0
        self.update()

    def reorient_to_bearing(self) -> None:
        self.reset_view()

    # ------------------------------------------------------------------
    def initializeGL(self) -> None:  # noqa: N802
        glClearColor(0.03, 0.04, 0.06, 1.0)
        glEnable(GL_DEPTH_TEST)
        self._quadric = gluNewQuadric()
        gluQuadricTexture(self._quadric, True)
        gluQuadricNormals(self._quadric, GLU_SMOOTH)
        gluQuadricDrawStyle(self._quadric, GLU_FILL)
        gluQuadricOrientation(self._quadric, GLU_INSIDE)

    def resizeGL(self, width: int, height: int) -> None:  # noqa: N802
        glViewport(0, 0, width, height)

    def paintGL(self) -> None:  # noqa: N802
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = max(1e-3, self.width() / max(1, self.height()))
        gluPerspective(math.degrees(self._fov_y), aspect, 0.01, 10.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        dir_x, dir_y, dir_z = self._orientation_vector()
        gluLookAt(0.0, 0.0, 0.0, dir_x, dir_y, dir_z, 0.0, 1.0, 0.0)

        if self._image is None:
            return
        if self._pending_upload:
            self._upload_texture()

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._texture_id or 0)
        gluSphere(self._quadric, 1.0, 128, 64)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._instructions_visible and self._instruction_text:
            painter.setPen(QColor(220, 220, 220))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawText(16, 28, self._instruction_text)

        if self._selection_point is not None:
            pen = QPen(Qt.GlobalColor.white)
            pen.setWidth(2)
            painter.setPen(pen)
            x = self._selection_point.x()
            y = self._selection_point.y()
            painter.drawLine(int(x) - 15, int(y), int(x) + 15, int(y))
            painter.drawLine(int(x), int(y) - 15, int(x), int(y) + 15)

        if self._show_orientation_overlay:
            yaw_deg, pitch_deg = self.orientation_degrees()
            primary = (
                f"Yaw {yaw_deg:+06.1f} deg | Pitch {pitch_deg:+06.1f} deg | "
                f"dx {self._last_drag_dx:+06.1f} | dy {self._last_drag_dy:+06.1f}"
            )
            overlay_lines = [primary]
            if self._overlay_metadata:
                overlay_lines.extend(
                    line.strip() for line in self._overlay_metadata.splitlines() if line.strip()
                )
            metrics = painter.fontMetrics()
            text_width = max((metrics.horizontalAdvance(line) for line in overlay_lines), default=0)
            line_height = metrics.height()
            x = (self.width() - text_width) / 2.0
            y_base = self.height() - metrics.descent() - 12
            painter.setPen(QColor(200, 220, 255))
            for idx, line in enumerate(reversed(overlay_lines)):
                y = y_base - idx * line_height
                painter.drawText(int(x), int(y), line)

        painter.end()

    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position()
            self._last_pos = event.position()
            self._dragging = False
            self._last_drag_dx = 0.0
            self._last_drag_dy = 0.0
            self._hide_instructions()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._dragging = True
            delta = pos - self._last_pos
            raw_dx = float(delta.x())
            raw_dy = float(delta.y())
            self._last_drag_dx = raw_dx
            self._last_drag_dy = raw_dy

            dx = -raw_dx if self._invert_x else raw_dx
            dy = -raw_dy if self._invert_y else raw_dy

            yaw_step = -dx * self._mouse_sensitivity
            pitch_step = -dy * self._mouse_sensitivity

            self._yaw = self._normalize_angle(self._yaw + yaw_step)
            self._pitch = float(
                np.clip(self._pitch + pitch_step, self._min_pitch, self._max_pitch)
            )

            self._hide_instructions()
            self.update()
            logger.debug(
                "Mouse drag dx={:.3f}, dy={:.3f}, yaw={:.3f}, pitch={:.3f}",
                raw_dx,
                raw_dy,
                self._yaw,
                self._pitch,
            )
        else:
            theta, phi = self._angles_from_screen(pos.x(), pos.y())
            self.pointHovered.emit(theta, phi)
        self._last_pos = pos
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            delta = event.position() - self._press_pos
            if delta.manhattanLength() < 6:
                self._selection_point = QPointF(event.position())
                theta, phi = self._angles_from_screen(event.position().x(), event.position().y())
                self.pointSelected.emit(theta, phi)
                self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        degrees = event.angleDelta().y() / 8.0
        steps = degrees / 15.0
        if steps != 0:
            self._hide_instructions()
            factor = math.pow(0.9, steps)
            self._fov_y = float(np.clip(self._fov_y * factor, self._min_fov, self._max_fov))
            self.update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        handled = False
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_A):
            self._yaw = self._normalize_angle(self._yaw - self._keyboard_step)
            handled = True
        elif key in (Qt.Key_Right, Qt.Key_D):
            self._yaw = self._normalize_angle(self._yaw + self._keyboard_step)
            handled = True
        elif key in (Qt.Key_Up, Qt.Key_W):
            self._pitch = float(
                np.clip(self._pitch + self._keyboard_step, self._min_pitch, self._max_pitch)
            )
            handled = True
        elif key in (Qt.Key_Down, Qt.Key_S):
            self._pitch = float(
                np.clip(self._pitch - self._keyboard_step, self._min_pitch, self._max_pitch)
            )
            handled = True
        elif key in (Qt.Key_R, Qt.Key_Home):
            self.reset_view()
            handled = True

        if handled:
            self._hide_instructions()
            self.update()
            event.accept()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    def _angles_from_screen(self, x: float, y: float) -> Tuple[float, float]:
        width = max(1.0, float(self.width()))
        height = max(1.0, float(self.height()))
        x_norm = x / width
        y_norm = y / height

        fov_y = self._fov_y
        aspect = width / height
        fov_x = 2.0 * math.atan(math.tan(fov_y / 2.0) * aspect)

        theta = self._normalize_angle(self._yaw + (x_norm - 0.5) * fov_x)
        phi = self._pitch + (0.5 - y_norm) * fov_y
        phi = float(np.clip(phi, self._min_pitch, self._max_pitch))
        return theta, phi

    def _upload_texture(self) -> None:
        if self._image is None:
            return
        image = self._image
        height, width, _ = image.shape
        texture_id = self._texture_id or glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            width,
            height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            image,
        )
        glBindTexture(GL_TEXTURE_2D, 0)
        self._texture_id = texture_id
        self._pending_upload = False

    def _delete_texture(self) -> None:
        if self._texture_id is not None:
            glDeleteTextures([self._texture_id])
            self._texture_id = None

    def closeEvent(self, event) -> None:  # noqa: N802
        self._delete_texture()
        if self._quadric is not None:
            gluDeleteQuadric(self._quadric)
            self._quadric = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    def _normalize_angle(self, value: float) -> float:
        two_pi = 2.0 * math.pi
        value %= two_pi
        if value < 0.0:
            value += two_pi
        return value

    def set_orientation_overlay_enabled(self, enabled: bool) -> None:
        self._show_orientation_overlay = enabled
        self.update()

    def orientation_degrees(self) -> Tuple[float, float]:
        yaw_deg = math.degrees(self._yaw)
        if yaw_deg > 180.0:
            yaw_deg -= 360.0
        pitch_deg = math.degrees(self._pitch)
        return yaw_deg, pitch_deg

    def _orientation_vector(self) -> Tuple[float, float, float]:
        yaw = self._yaw + self._yaw_offset
        cos_pitch = math.cos(self._pitch)
        return (
            -math.sin(yaw) * cos_pitch,
            math.sin(self._pitch),
            -math.cos(yaw) * cos_pitch,
        )

    def _update_initial_orientation(self) -> None:
        self._initial_yaw = self._normalize_angle(-self._bearing_reference)

    def _hide_instructions(self) -> None:
        if self._instructions_visible:
            self._instructions_visible = False
            self.update()

    # Control hooks ----------------------------------------------------
    def set_mouse_sensitivity(self, sensitivity: float) -> None:
        sensitivity = float(np.clip(sensitivity, self._min_sensitivity, self._max_sensitivity))
        self._mouse_sensitivity = sensitivity
        logger.debug("Mouse sensitivity set to %.4f", self._mouse_sensitivity)

    def mouse_sensitivity(self) -> float:
        return self._mouse_sensitivity

    def set_invert_x(self, enabled: bool) -> None:
        self._invert_x = bool(enabled)
        logger.debug("Invert X set to %s", self._invert_x)

    def set_invert_y(self, enabled: bool) -> None:
        self._invert_y = bool(enabled)
        logger.debug("Invert Y set to %s", self._invert_y)

    def set_overlay_metadata(self, metadata: str) -> None:
        self._overlay_metadata = metadata.strip()
        if self._show_orientation_overlay:
            self.update()

