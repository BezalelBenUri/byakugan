"""Qt model for displaying measured points."""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt

from ..models.selection import PointMeasurement


class MeasurementTableModel(QAbstractTableModel):
    """Model backing the measurement table."""

    HEADERS = [
        "Pixel U",
        "Pixel V",
        "Depth (m)",
        "East (m)",
        "North (m)",
        "Up (m)",
        "Latitude",
        "Longitude",
        "Altitude",
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: List[PointMeasurement] = []

    # Qt Model API ---------------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802,E501
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid() or role not in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.TextAlignmentRole}:
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return None
            return None

        measurement = self._rows[index.row()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        mapping = {
            0: measurement.pixel.u,
            1: measurement.pixel.v,
            2: f"{measurement.depth_m:.2f}",
            3: f"{measurement.enu_vector[0]:.2f}",
            4: f"{measurement.enu_vector[1]:.2f}",
            5: f"{measurement.enu_vector[2]:.2f}",
            6: f"{measurement.latitude:.7f}",
            7: f"{measurement.longitude:.7f}",
            8: f"{measurement.altitude:.2f}",
        }
        return mapping.get(index.column(), "")

    # Mutators --------------------------------------------------------------
    def add_measurement(self, measurement: PointMeasurement) -> None:
        """Append a measurement to the table."""
        next_row = len(self._rows)
        self.beginInsertRows(QModelIndex(), next_row, next_row)
        self._rows.append(measurement)
        self.endInsertRows()

    def clear(self) -> None:
        """Remove all measurements."""
        self.beginResetModel()
        self._rows.clear()
        self.endResetModel()

    def measurements(self) -> List[PointMeasurement]:
        return list(self._rows)

    def measurement_at(self, row: int) -> Optional[PointMeasurement]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None
