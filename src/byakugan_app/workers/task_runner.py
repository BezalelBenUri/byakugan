"""Utilities for running functions in background threads."""
from __future__ import annotations

import traceback
from typing import Any, Callable

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal


class TaskSignals(QObject):
    """Signals available from a background task."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class FunctionTask(QRunnable):
    """Wrap a callable for execution in the Qt thread pool."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.signals.failed.emit(f"{exc}\n{tb}")
        else:
            self.signals.finished.emit(result)


class TaskRunner:
    """Thin wrapper around QThreadPool for convenience."""

    def __init__(self, max_threads: int | None = None) -> None:
        self._pool = QThreadPool.globalInstance()
        if max_threads is not None:
            self._pool.setMaxThreadCount(max_threads)

    def submit(self, task: FunctionTask) -> None:
        self._pool.start(task)
