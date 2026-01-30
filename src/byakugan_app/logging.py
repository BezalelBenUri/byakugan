"""Logging configuration helpers."""
from __future__ import annotations

from loguru import logger


def configure_logging() -> None:
    """Configure loguru with a friendly default sink."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
