"""
logger.py

Sets up ONE central logging configuration that every other file in the
project should use.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ml_detective.config.settings import settings

_LOGS_DIR = Path("logs")
_LOGS_DIR.mkdir(exist_ok=True)


def get_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)

    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level.upper())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        filename=_LOGS_DIR / "ml_detective.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger