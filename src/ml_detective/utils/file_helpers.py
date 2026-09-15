"""
file_helpers.py

Small, GENERIC file-related helper functions.
"""

from pathlib import Path


def get_file_size_mb(file_path: Path) -> float:
    size_in_bytes = file_path.stat().st_size
    size_in_mb = size_in_bytes / (1024 * 1024)
    return round(size_in_mb, 2)


def ensure_directory_exists(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)