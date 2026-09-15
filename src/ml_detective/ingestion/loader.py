"""
loader.py

Handles turning a file on disk into a pandas DataFrame.
"""

from pathlib import Path

import pandas as pd

from ml_detective.utils.exceptions import EmptyDatasetError, UnsupportedFileTypeError
from ml_detective.utils.logger import get_logger
from ml_detective.utils.timing import timeit

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".csv"}


def load_csv(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{file_path.suffix}'. "
            f"Only {_SUPPORTED_EXTENSIONS} files are currently supported."
        )

    logger.info(f"Loading CSV file: {file_path}")

    dataframe = _read_csv_safely(file_path)

    if dataframe.shape[0] == 0:
        raise EmptyDatasetError(
            f"The file '{file_path.name}' has column headers but zero data rows."
        )

    logger.info(
        f"Loaded '{file_path.name}' successfully: "
        f"{dataframe.shape[0]} rows, {dataframe.shape[1]} columns."
    )
    return dataframe


@timeit
def _read_csv_safely(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise EmptyDatasetError(f"The file '{file_path.name}' is completely empty.") from error
    except pd.errors.ParserError as error:
        raise UnsupportedFileTypeError(
            f"The file '{file_path.name}' could not be parsed as a valid CSV. "
            f"It may be corrupted or actually be a different file type."
        ) from error