"""
task_detector.py

Guesses the target column and detects the task type (regression,
binary classification, multiclass classification).
"""

from enum import Enum

import pandas as pd

from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

_COMMON_TARGET_NAMES = {
    "target", "label", "class", "y", "outcome",
    "churned", "churn", "survived", "default", "fraud",
    "is_fraud", "price", "output", "result",
}

_MAX_UNIQUE_VALUES_FOR_DISGUISED_CATEGORY = 20


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    UNKNOWN = "unknown"


def guess_target_column(dataframe: pd.DataFrame) -> str | None:
    for column_name in dataframe.columns:
        if column_name.strip().lower() in _COMMON_TARGET_NAMES:
            logger.info(f"Guessed target column '{column_name}' by matching a common name.")
            return column_name

    if len(dataframe.columns) > 0:
        last_column = dataframe.columns[-1]
        logger.info(
            f"No common target name matched. Falling back to the LAST "
            f"column '{last_column}' as a guess."
        )
        return last_column

    return None


def detect_task_type(dataframe: pd.DataFrame, target_column: str) -> TaskType:
    target_series = dataframe[target_column].dropna()
    number_of_unique_values = target_series.nunique()
    number_of_rows = len(target_series)

    is_text_type = pd.api.types.is_string_dtype(target_series) or isinstance(
        target_series.dtype, pd.CategoricalDtype
    )

    if is_text_type:
        if number_of_unique_values == 2:
            return TaskType.BINARY_CLASSIFICATION
        elif number_of_unique_values > 2:
            return TaskType.MULTICLASS_CLASSIFICATION
        else:
            logger.warning(
                f"Target column '{target_column}' has only "
                f"{number_of_unique_values} unique value(s). Cannot determine task type."
            )
            return TaskType.UNKNOWN

    all_values_are_whole_numbers = (target_series == target_series.round(0)).all()

    if number_of_unique_values == 2 and all_values_are_whole_numbers:
        return TaskType.BINARY_CLASSIFICATION

    has_repetition = number_of_unique_values < number_of_rows

    if (
        number_of_unique_values <= _MAX_UNIQUE_VALUES_FOR_DISGUISED_CATEGORY
        and has_repetition
        and all_values_are_whole_numbers
    ):
        return TaskType.MULTICLASS_CLASSIFICATION

    return TaskType.REGRESSION