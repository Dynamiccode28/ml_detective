"""
profiler.py

Computes the DatasetProfile: neutral, descriptive facts about a dataset.
"""

import math

import numpy as np
import pandas as pd

from ml_detective.ingestion.task_detector import TaskType, detect_task_type
from ml_detective.profiling.models import ColumnProfile, DatasetProfile, TargetProfile
from ml_detective.utils.logger import get_logger
from ml_detective.utils.timing import timeit

logger = get_logger(__name__)


def _profile_column(dataframe: pd.DataFrame, column_name: str) -> ColumnProfile:
    series = dataframe[column_name]
    total_rows = len(series)

    unique_count = int(series.nunique(dropna=True))
    cardinality_ratio = round(unique_count / total_rows, 4) if total_rows > 0 else 0.0
    missing_pct = round(series.isna().mean() * 100, 2)

    sparsity_pct = None
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            sparsity_pct = round((non_null == 0).mean() * 100, 2)

    return ColumnProfile(
        column_name=column_name,
        dtype=str(series.dtype),
        unique_count=unique_count,
        cardinality_ratio=cardinality_ratio,
        missing_pct=missing_pct,
        sparsity_pct=sparsity_pct,
    )


def _calculate_entropy(class_counts: pd.Series) -> float:
    total = class_counts.sum()
    probabilities = class_counts / total
    return round(-sum(p * math.log2(p) for p in probabilities if p > 0), 4)


def _profile_target(dataframe: pd.DataFrame, target_column: str) -> TargetProfile:
    task_type = detect_task_type(dataframe, target_column)
    target_series = dataframe[target_column].dropna()

    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        class_counts = target_series.value_counts()
        imbalance_ratio = round(class_counts.max() / class_counts.min(), 2)
        entropy = _calculate_entropy(class_counts)

        return TargetProfile(
            task_type=task_type.value,
            class_distribution=class_counts.to_dict(),
            imbalance_ratio=imbalance_ratio,
            entropy=entropy,
            summary_stats=None,
        )

    if task_type == TaskType.REGRESSION:
        return TargetProfile(
            task_type=task_type.value,
            class_distribution=None,
            imbalance_ratio=None,
            entropy=None,
            summary_stats={
                "mean": round(target_series.mean(), 4),
                "std": round(target_series.std(), 4),
                "min": round(target_series.min(), 4),
                "max": round(target_series.max(), 4),
            },
        )

    return TargetProfile(
        task_type=task_type.value,
        class_distribution=None,
        imbalance_ratio=None,
        entropy=None,
        summary_stats=None,
    )


@timeit
def profile_dataset(dataframe: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
    logger.info("Profiling dataset...")

    n_rows, n_cols = dataframe.shape
    memory_usage_mb = round(dataframe.memory_usage(deep=True).sum() / (1024 * 1024), 4)

    column_profiles = {
        column_name: _profile_column(dataframe, column_name)
        for column_name in dataframe.columns
    }

    target_profile = None
    if target_column is not None:
        target_profile = _profile_target(dataframe, target_column)

    logger.info(f"Profiling complete: {n_rows} rows, {n_cols} columns, {memory_usage_mb} MB.")

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        memory_usage_mb=memory_usage_mb,
        column_profiles=column_profiles,
        target_profile=target_profile,
    )