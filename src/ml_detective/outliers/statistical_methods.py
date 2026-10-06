"""
statistical_methods.py

Three simple, fast, single-column outlier detection methods.
"""

import numpy as np
import pandas as pd

from ml_detective.config.thresholds import get_thresholds
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def detect_outliers_zscore(series: pd.Series) -> set[int]:
    threshold = get_thresholds()["outliers"]["z_score_threshold"]
    clean_series = series.dropna()

    if clean_series.std() == 0 or len(clean_series) < 2:
        return set()

    z_scores = (clean_series - clean_series.mean()) / clean_series.std()
    return set(clean_series[z_scores.abs() > threshold].index)


def detect_outliers_modified_zscore(series: pd.Series) -> set[int]:
    threshold = get_thresholds()["outliers"]["modified_z_score_threshold"]
    clean_series = series.dropna()

    median = clean_series.median()
    mad = (clean_series - median).abs().median()

    if mad == 0 or len(clean_series) < 2:
        return set()

    modified_z_scores = 0.6745 * (clean_series - median) / mad
    return set(clean_series[modified_z_scores.abs() > threshold].index)


def detect_outliers_iqr(series):
    clean_series = series.dropna()

    # IQR is only meaningful for numeric data.
    if not pd.api.types.is_numeric_dtype(clean_series):
        return []

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return set(clean_series[
        (clean_series < lower_bound) | (clean_series > upper_bound)
    ].index.tolist())
