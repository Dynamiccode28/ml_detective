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


def detect_outliers_iqr(series: pd.Series) -> set[int]:
    multiplier = get_thresholds()["outliers"]["iqr_multiplier"]
    clean_series = series.dropna()

    if len(clean_series) < 4:
        return set()

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return set()

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
    return set(clean_series[outlier_mask].index)