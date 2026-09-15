import pandas as pd

from ml_detective.outliers.statistical_methods import (
    detect_outliers_iqr,
    detect_outliers_modified_zscore,
    detect_outliers_zscore,
)


def _build_series_with_one_obvious_outlier() -> pd.Series:
    normal_values = [50 + i for i in range(-10, 10)]
    normal_values.append(1000)
    return pd.Series(normal_values)


def test_zscore_detects_the_planted_outlier():
    series = _build_series_with_one_obvious_outlier()
    outlier_indices = detect_outliers_zscore(series)
    assert 20 in outlier_indices


def test_modified_zscore_detects_the_planted_outlier():
    series = _build_series_with_one_obvious_outlier()
    outlier_indices = detect_outliers_modified_zscore(series)
    assert 20 in outlier_indices


def test_iqr_detects_the_planted_outlier():
    series = _build_series_with_one_obvious_outlier()
    outlier_indices = detect_outliers_iqr(series)
    assert 20 in outlier_indices


def test_all_methods_find_nothing_in_perfectly_uniform_data():
    constant_series = pd.Series([50] * 30)

    assert detect_outliers_zscore(constant_series) == set()
    assert detect_outliers_modified_zscore(constant_series) == set()
    assert detect_outliers_iqr(constant_series) == set()