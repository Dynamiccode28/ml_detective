"""
test_ml_outlier_methods.py
"""

import numpy as np
import pandas as pd

from ml_detective.outliers.clustering_methods import detect_outliers_dbscan
from ml_detective.outliers.ml_methods import (
    detect_outliers_isolation_forest,
    detect_outliers_local_outlier_factor,
)


def _build_dataframe_with_planted_outliers() -> pd.DataFrame:
    np.random.seed(42)
    normal_points = pd.DataFrame({
        "feature_a": np.random.normal(50, 5, size=100),
        "feature_b": np.random.normal(100, 10, size=100),
    })
    extreme_points = pd.DataFrame({
        "feature_a": [500, 510, 495],
        "feature_b": [900, 920, 880],
    })
    combined = pd.concat([normal_points, extreme_points], ignore_index=True)
    return combined


def test_isolation_forest_detects_planted_extreme_points():
    dataframe = _build_dataframe_with_planted_outliers()
    outlier_indices = detect_outliers_isolation_forest(dataframe, ["feature_a", "feature_b"])

    assert {100, 101, 102}.issubset(outlier_indices)


def test_local_outlier_factor_detects_planted_extreme_points():
    dataframe = _build_dataframe_with_planted_outliers()
    outlier_indices = detect_outliers_local_outlier_factor(dataframe, ["feature_a", "feature_b"])

    assert {100, 101, 102}.issubset(outlier_indices)


def test_dbscan_detects_planted_extreme_points_as_noise():
    dataframe = _build_dataframe_with_planted_outliers()
    outlier_indices = detect_outliers_dbscan(dataframe, ["feature_a", "feature_b"])

    assert {100, 101, 102}.issubset(outlier_indices)