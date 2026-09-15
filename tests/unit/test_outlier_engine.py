import numpy as np
import pandas as pd

from ml_detective.outliers.outlier_engine import (
    analyze_column_outliers,
    analyze_multivariate_outliers,
)


def test_analyze_column_outliers_flags_planted_outlier():
    normal_values = [50 + i for i in range(-10, 10)]
    normal_values.append(1000)
    series = pd.Series(normal_values)

    finding = analyze_column_outliers(series)

    assert finding.passed is False
    assert 20 in finding.evidence["confident_outlier_row_indices"]
    assert finding.evidence["vote_counts"][20] >= 2


def test_analyze_column_outliers_passes_clean_data():
    np.random.seed(42)
    clean_series = pd.Series(np.random.normal(50, 5, size=200))

    finding = analyze_column_outliers(clean_series)

    assert finding.passed is True


def test_analyze_multivariate_outliers_flags_planted_points():
    np.random.seed(42)
    normal_points = pd.DataFrame({
        "feature_a": np.random.normal(50, 5, size=100),
        "feature_b": np.random.normal(100, 10, size=100),
    })
    extreme_points = pd.DataFrame({
        "feature_a": [500, 510, 495],
        "feature_b": [900, 920, 880],
    })
    dataframe = pd.concat([normal_points, extreme_points], ignore_index=True)

    finding = analyze_multivariate_outliers(dataframe, ["feature_a", "feature_b"])

    assert finding.passed is False
    assert {100, 101, 102}.issubset(set(finding.evidence["confident_outlier_row_indices"]))


def test_analyze_multivariate_outliers_requires_at_least_two_columns():
    dataframe = pd.DataFrame({"only_one_column": [1, 2, 3, 4, 5]})

    finding = analyze_multivariate_outliers(dataframe, ["only_one_column"])

    assert finding.passed is True
    assert "at least 2" in finding.message