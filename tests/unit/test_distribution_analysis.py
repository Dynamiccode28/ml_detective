import numpy as np
import pandas as pd

from ml_detective.statistics.distribution_analysis import (
    calculate_skewness_and_kurtosis,
    check_normality,
)


def test_symmetric_normal_data_has_low_skewness():
    np.random.seed(42)
    symmetric_data = pd.Series(np.random.normal(loc=50, scale=10, size=1000))

    finding = calculate_skewness_and_kurtosis(symmetric_data)

    assert finding.passed is True
    assert abs(finding.evidence["skewness"]) < 0.3


def test_heavily_right_skewed_data_is_detected():
    np.random.seed(42)
    right_skewed_data = pd.Series(np.random.exponential(scale=2.0, size=1000))

    finding = calculate_skewness_and_kurtosis(right_skewed_data)

    assert finding.passed is False
    assert finding.evidence["skewness"] > 1.0
    assert "right" in finding.message


def test_normality_check_accepts_normal_data():
    np.random.seed(42)
    normal_data = pd.Series(np.random.normal(loc=0, scale=1, size=500))

    finding = check_normality(normal_data)

    assert finding.passed is True
    assert finding.evidence["test_used"] == "Shapiro-Wilk"


def test_normality_check_rejects_uniform_data():
    np.random.seed(42)
    uniform_data = pd.Series(np.random.uniform(low=0, high=100, size=500))

    finding = check_normality(uniform_data)

    assert finding.passed is False


def test_normality_check_uses_ks_test_above_shapiro_limit():
    np.random.seed(42)
    large_normal_data = pd.Series(np.random.normal(loc=0, scale=1, size=6000))

    finding = check_normality(large_normal_data)

    assert finding.evidence["test_used"] == "Kolmogorov-Smirnov"