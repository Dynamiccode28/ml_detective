"""
distribution_analysis.py

Analyzes the SHAPE of a single numeric column's distribution:
skewness, kurtosis, and formal normality testing.
"""

import numpy as np
import pandas as pd
from scipy import stats

from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

_SHAPIRO_WILK_MAX_SAMPLE_SIZE = 5000
_SKEWNESS_NOTABLE_THRESHOLD = 1.0
_KURTOSIS_NOTABLE_THRESHOLD = 3.0


def calculate_skewness_and_kurtosis(series: pd.Series) -> ValidationFinding:
    clean_series = series.dropna()

    if len(clean_series) < 3:
        return ValidationFinding(
            check_name="skewness_and_kurtosis",
            passed=True,
            severity=Severity.LOW,
            message=f"Column '{series.name}' has too few values to assess distribution shape.",
        )

    skewness = round(float(stats.skew(clean_series)), 4)
    excess_kurtosis = round(float(stats.kurtosis(clean_series)), 4)

    is_notably_skewed = abs(skewness) > _SKEWNESS_NOTABLE_THRESHOLD
    is_notably_heavy_tailed = abs(excess_kurtosis) > _KURTOSIS_NOTABLE_THRESHOLD

    if not is_notably_skewed and not is_notably_heavy_tailed:
        return ValidationFinding(
            check_name="skewness_and_kurtosis",
            passed=True,
            severity=Severity.LOW,
            message=f"Column '{series.name}' has a roughly symmetric, normal-like shape.",
            evidence={"skewness": skewness, "excess_kurtosis": excess_kurtosis},
        )

    direction = "right (long tail of high values)" if skewness > 0 else "left (long tail of low values)"

    return ValidationFinding(
        check_name="skewness_and_kurtosis",
        passed=False,
        severity=Severity.MEDIUM,
        message=(
            f"Column '{series.name}' is skewed {direction} "
            f"(skewness={skewness}, excess kurtosis={excess_kurtosis}). "
            f"Consider a transformation (e.g. log) before modeling."
        ),
        evidence={"skewness": skewness, "excess_kurtosis": excess_kurtosis},
    )


def check_normality(series: pd.Series, alpha: float = 0.05) -> ValidationFinding:
    clean_series = series.dropna()

    if len(clean_series) < 3:
        return ValidationFinding(
            check_name="normality_test",
            passed=True,
            severity=Severity.LOW,
            message=f"Column '{series.name}' has too few values to test for normality.",
        )

    if len(clean_series) <= _SHAPIRO_WILK_MAX_SAMPLE_SIZE:
        test_name = "Shapiro-Wilk"
        statistic, p_value = stats.shapiro(clean_series)
    else:
        test_name = "Kolmogorov-Smirnov"
        standardized = (clean_series - clean_series.mean()) / clean_series.std()
        statistic, p_value = stats.kstest(standardized, "norm")

    statistic = round(float(statistic), 4)
    p_value = round(float(p_value), 6)
    looks_normal = p_value >= alpha

    if looks_normal:
        message = (
            f"Column '{series.name}' shows no strong evidence against normality "
            f"({test_name} test, p={p_value}). This does not PROVE it is normal, "
            f"only that we lack strong evidence otherwise."
        )
        severity = Severity.LOW
    else:
        message = (
            f"Column '{series.name}' significantly deviates from a normal "
            f"distribution ({test_name} test, p={p_value})."
        )
        severity = Severity.LOW

    return ValidationFinding(
        check_name="normality_test",
        passed=looks_normal,
        severity=severity,
        message=message,
        evidence={
            "test_used": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "alpha": alpha,
            "sample_size": len(clean_series),
        },
    )