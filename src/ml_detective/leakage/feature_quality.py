"""
Investigates feature USABILITY problems that aren't about leakage:
high cardinality, low variance, and feature redundancy. Reuses logic
already built in earlier phases, reframed with feature-usability
reasoning rather than general data-quality reasoning.
"""

import pandas as pd

from ml_detective.config.thresholds import get_thresholds
from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.statistics.relationship_tests import calculate_pearson_correlation
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def check_high_cardinality_features(dataframe: pd.DataFrame, feature_columns: list[str]) -> ValidationFinding:
    """
    Flags features where almost every value is unique (like an ID
    column) -- these can't generalize to new, unseen data, since a
    model can't learn a pattern from values it will never see again.
    """
    threshold = get_thresholds()["data_quality"]["high_cardinality_ratio_threshold"]

    high_cardinality_columns = {}
    for column_name in feature_columns:
        series = dataframe[column_name].dropna()
        if len(series) == 0:
            continue
        cardinality_ratio = round(series.nunique() / len(series), 4)
        if cardinality_ratio >= threshold:
            high_cardinality_columns[column_name] = cardinality_ratio

    if not high_cardinality_columns:
        return ValidationFinding(
            check_name="high_cardinality_features",
            passed=True,
            severity=Severity.LOW,
            message="No features with problematically high cardinality found.",
        )

    return ValidationFinding(
        check_name="high_cardinality_features",
        passed=False,
        severity=Severity.MEDIUM,
        message=(
            f"{len(high_cardinality_columns)} feature(s) have very high cardinality "
            f"(likely identifier-like columns unsuitable as model features)."
        ),
        evidence={"high_cardinality_columns": high_cardinality_columns},
    )


def check_low_variance_features(dataframe: pd.DataFrame, feature_columns: list[str]) -> ValidationFinding:
    """
    Flags numeric features with near-zero variance relative to their
    mean (coefficient of variation) -- these barely change across rows,
    so they carry little information a model could use.
    """
    low_variance_columns = {}

    for column_name in feature_columns:
        series = dataframe[column_name].dropna()
        if not pd.api.types.is_numeric_dtype(series) or len(series) < 2:
            continue

        mean_value = series.mean()
        std_value = series.std()

        if mean_value == 0:
            continue  # coefficient of variation is undefined when mean is 0

        coefficient_of_variation = abs(std_value / mean_value)
        if coefficient_of_variation < 0.01:  # less than 1% relative variation
            low_variance_columns[column_name] = round(coefficient_of_variation, 6)

    if not low_variance_columns:
        return ValidationFinding(
            check_name="low_variance_features",
            passed=True,
            severity=Severity.LOW,
            message="No numeric features with problematically low variance found.",
        )

    return ValidationFinding(
        check_name="low_variance_features",
        passed=False,
        severity=Severity.MEDIUM,
        message=f"{len(low_variance_columns)} numeric feature(s) show very low variance.",
        evidence={"low_variance_columns": low_variance_columns},
    )


def check_feature_redundancy(dataframe: pd.DataFrame, numeric_feature_columns: list[str]) -> ValidationFinding:
    """
    Flags PAIRS of numeric features that are very highly correlated
    with EACH OTHER (not the target) -- a sign that one of the pair is
    likely redundant and could be dropped.
    """
    redundant_pairs = []

    for i, column_a in enumerate(numeric_feature_columns):
        for column_b in numeric_feature_columns[i + 1:]:
            finding = calculate_pearson_correlation(dataframe[column_a], dataframe[column_b])
            correlation = abs(finding.evidence.get("correlation", 0))
            if correlation >= 0.9:
                redundant_pairs.append({"feature_a": column_a, "feature_b": column_b, "correlation": correlation})

    if not redundant_pairs:
        return ValidationFinding(
            check_name="feature_redundancy",
            passed=True,
            severity=Severity.LOW,
            message="No highly redundant feature pairs found.",
        )

    return ValidationFinding(
        check_name="feature_redundancy",
        passed=False,
        severity=Severity.MEDIUM,
        message=f"Found {len(redundant_pairs)} highly redundant feature pair(s) (correlation >= 0.9).",
        evidence={"redundant_pairs": redundant_pairs},
    )