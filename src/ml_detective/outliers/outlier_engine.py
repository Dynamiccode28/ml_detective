"""
outlier_engine.py

Combines ALL outlier detection methods using a VOTING approach.
"""

import pandas as pd

from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.outliers.clustering_methods import detect_outliers_dbscan
from ml_detective.outliers.ml_methods import (
    detect_outliers_isolation_forest,
    detect_outliers_local_outlier_factor,
)
from ml_detective.outliers.statistical_methods import (
    detect_outliers_iqr,
    detect_outliers_modified_zscore,
    detect_outliers_zscore,
)
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_column_outliers(series: pd.Series) -> ValidationFinding:
    results_by_method = {
        "z_score": detect_outliers_zscore(series),
        "modified_z_score": detect_outliers_modified_zscore(series),
        "iqr": detect_outliers_iqr(series),
    }

    vote_counts: dict[int, int] = {}
    for outlier_indices in results_by_method.values():
        for row_index in outlier_indices:
            vote_counts[row_index] = vote_counts.get(row_index, 0) + 1

    confident_outliers = {index: votes for index, votes in vote_counts.items() if votes >= 2}

    if not confident_outliers:
        return ValidationFinding(
            check_name="statistical_outliers",
            passed=True,
            severity=Severity.LOW,
            message=f"No confident outliers found in '{series.name}' (using Z-score, Modified Z-score, and IQR).",
            evidence={"method_results": {k: len(v) for k, v in results_by_method.items()}},
        )

    outlier_pct = round(len(confident_outliers) / len(series.dropna()) * 100, 2)
    severity = Severity.HIGH if outlier_pct > 5 else Severity.MEDIUM

    return ValidationFinding(
        check_name="statistical_outliers",
        passed=False,
        severity=severity,
        message=(
            f"Found {len(confident_outliers)} confident outlier(s) in '{series.name}' "
            f"({outlier_pct}% of values), agreed on by 2+ of 3 detection methods."
        ),
        evidence={
            "method_results": {k: len(v) for k, v in results_by_method.items()},
            "confident_outlier_row_indices": sorted(confident_outliers.keys()),
            "vote_counts": confident_outliers,
        },
    )


def analyze_multivariate_outliers(dataframe: pd.DataFrame, numeric_columns: list[str]) -> ValidationFinding:
    if len(numeric_columns) < 2:
        return ValidationFinding(
            check_name="multivariate_outliers",
            passed=True,
            severity=Severity.LOW,
            message="Need at least 2 numeric columns to detect multivariate outliers.",
        )

    results_by_method = {
        "isolation_forest": detect_outliers_isolation_forest(dataframe, numeric_columns),
        "local_outlier_factor": detect_outliers_local_outlier_factor(dataframe, numeric_columns),
        "dbscan": detect_outliers_dbscan(dataframe, numeric_columns),
    }

    vote_counts: dict[int, int] = {}
    for outlier_indices in results_by_method.values():
        for row_index in outlier_indices:
            vote_counts[row_index] = vote_counts.get(row_index, 0) + 1

    confident_outliers = {index: votes for index, votes in vote_counts.items() if votes >= 2}

    if not confident_outliers:
        return ValidationFinding(
            check_name="multivariate_outliers",
            passed=True,
            severity=Severity.LOW,
            message="No confident multivariate outliers found (using Isolation Forest, LOF, and DBSCAN).",
            evidence={"method_results": {k: len(v) for k, v in results_by_method.items()}},
        )

    outlier_pct = round(len(confident_outliers) / len(dataframe) * 100, 2)
    severity = Severity.HIGH if outlier_pct > 5 else Severity.MEDIUM

    return ValidationFinding(
        check_name="multivariate_outliers",
        passed=False,
        severity=severity,
        message=(
            f"Found {len(confident_outliers)} confident multivariate outlier(s) "
            f"({outlier_pct}% of rows), agreed on by 2+ of 3 methods."
        ),
        evidence={
            "method_results": {k: len(v) for k, v in results_by_method.items()},
            "confident_outlier_row_indices": sorted(confident_outliers.keys()),
            "vote_counts": confident_outliers,
        },
    )