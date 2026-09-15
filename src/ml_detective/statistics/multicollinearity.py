"""
multicollinearity.py

Detects features highly correlated WITH EACH OTHER using VIF.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ml_detective.config.thresholds import get_thresholds
from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_vif(dataframe: pd.DataFrame, numeric_columns: list[str]) -> ValidationFinding:
    thresholds = get_thresholds()["multicollinearity"]
    warning_threshold = thresholds["vif_warning_threshold"]
    critical_threshold = thresholds["vif_critical_threshold"]

    usable_data = dataframe[numeric_columns].dropna()

    if len(numeric_columns) < 2 or len(usable_data) < len(numeric_columns) + 1:
        return ValidationFinding(
            check_name="multicollinearity_vif",
            passed=True,
            severity=Severity.LOW,
            message="Not enough numeric columns or rows to compute VIF meaningfully.",
        )

    vif_scores = {}
    for i, column_name in enumerate(numeric_columns):
        try:
            vif_value = variance_inflation_factor(usable_data.values, i)
            vif_scores[column_name] = round(float(vif_value), 2)
        except Exception as error:
            logger.warning(f"Could not compute VIF for '{column_name}': {error}")

    problematic_columns = {
        name: score for name, score in vif_scores.items() if score >= warning_threshold
    }

    if not problematic_columns:
        return ValidationFinding(
            check_name="multicollinearity_vif",
            passed=True,
            severity=Severity.LOW,
            message="No significant multicollinearity detected among numeric features.",
            evidence={"vif_scores": vif_scores},
        )

    worst_column = max(problematic_columns, key=problematic_columns.get)
    worst_score = problematic_columns[worst_column]
    severity = Severity.CRITICAL if worst_score >= critical_threshold else Severity.HIGH

    return ValidationFinding(
        check_name="multicollinearity_vif",
        passed=False,
        severity=severity,
        message=(
            f"{len(problematic_columns)} feature(s) show multicollinearity. "
            f"Worst: '{worst_column}' has VIF={worst_score}."
        ),
        evidence={"vif_scores": vif_scores, "problematic_columns": problematic_columns},
    )