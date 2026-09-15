"""
relationship_tests.py

Measures relationships BETWEEN columns using the right test for each
combination of column types.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

_CORRELATION_STRONG_THRESHOLD = 0.7
_P_VALUE_SIGNIFICANCE_THRESHOLD = 0.05


def calculate_pearson_correlation(series_a: pd.Series, series_b: pd.Series) -> ValidationFinding:
    paired = pd.concat([series_a, series_b], axis=1).dropna()

    if len(paired) < 3:
        return ValidationFinding(
            check_name="pearson_correlation",
            passed=True,
            severity=Severity.LOW,
            message=f"Not enough overlapping data between '{series_a.name}' and '{series_b.name}' to compute correlation.",
        )

    correlation, p_value = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
    correlation = round(float(correlation), 4)
    p_value = round(float(p_value), 6)

    is_strong = abs(correlation) >= _CORRELATION_STRONG_THRESHOLD

    return ValidationFinding(
        check_name="pearson_correlation",
        passed=not is_strong,
        severity=Severity.MEDIUM if is_strong else Severity.LOW,
        message=(
            f"'{series_a.name}' and '{series_b.name}' have a "
            f"{'strong' if is_strong else 'weak-to-moderate'} linear correlation "
            f"(r={correlation}, p={p_value}). Note: Pearson correlation only "
            f"detects straight-line relationships, not curved ones."
        ),
        evidence={"correlation": correlation, "p_value": p_value, "n_pairs": len(paired)},
    )


def chi_square_test(series_a: pd.Series, series_b: pd.Series) -> ValidationFinding:
    paired = pd.concat([series_a, series_b], axis=1).dropna()
    contingency_table = pd.crosstab(paired.iloc[:, 0], paired.iloc[:, 1])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return ValidationFinding(
            check_name="chi_square_test",
            passed=True,
            severity=Severity.LOW,
            message=f"Not enough category variety between '{series_a.name}' and '{series_b.name}' to run a chi-square test.",
        )

    chi2_statistic, p_value, degrees_of_freedom, _ = stats.chi2_contingency(contingency_table)
    p_value = round(float(p_value), 6)
    are_related = p_value < _P_VALUE_SIGNIFICANCE_THRESHOLD

    return ValidationFinding(
        check_name="chi_square_test",
        passed=not are_related,
        severity=Severity.MEDIUM if are_related else Severity.LOW,
        message=(
            f"'{series_a.name}' and '{series_b.name}' "
            f"{'show a statistically significant association' if are_related else 'appear independent'} "
            f"(chi2={round(float(chi2_statistic), 4)}, p={p_value})."
        ),
        evidence={
            "chi2_statistic": round(float(chi2_statistic), 4),
            "p_value": p_value,
            "degrees_of_freedom": int(degrees_of_freedom),
        },
    )


def anova_test(numeric_series: pd.Series, categorical_series: pd.Series) -> ValidationFinding:
    paired = pd.concat([numeric_series, categorical_series], axis=1).dropna()
    paired.columns = ["numeric", "category"]

    groups = [group["numeric"].values for _, group in paired.groupby("category", observed=True)]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return ValidationFinding(
            check_name="anova_test",
            passed=True,
            severity=Severity.LOW,
            message=f"Not enough groups in '{categorical_series.name}' to run ANOVA against '{numeric_series.name}'.",
        )

    f_statistic, p_value = stats.f_oneway(*groups)
    p_value = round(float(p_value), 6)
    group_matters = p_value < _P_VALUE_SIGNIFICANCE_THRESHOLD

    return ValidationFinding(
        check_name="anova_test",
        passed=not group_matters,
        severity=Severity.MEDIUM if group_matters else Severity.LOW,
        message=(
            f"Average '{numeric_series.name}' "
            f"{'differs significantly' if group_matters else 'does not differ significantly'} "
            f"across groups of '{categorical_series.name}' "
            f"(F={round(float(f_statistic), 4)}, p={p_value})."
        ),
        evidence={"f_statistic": round(float(f_statistic), 4), "p_value": p_value, "n_groups": len(groups)},
    )


def calculate_mutual_information(
    feature_series: pd.Series, target_series: pd.Series, target_is_classification: bool
) -> ValidationFinding:
    paired = pd.concat([feature_series, target_series], axis=1).dropna()
    paired.columns = ["feature", "target"]

    if len(paired) < 5:
        return ValidationFinding(
            check_name="mutual_information",
            passed=True,
            severity=Severity.LOW,
            message=f"Not enough data to compute mutual information for '{feature_series.name}'.",
        )

    feature_values = paired["feature"]
    if not pd.api.types.is_numeric_dtype(feature_values):
        feature_values = LabelEncoder().fit_transform(feature_values.astype(str))
    feature_values = np.array(feature_values).reshape(-1, 1)

    target_values = paired["target"]
    if target_is_classification and not pd.api.types.is_numeric_dtype(target_values):
        target_values = LabelEncoder().fit_transform(target_values.astype(str))

    if target_is_classification:
        mi_score = mutual_info_classif(feature_values, target_values, random_state=42)[0]
    else:
        mi_score = mutual_info_regression(feature_values, target_values, random_state=42)[0]

    mi_score = round(float(mi_score), 4)

    return ValidationFinding(
        check_name="mutual_information",
        passed=True,
        severity=Severity.LOW,
        message=f"'{feature_series.name}' has a mutual information score of {mi_score} with the target.",
        evidence={"mutual_information_score": mi_score},
    )