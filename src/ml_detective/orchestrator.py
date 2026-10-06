"""
Single entry point that runs the full investigation pipeline end-to-end:
validation, outlier detection, leakage/feature-quality checks,
multicollinearity, profiling, scoring, and reasoning. The dashboard
(and any future API) calls ONLY this function -- pipeline logic lives
in one place, not duplicated across UI code.
"""

import pandas as pd

from ml_detective.ingestion.task_detector import TaskType, detect_task_type, guess_target_column
from ml_detective.ingestion.validators import run_all_validations
from ml_detective.leakage.feature_quality import (
    check_feature_redundancy,
    check_high_cardinality_features,
    check_low_variance_features,
)
from ml_detective.leakage.leakage_detector import scan_all_features_for_leakage
from ml_detective.outliers.outlier_engine import analyze_column_outliers, analyze_multivariate_outliers
from ml_detective.profiling.profiler import profile_dataset
from ml_detective.reasoning.reasoning_engine import build_all_detective_findings
from ml_detective.scoring.health_score import compute_health_score
from ml_detective.statistics.multicollinearity import calculate_vif
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def run_full_investigation(dataframe: pd.DataFrame, target_column: str | None = None) -> dict:
    if target_column is None:
        target_column = guess_target_column(dataframe)

    task_type = detect_task_type(dataframe, target_column) if target_column else TaskType.UNKNOWN
    target_is_classification = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    feature_columns = [c for c in dataframe.columns if c != target_column]
    numeric_feature_columns = [
    c for c in feature_columns
    if pd.api.types.is_numeric_dtype(dataframe[c]) and not pd.api.types.is_bool_dtype(dataframe[c])
]

    findings = []
    findings += run_all_validations(dataframe)

    for col in numeric_feature_columns:
        findings.append(analyze_column_outliers(dataframe[col]))
    if len(numeric_feature_columns) >= 2:
        findings.append(analyze_multivariate_outliers(dataframe, numeric_feature_columns))

    if target_column:
        findings += scan_all_features_for_leakage(
            dataframe, feature_columns, target_column, target_is_classification
        )

    findings.append(check_high_cardinality_features(dataframe, feature_columns))
    findings.append(check_low_variance_features(dataframe, feature_columns))
    if len(numeric_feature_columns) >= 2:
        findings.append(check_feature_redundancy(dataframe, numeric_feature_columns))
        findings.append(calculate_vif(dataframe, numeric_feature_columns))

    profile = profile_dataset(dataframe, target_column=target_column)
    health_report = compute_health_score(findings)
    detective_findings = build_all_detective_findings(findings)

    logger.info(f"Full investigation complete: {len(findings)} checks run.")

    return {
        "target_column": target_column,
        "task_type": task_type.value,
        "findings": findings,
        "profile": profile,
        "health_report": health_report,
        "detective_findings": detective_findings,
    }