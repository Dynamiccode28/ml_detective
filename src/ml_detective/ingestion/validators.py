"""
validators.py

Each function checks exactly ONE thing about a dataset, and returns a
ValidationFinding.
"""

import pandas as pd

from ml_detective.config.thresholds import get_thresholds
from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def check_missing_values(dataframe: pd.DataFrame) -> ValidationFinding:
    thresholds = get_thresholds()["data_quality"]
    warning_threshold = thresholds["missing_value_warning_threshold_pct"]
    critical_threshold = thresholds["missing_value_critical_threshold_pct"]
    placeholder_values = set(thresholds["missing_value_placeholders"])

    is_true_missing = dataframe.isna()
    is_hidden_missing = dataframe.astype(str).apply(
        lambda column: column.str.strip().isin(placeholder_values)
    )
    combined_missing_mask = is_true_missing | is_hidden_missing

    missing_pct_per_column = (combined_missing_mask.mean() * 100).round(2)
    columns_with_missing = missing_pct_per_column[missing_pct_per_column > 0].to_dict()

    hidden_missing_counts = is_hidden_missing.sum()
    columns_with_hidden_missing = hidden_missing_counts[hidden_missing_counts > 0].to_dict()

    if not columns_with_missing:
        return ValidationFinding(
            check_name="missing_values",
            passed=True,
            severity=Severity.LOW,
            message="No missing values (true or hidden) found in any column.",
        )

    worst_column, worst_pct = max(columns_with_missing.items(), key=lambda item: item[1])

    if worst_pct >= critical_threshold:
        severity = Severity.CRITICAL
    elif worst_pct >= warning_threshold:
        severity = Severity.HIGH
    else:
        severity = Severity.LOW

    message = (
        f"{len(columns_with_missing)} column(s) have missing values. "
        f"Worst: '{worst_column}' is {worst_pct}% empty."
    )
    if columns_with_hidden_missing:
        message += (
            f" Note: {len(columns_with_hidden_missing)} column(s) have HIDDEN "
            f"missing values disguised as placeholder text (e.g. blank spaces)."
        )

    return ValidationFinding(
        check_name="missing_values",
        passed=False,
        severity=severity,
        message=message,
        evidence={
            "missing_percentage_by_column": columns_with_missing,
            "hidden_missing_value_counts_by_column": columns_with_hidden_missing,
        },
    )


def check_duplicate_rows(dataframe: pd.DataFrame) -> ValidationFinding:
    duplicate_mask = dataframe.duplicated(keep="first")
    number_of_duplicates = int(duplicate_mask.sum())

    if number_of_duplicates == 0:
        return ValidationFinding(
            check_name="duplicate_rows",
            passed=True,
            severity=Severity.LOW,
            message="No duplicate rows found.",
        )

    duplicate_pct = round((number_of_duplicates / len(dataframe)) * 100, 2)
    severity = Severity.HIGH if duplicate_pct > 5 else Severity.MEDIUM

    return ValidationFinding(
        check_name="duplicate_rows",
        passed=False,
        severity=severity,
        message=f"Found {number_of_duplicates} duplicate row(s) ({duplicate_pct}% of the dataset).",
        evidence={
            "duplicate_row_count": number_of_duplicates,
            "duplicate_row_indices": dataframe[duplicate_mask].index.tolist(),
        },
    )


def check_duplicate_columns(dataframe: pd.DataFrame) -> ValidationFinding:
    duplicate_column_groups: list[list[str]] = []
    checked_columns: set[str] = set()

    columns = list(dataframe.columns)
    for i, column_a in enumerate(columns):
        if column_a in checked_columns:
            continue
        matches = [column_a]
        for column_b in columns[i + 1:]:
            if column_b in checked_columns:
                continue
            if dataframe[column_a].equals(dataframe[column_b]):
                matches.append(column_b)
                checked_columns.add(column_b)
        if len(matches) > 1:
            duplicate_column_groups.append(matches)
            checked_columns.add(column_a)

    if not duplicate_column_groups:
        return ValidationFinding(
            check_name="duplicate_columns",
            passed=True,
            severity=Severity.LOW,
            message="No duplicate columns found.",
        )

    return ValidationFinding(
        check_name="duplicate_columns",
        passed=False,
        severity=Severity.MEDIUM,
        message=f"Found {len(duplicate_column_groups)} group(s) of identical columns.",
        evidence={"duplicate_column_groups": duplicate_column_groups},
    )


def check_constant_and_near_constant_columns(dataframe: pd.DataFrame) -> ValidationFinding:
    constant_columns: list[str] = []
    near_constant_columns: dict[str, float] = {}

    for column_name in dataframe.columns:
        value_counts = dataframe[column_name].value_counts(normalize=True, dropna=True)
        if value_counts.empty:
            continue

        top_value_share = value_counts.iloc[0]

        if top_value_share == 1.0:
            constant_columns.append(column_name)
        elif top_value_share >= 0.95:
            near_constant_columns[column_name] = round(top_value_share * 100, 2)

    if not constant_columns and not near_constant_columns:
        return ValidationFinding(
            check_name="constant_or_near_constant_columns",
            passed=True,
            severity=Severity.LOW,
            message="No constant or near-constant columns found.",
        )

    severity = Severity.HIGH if constant_columns else Severity.MEDIUM

    return ValidationFinding(
        check_name="constant_or_near_constant_columns",
        passed=False,
        severity=severity,
        message=(
            f"Found {len(constant_columns)} constant column(s) and "
            f"{len(near_constant_columns)} near-constant column(s)."
        ),
        evidence={
            "constant_columns": constant_columns,
            "near_constant_columns_with_dominant_value_pct": near_constant_columns,
        },
    )


def check_mixed_type_columns(dataframe: pd.DataFrame) -> ValidationFinding:
    mixed_type_columns: dict[str, dict[str, list]] = {}

    for column_name in dataframe.columns:
        series = dataframe[column_name].dropna()
        if series.empty or not pd.api.types.is_string_dtype(series):
            continue

        numeric_attempt = pd.to_numeric(series, errors="coerce")
        successfully_numeric_mask = numeric_attempt.notna()

        number_of_numeric_values = int(successfully_numeric_mask.sum())
        number_of_non_numeric_values = int((~successfully_numeric_mask).sum())

        if number_of_numeric_values > 0 and number_of_non_numeric_values > 0:
            non_numeric_examples = series[~successfully_numeric_mask].unique().tolist()[:5]
            mixed_type_columns[column_name] = {
                "numeric_looking_count": number_of_numeric_values,
                "non_numeric_example_values": non_numeric_examples,
            }

    if not mixed_type_columns:
        return ValidationFinding(
            check_name="mixed_type_columns",
            passed=True,
            severity=Severity.LOW,
            message="No columns with mixed data types found.",
        )

    return ValidationFinding(
        check_name="mixed_type_columns",
        passed=False,
        severity=Severity.HIGH,
        message=f"Found {len(mixed_type_columns)} column(s) with inconsistent value types.",
        evidence={"mixed_type_columns": mixed_type_columns},
    )


def check_rare_categories(dataframe: pd.DataFrame, rare_threshold_pct: float = 2.0) -> ValidationFinding:
    suspicious_columns: dict[str, dict[str, float]] = {}

    for column_name in dataframe.columns:
        series = dataframe[column_name].dropna()
        if series.empty or not pd.api.types.is_string_dtype(series):
            continue

        if series.nunique() > 30:
            continue

        value_percentages = (series.value_counts(normalize=True) * 100).round(2)
        rare_values = value_percentages[value_percentages < rare_threshold_pct]

        if not rare_values.empty:
            suspicious_columns[column_name] = rare_values.to_dict()

    if not suspicious_columns:
        return ValidationFinding(
            check_name="rare_categories",
            passed=True,
            severity=Severity.LOW,
            message="No suspiciously rare category values found.",
        )

    return ValidationFinding(
        check_name="rare_categories",
        passed=False,
        severity=Severity.MEDIUM,
        message=(
            f"Found {len(suspicious_columns)} column(s) with rare category values "
            f"(possible typos or genuine rare cases -- worth a manual look)."
        ),
        evidence={"rare_category_values_by_column": suspicious_columns},
    )


def run_all_validations(dataframe: pd.DataFrame) -> list[ValidationFinding]:
    logger.info("Running all validation checks...")

    findings = [
        check_missing_values(dataframe),
        check_duplicate_rows(dataframe),
        check_duplicate_columns(dataframe),
        check_constant_and_near_constant_columns(dataframe),
        check_mixed_type_columns(dataframe),
        check_rare_categories(dataframe),
    ]

    failed_count = sum(1 for finding in findings if not finding.passed)
    logger.info(f"Validation complete: {failed_count} of {len(findings)} checks flagged an issue.")

    return findings