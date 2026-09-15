from ml_detective.ingestion.loader import load_csv
from ml_detective.ingestion.validators import (
    check_constant_and_near_constant_columns,
    check_duplicate_columns,
    check_duplicate_rows,
    check_missing_values,
    check_mixed_type_columns,
    check_rare_categories,
    run_all_validations,
)

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_check_missing_values_detects_planted_missing_values():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_missing_values(dataframe)

    assert finding.passed is False
    assert "age" in finding.evidence["missing_percentage_by_column"]
    assert "monthly_salary" in finding.evidence["missing_percentage_by_column"]


def test_check_missing_values_detects_hidden_missing_values():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_missing_values(dataframe)

    assert finding.passed is False
    hidden_counts = finding.evidence["hidden_missing_value_counts_by_column"]
    assert hidden_counts["years_of_experience"] == 5


def test_check_duplicate_rows_detects_the_one_planted_duplicate():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_duplicate_rows(dataframe)

    assert finding.passed is False
    assert finding.evidence["duplicate_row_count"] == 1
    assert 11 in finding.evidence["duplicate_row_indices"]


def test_check_duplicate_columns_detects_salary_copy():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_duplicate_columns(dataframe)

    assert finding.passed is False
    flattened_groups = [col for group in finding.evidence["duplicate_column_groups"] for col in group]
    assert "monthly_salary" in flattened_groups
    assert "salary_copy" in flattened_groups


def test_check_constant_columns_detects_country():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_constant_and_near_constant_columns(dataframe)

    assert finding.passed is False
    assert "country" in finding.evidence["constant_columns"]
    assert "is_active" in finding.evidence["near_constant_columns_with_dominant_value_pct"]


def test_check_mixed_type_columns_detects_last_bonus_pct():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_mixed_type_columns(dataframe)

    assert finding.passed is False
    assert "last_bonus_pct" in finding.evidence["mixed_type_columns"]
    assert finding.evidence["mixed_type_columns"]["last_bonus_pct"]["numeric_looking_count"] > 0


def test_check_rare_categories_detects_department_typo():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_rare_categories(dataframe)

    assert finding.passed is False
    assert "department" in finding.evidence["rare_category_values_by_column"]
    assert "Enginering" in finding.evidence["rare_category_values_by_column"]["department"]


def test_run_all_validations_returns_one_finding_per_check():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    findings = run_all_validations(dataframe)

    check_names = {finding.check_name for finding in findings}
    assert check_names == {
        "missing_values",
        "duplicate_rows",
        "duplicate_columns",
        "constant_or_near_constant_columns",
        "mixed_type_columns",
        "rare_categories",
    }

    assert all(not finding.passed for finding in findings)