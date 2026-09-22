"""
Tests leakage detection using synthetic dataset's DELIBERATELY
planted leaky column (exit_interview_completed), plus hand-built
negative and name-only cases.
"""

import pandas as pd

from ml_detective.ingestion.loader import load_csv
from ml_detective.leakage.leakage_detector import (
    check_feature_for_leakage,
    scan_all_features_for_leakage,
)
from ml_detective.leakage.name_heuristics import has_suspicious_name

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_has_suspicious_name_matches_exit_interview_completed():
    matches = has_suspicious_name("exit_interview_completed")
    assert "completed" in matches
    assert "exit_" in matches


def test_has_suspicious_name_returns_empty_for_innocent_column():
    matches = has_suspicious_name("monthly_salary")
    assert matches == []


def test_check_feature_for_leakage_flags_exit_interview_completed():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    finding = check_feature_for_leakage(
        dataframe,
        feature_column="exit_interview_completed",
        target_column="churned",
        target_is_classification=True,
    )

    assert finding.passed is False
    assert finding.severity.value in ("high", "critical")


def test_check_feature_for_leakage_does_not_flag_innocent_feature():
    dataframe = load_csv(SAMPLE_CSV_PATH)

    finding = check_feature_for_leakage(
        dataframe,
        feature_column="age",
        target_column="churned",
        target_is_classification=True,
    )

    assert finding.passed is True


def test_scan_all_features_for_leakage_returns_one_finding_per_feature():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    feature_columns = ["age", "monthly_salary", "exit_interview_completed"]

    findings = scan_all_features_for_leakage(
        dataframe, feature_columns, target_column="churned", target_is_classification=True
    )

    assert len(findings) == 3
    flagged_names = {f.check_name for f in findings}
    assert flagged_names == {"target_leakage"}