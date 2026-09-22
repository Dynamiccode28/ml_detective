"""
Tests high cardinality, low variance, and redundancy checks using our
synthetic dataset and hand-built edge cases.
"""

import pandas as pd

from ml_detective.ingestion.loader import load_csv
from ml_detective.leakage.feature_quality import (
    check_feature_redundancy,
    check_high_cardinality_features,
    check_low_variance_features,
)

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_check_high_cardinality_features_flags_employee_id():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    finding = check_high_cardinality_features(dataframe, feature_columns=["employee_id", "department", "age"])

    assert finding.passed is False
    assert "employee_id" in finding.evidence["high_cardinality_columns"]


def test_check_low_variance_features_flags_planted_constant_column():
    """We build a fresh, obviously-low-variance numeric column to test
    the coefficient-of-variation logic directly and unambiguously."""
    dataframe = pd.DataFrame({
        "nearly_constant": [100.0, 100.01, 99.99, 100.0, 100.02] * 20,
        "normal_variation": [10, 50, 90, 30, 70] * 20,
    })

    finding = check_low_variance_features(dataframe, feature_columns=["nearly_constant", "normal_variation"])

    assert finding.passed is False
    assert "nearly_constant" in finding.evidence["low_variance_columns"]
    assert "normal_variation" not in finding.evidence["low_variance_columns"]


def test_check_feature_redundancy_flags_near_duplicate_pair():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    # monthly_salary and salary_copy are IDENTICAL (planted in Phase 3),
    # so correlation between them should be a perfect 1.0.
    finding = check_feature_redundancy(dataframe, numeric_feature_columns=["monthly_salary", "salary_copy", "age"])

    assert finding.passed is False
    redundant_pairs = finding.evidence["redundant_pairs"]
    assert any(
        {"monthly_salary", "salary_copy"} == {pair["feature_a"], pair["feature_b"]}
        for pair in redundant_pairs
    )