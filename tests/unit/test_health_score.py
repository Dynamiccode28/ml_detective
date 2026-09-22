"""
Uses hand-built findings with KNOWN severities so expected scores are
computable by hand, plus real findings from the synthetic dataset.
"""

from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.ingestion.loader import load_csv
from ml_detective.ingestion.validators import run_all_validations
from ml_detective.scoring.health_score import compute_health_score

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_perfect_dataset_scores_100():
    findings = [
        ValidationFinding("missing_values", passed=True, severity=Severity.LOW, message=""),
        ValidationFinding("duplicate_rows", passed=True, severity=Severity.LOW, message=""),
    ]
    report = compute_health_score(findings)
    assert report.overall_score == 100.0
    assert report.grade == "A"


def test_single_critical_failure_reduces_score_by_exactly_penalty():
    findings = [
        ValidationFinding("missing_values", passed=False, severity=Severity.CRITICAL, message=""),
    ]
    report = compute_health_score(findings)
    # completeness: 100 - 50 = 50. Other 5 dimensions unassessed = 100 each.
    assert report.dimension_scores["completeness"].score == 50.0
    weights = {"completeness": 0.25, "consistency": 0.20, "uniqueness": 0.15,
               "validity": 0.20, "accuracy": 0.10, "timeliness": 0.10}
    expected = round(50.0 * 0.25 + 100.0 * (1 - 0.25), 2)
    assert report.overall_score == expected


def test_unmapped_check_name_is_ignored_not_crashed_on():
    findings = [
        ValidationFinding("some_future_unmapped_check", passed=False, severity=Severity.CRITICAL, message=""),
    ]
    report = compute_health_score(findings)
    assert report.overall_score == 100.0  # nothing mapped, nothing penalized


def test_timeliness_is_unassessed_when_no_checks_exist():
    report = compute_health_score([])
    assert report.dimension_scores["timeliness"].assessed is False
    assert report.dimension_scores["timeliness"].score == 100.0


def test_real_synthetic_dataset_produces_a_reduced_score():
    """synthetic data has known planted problems across multiple
    dimensions, so the overall score should be meaningfully below 100."""
    dataframe = load_csv(SAMPLE_CSV_PATH)
    findings = run_all_validations(dataframe)
    report = compute_health_score(findings)

    assert report.overall_score < 100.0
    assert report.dimension_scores["completeness"].findings_failed > 0
    assert report.dimension_scores["uniqueness"].findings_failed > 0