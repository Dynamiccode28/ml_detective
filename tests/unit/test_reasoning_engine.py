from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.ingestion.loader import load_csv
from ml_detective.ingestion.validators import run_all_validations
from ml_detective.reasoning.confidence import estimate_confidence
from ml_detective.reasoning.reasoning_engine import (
    build_all_detective_findings,
    build_detective_finding,
)

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_deterministic_check_has_full_confidence():
    finding = ValidationFinding("duplicate_rows", passed=False, severity=Severity.MEDIUM,
                                  message="", evidence={"duplicate_row_count": 1})
    assert estimate_confidence(finding) == 1.0


def test_leakage_confidence_is_higher_with_two_signals():
    single_signal = ValidationFinding("target_leakage", passed=False, severity=Severity.HIGH,
                                        message="", evidence={"suspicious_name_fragments": ["exit_"], "statistical_evidence": {}})
    both_signals = ValidationFinding("target_leakage", passed=False, severity=Severity.CRITICAL,
                                       message="", evidence={"suspicious_name_fragments": ["exit_"], "statistical_evidence": {"correlation": 0.99}})
    assert estimate_confidence(both_signals) > estimate_confidence(single_signal)


def test_vote_based_confidence_scales_with_agreement():
    two_of_three = ValidationFinding("statistical_outliers", passed=False, severity=Severity.MEDIUM,
                                       message="", evidence={"vote_counts": {5: 2}})
    three_of_three = ValidationFinding("statistical_outliers", passed=False, severity=Severity.MEDIUM,
                                         message="", evidence={"vote_counts": {5: 3}})
    assert estimate_confidence(three_of_three) > estimate_confidence(two_of_three)


def test_build_detective_finding_produces_all_required_fields():
    finding = ValidationFinding("missing_values", passed=False, severity=Severity.HIGH,
                                  message="Age is 10% missing", evidence={"missing_percentage_by_column": {"age": 10.0}})
    detective_finding = build_detective_finding(finding)

    assert detective_finding.finding == finding.message
    assert detective_finding.reasoning
    assert detective_finding.business_impact
    assert detective_finding.model_impact
    assert detective_finding.recommended_fix
    assert 0.0 <= detective_finding.confidence <= 1.0


def test_build_all_detective_findings_skips_passed_checks():
    findings = [
        ValidationFinding("duplicate_rows", passed=True, severity=Severity.LOW, message=""),
        ValidationFinding("missing_values", passed=False, severity=Severity.HIGH, message=""),
    ]
    detective_findings = build_all_detective_findings(findings)
    assert len(detective_findings) == 1


def test_end_to_end_on_real_synthetic_findings():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    findings = run_all_validations(dataframe)
    detective_findings = build_all_detective_findings(findings)

    assert len(detective_findings) > 0
    for df in detective_findings:
        assert df.recommended_fix != ""