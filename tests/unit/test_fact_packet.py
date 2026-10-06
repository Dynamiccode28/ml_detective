"""test_fact_packet.py — tests the LLM-input boundary, no LLM calls needed."""

from ml_detective.ingestion.findings import Severity
from ml_detective.profiling.models import DatasetProfile
from ml_detective.reasoning.models import DetectiveFinding
from ml_detective.reporting.fact_packet import build_fact_packet
from ml_detective.scoring.models import DimensionScore, HealthScoreReport


def _build_test_profile():
    return DatasetProfile(n_rows=200, n_cols=10, memory_usage_mb=0.5, column_profiles={}, target_profile=None)


def _build_test_health_report():
    dims = {d: DimensionScore(d, 80.0, 1, 1, True) for d in
            ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]}
    return HealthScoreReport(overall_score=82.5, grade="B", dimension_scores=dims)


def test_fact_packet_groups_findings_by_severity():
    findings = [
        DetectiveFinding("f1", {}, "r1", Severity.CRITICAL, 1.0, "b1", "m1", "fix1"),
        DetectiveFinding("f2", {}, "r2", Severity.LOW, 1.0, "b2", "m2", "fix2"),
    ]
    packet = build_fact_packet(_build_test_profile(), _build_test_health_report(), findings)

    assert len(packet["findings_by_severity"]["critical"]) == 1
    assert len(packet["findings_by_severity"]["low"]) == 1
    assert packet["total_findings"] == 2


def test_fact_packet_contains_no_extra_computed_fields_beyond_inputs():
    """Sanity check: the packet's health_score block should match exactly
    what was passed in, not some re-derived number."""
    health_report = _build_test_health_report()
    packet = build_fact_packet(_build_test_profile(), health_report, [])

    assert packet["health_score"]["overall"] == health_report.overall_score
    assert packet["health_score"]["grade"] == health_report.grade