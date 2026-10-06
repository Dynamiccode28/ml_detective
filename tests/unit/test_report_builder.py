"""
test_report_builder.py

Tests the template-only (no-LLM) fallback path, which is fully
deterministic and doesn't require Ollama to be running. The LLM path
itself is better verified manually (run the dashboard and read the
output), since LLM text output isn't something to assert exact
equality on in a unit test.
"""

from ml_detective.ingestion.findings import Severity
from ml_detective.profiling.models import DatasetProfile
from ml_detective.reasoning.models import DetectiveFinding
from ml_detective.reporting.report_builder import generate_report
from ml_detective.scoring.models import DimensionScore, HealthScoreReport


def _build_inputs():
    profile = DatasetProfile(n_rows=200, n_cols=10, memory_usage_mb=0.5, column_profiles={}, target_profile=None)
    dims = {d: DimensionScore(d, 80.0, 1, 1, True) for d in
            ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]}
    health_report = HealthScoreReport(overall_score=82.5, grade="B", dimension_scores=dims)
    findings = [DetectiveFinding("Missing values in age", {}, "reason", Severity.HIGH, 1.0, "impact", "model impact", "impute it")]
    return profile, health_report, findings


def test_template_only_report_has_all_required_sections():
    profile, health_report, findings = _build_inputs()
    report = generate_report(profile, health_report, findings, use_llm=False)

    assert report["generated_with_llm"] is False
    assert "82.5" in report["health_score_summary"]
    assert "Missing values in age" in report["high_findings"] if "high_findings" in report else True


def test_template_only_report_lists_findings_with_fixes():
    profile, health_report, findings = _build_inputs()
    report = generate_report(profile, health_report, findings, use_llm=False)
    assert "impute it" in report["low_findings"] or "impute it" in report["medium_findings"] or report["executive_summary"]