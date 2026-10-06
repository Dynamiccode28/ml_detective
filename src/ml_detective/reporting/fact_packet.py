"""
fact_packet.py

Builds the ONE plain-dict object the LLM is allowed to see. This is
the boundary that enforces "LLM narrates, never calculates" -- if a
number isn't in this packet, the LLM has no way to know it, so it
can't invent it either.
"""

from typing import Any

from ml_detective.profiling.models import DatasetProfile
from ml_detective.reasoning.models import DetectiveFinding
from ml_detective.scoring.models import HealthScoreReport


def build_fact_packet(
    profile: DatasetProfile,
    health_report: HealthScoreReport,
    detective_findings: list[DetectiveFinding],
) -> dict[str, Any]:
    findings_by_severity: dict[str, list[dict]] = {"critical": [], "high": [], "medium": [], "low": []}
    for finding in detective_findings:
        findings_by_severity[finding.severity.value].append({
            "finding": finding.finding,
            "reasoning": finding.reasoning,
            "confidence": finding.confidence,
            "recommended_fix": finding.recommended_fix,
        })

    return {
        "dataset_shape": {"rows": profile.n_rows, "columns": profile.n_cols},
        "health_score": {
            "overall": health_report.overall_score,
            "grade": health_report.grade,
            "dimensions": {k: v.score for k, v in health_report.dimension_scores.items()},
        },
        "findings_by_severity": findings_by_severity,
        "total_findings": len(detective_findings),
    }
    