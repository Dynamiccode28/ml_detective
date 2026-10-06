"""
report_builder.py

Assembles the final report by calling the LLM once per section, using
the fact packet as the only source of truth. Falls back to a
template-only report (no LLM) if Ollama is unavailable -- the report
is still fully usable, just less fluently written.
"""

from ml_detective.profiling.models import DatasetProfile
from ml_detective.reasoning.models import DetectiveFinding
from ml_detective.reporting.fact_packet import build_fact_packet
from ml_detective.reporting.llm_client import LLMUnavailableError, generate_text
from ml_detective.reporting.prompts import (
    build_executive_summary_prompt,
    build_next_steps_prompt,
    build_section_narrative_prompt,
)
from ml_detective.scoring.models import HealthScoreReport
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

def _narrate_section(section_title: str, items: list[dict]) -> str:
    if not items:
        return f"No {section_title.lower()} were found."
    return generate_text(build_section_narrative_prompt(section_title, items))

def _template_fallback_summary(fact_packet: dict) -> str:
    score = fact_packet["health_score"]
    return (
        f"This dataset has {fact_packet['dataset_shape']['rows']} rows and "
        f"{fact_packet['dataset_shape']['columns']} columns, with an overall "
        f"health score of {score['overall']}/100 (grade {score['grade']}). "
        f"{fact_packet['total_findings']} issue(s) were identified across all checks."
    )


def generate_report(
    profile: DatasetProfile,
    health_report: HealthScoreReport,
    detective_findings: list[DetectiveFinding],
    use_llm: bool = True,
) -> dict[str, str]:
    fact_packet = build_fact_packet(profile, health_report, detective_findings)

    if not use_llm:
        return _build_template_only_report(fact_packet)

    try:
        executive_summary = generate_text(build_executive_summary_prompt(fact_packet))
        critical_section = _narrate_section("Critical Findings", fact_packet["findings_by_severity"]["critical"])
        medium_section = _narrate_section("Medium Findings", fact_packet["findings_by_severity"]["medium"])
        low_section = _narrate_section("Low Findings", fact_packet["findings_by_severity"]["low"])
        next_steps = generate_text(build_next_steps_prompt(fact_packet))

        return {
            "executive_summary": executive_summary,
            "critical_findings": critical_section,
            "medium_findings": medium_section,
            "low_findings": low_section,
            "next_steps": next_steps,
            "health_score_summary": f"{fact_packet['health_score']['overall']}/100 (Grade {fact_packet['health_score']['grade']})",
            "generated_with_llm": True,
        }
    except LLMUnavailableError as error:
        logger.warning(f"LLM unavailable, falling back to template report: {error}")
        return _build_template_only_report(fact_packet)


def _build_template_only_report(fact_packet: dict) -> dict[str, str]:
    def _list_findings(items: list[dict]) -> str:
        if not items:
            return "None found."
        return "\n".join(f"- {item['finding']} (Fix: {item['recommended_fix']})" for item in items)

    return {
        "executive_summary": _template_fallback_summary(fact_packet),
        "critical_findings": _list_findings(fact_packet["findings_by_severity"]["critical"]),
        "medium_findings": _list_findings(fact_packet["findings_by_severity"]["medium"]),
        "low_findings": _list_findings(fact_packet["findings_by_severity"]["low"]),
        "next_steps": "Review findings above, prioritizing CRITICAL and HIGH severity items first.",

        "health_score_summary": f"{fact_packet['health_score']['overall']}/100 (Grade {fact_packet['health_score']['grade']})",
        "generated_with_llm": False,
    }
