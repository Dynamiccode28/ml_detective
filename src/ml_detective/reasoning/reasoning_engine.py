"""reasoning_engine.py — combines a ValidationFinding into a full DetectiveFinding."""

from ml_detective.ingestion.findings import ValidationFinding
from ml_detective.reasoning.confidence import estimate_confidence
from ml_detective.reasoning.explanations import get_explanation
from ml_detective.reasoning.models import DetectiveFinding
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def build_detective_finding(finding: ValidationFinding) -> DetectiveFinding:
    explanation = get_explanation(finding)
    confidence = estimate_confidence(finding)

    return DetectiveFinding(
        finding=finding.message,
        evidence=finding.evidence,
        reasoning=explanation["reasoning"],
        severity=finding.severity,
        confidence=confidence,
        business_impact=explanation["business_impact"],
        model_impact=explanation["model_impact"],
        recommended_fix=explanation["recommended_fix"],
    )


def build_all_detective_findings(findings: list[ValidationFinding]) -> list[DetectiveFinding]:
    logger.info(f"Building detective explanations for {len(findings)} finding(s)...")
    # Only build full explanations for FAILED findings -- passed checks
    # don't need a "recommended fix" or "business impact" narrative.
    return [build_detective_finding(f) for f in findings if not f.passed]