"""
Combines ValidationFindings from every prior phase into one weighted
0-100 health score, broken down by dimension (completeness, uniqueness,
consistency, validity, accuracy, timeliness).
"""

from ml_detective.config.thresholds import get_thresholds
from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.scoring.models import DimensionScore, HealthScoreReport
from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

_CHECK_NAME_TO_DIMENSION = {
    "missing_values": "completeness",
    "duplicate_rows": "uniqueness",
    "duplicate_columns": "uniqueness",
    "constant_or_near_constant_columns": "consistency",
    "mixed_type_columns": "consistency",
    "rare_categories": "consistency",
    "statistical_outliers": "validity",
    "multivariate_outliers": "validity",
    "target_leakage": "accuracy",
    "high_cardinality_features": "accuracy",
    "low_variance_features": "accuracy",
    "feature_redundancy": "accuracy",
    "multicollinearity_vif": "accuracy",
}

_SEVERITY_PENALTY = {
    Severity.LOW: 5,
    Severity.MEDIUM: 15,
    Severity.HIGH: 30,
    Severity.CRITICAL: 50,
}

_ALL_DIMENSIONS = ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]


def _grade_from_score(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def compute_health_score(findings: list[ValidationFinding]) -> HealthScoreReport:
    weights = get_thresholds()["health_score"]["weights"]

    findings_by_dimension: dict[str, list[ValidationFinding]] = {dim: [] for dim in _ALL_DIMENSIONS}
    for finding in findings:
        dimension = _CHECK_NAME_TO_DIMENSION.get(finding.check_name)
        if dimension:
            findings_by_dimension[dimension].append(finding)

    dimension_scores: dict[str, DimensionScore] = {}
    for dimension in _ALL_DIMENSIONS:
        dimension_findings = findings_by_dimension[dimension]

        if not dimension_findings:
            dimension_scores[dimension] = DimensionScore(
                dimension=dimension, score=100.0, findings_evaluated=0, findings_failed=0, assessed=False
            )
            continue

        penalty_total = sum(
            _SEVERITY_PENALTY[f.severity] for f in dimension_findings if not f.passed
        )
        score = max(0.0, 100.0 - penalty_total)
        failed_count = sum(1 for f in dimension_findings if not f.passed)

        dimension_scores[dimension] = DimensionScore(
            dimension=dimension,
            score=round(score, 2),
            findings_evaluated=len(dimension_findings),
            findings_failed=failed_count,
            assessed=True,
        )

    overall_score = round(
        sum(dimension_scores[dim].score * weights[dim] for dim in _ALL_DIMENSIONS), 2
    )

    logger.info(f"Health score computed: {overall_score}/100 ({_grade_from_score(overall_score)})")

    return HealthScoreReport(
        overall_score=overall_score,
        grade=_grade_from_score(overall_score),
        dimension_scores=dimension_scores,
    )