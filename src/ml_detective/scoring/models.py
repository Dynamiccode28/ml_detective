"""Structured shapes for the health score report."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DimensionScore:
    dimension: str
    score: float
    findings_evaluated: int
    findings_failed: int
    assessed: bool  # False if no checks mapped to this dimension

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "findings_evaluated": self.findings_evaluated,
            "findings_failed": self.findings_failed,
            "assessed": self.assessed,
        }


@dataclass
class HealthScoreReport:
    overall_score: float
    grade: str
    dimension_scores: dict[str, DimensionScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "dimension_scores": {k: v.to_dict() for k, v in self.dimension_scores.items()},
        }