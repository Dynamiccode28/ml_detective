from dataclasses import dataclass
from typing import Any

from ml_detective.ingestion.findings import Severity


@dataclass
class DetectiveFinding:
    finding: str
    evidence: dict[str, Any]
    reasoning: str
    severity: Severity
    confidence: float  # 0.0-1.0
    business_impact: str
    model_impact: str
    recommended_fix: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding": self.finding,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "business_impact": self.business_impact,
            "model_impact": self.model_impact,
            "recommended_fix": self.recommended_fix,
        }