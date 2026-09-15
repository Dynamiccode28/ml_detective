"""
findings.py

Defines "ValidationFinding" -- a single, consistent shape for reporting
the result of any single check.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationFinding:
    check_name: str
    passed: bool
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
        }