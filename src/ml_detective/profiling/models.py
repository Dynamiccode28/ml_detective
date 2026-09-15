"""
models.py

Defines the structured shapes used to describe a dataset's profile.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ColumnProfile:
    column_name: str
    dtype: str
    unique_count: int
    cardinality_ratio: float
    missing_pct: float
    sparsity_pct: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "cardinality_ratio": self.cardinality_ratio,
            "missing_pct": self.missing_pct,
            "sparsity_pct": self.sparsity_pct,
        }


@dataclass
class TargetProfile:
    task_type: str
    class_distribution: Optional[dict[str, int]]
    imbalance_ratio: Optional[float]
    entropy: Optional[float]
    summary_stats: Optional[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "class_distribution": self.class_distribution,
            "imbalance_ratio": self.imbalance_ratio,
            "entropy": self.entropy,
            "summary_stats": self.summary_stats,
        }


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    memory_usage_mb: float
    column_profiles: dict[str, ColumnProfile]
    target_profile: Optional[TargetProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_usage_mb": self.memory_usage_mb,
            "column_profiles": {
                name: profile.to_dict() for name, profile in self.column_profiles.items()
            },
            "target_profile": self.target_profile.to_dict() if self.target_profile else None,
        }