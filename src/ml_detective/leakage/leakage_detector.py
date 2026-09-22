"""
Combines multiple WEAK heuristics into one leakage suspicion finding
per feature:
    1. Suspiciously high correlation/mutual information with the target
    2. A suspicious column name
"""
import pandas as pd

from ml_detective.config.thresholds import get_thresholds
from ml_detective.ingestion.findings import Severity, ValidationFinding
from ml_detective.leakage.name_heuristics import has_suspicious_name
from ml_detective.statistics.relationship_tests import(calculate_mutual_information,calculate_pearson_correlation)
from ml_detective.utils.logger import get_logger

logger=get_logger(__name__)

def check_feature_for_leakage(
        dataframe:pd.DataFrame,
        feature_column:str,
        target_column:str,
        target_is_classification: bool,
)->ValidationFinding:
    correlation_threshold=get_thresholds()["leakage"]["correlation_suspicion_threshold"]
    suspicious_name_fragments=has_suspicious_name(feature_column)
    feature_series=dataframe[feature_column]
    target_series=dataframe[target_column]
    statistical_suspicion=False
    statistical_evidence={}
    if pd.api.types.is_numeric_dtype(feature_series) and pd.api.types.is_numeric_dtype(target_series):
        correlation_finding=calculate_pearson_correlation(feature_series,target_series)
        correlation_value=abs(correlation_finding.evidence.get("correlation",0))
        if correlation_value>=correlation_threshold:
            statistical_suspicion=True
            statistical_evidence["correlation"]=correlation_value
        else:
            mi_finding=calculate_mutual_information(
                feature_series,target_series, target_is_classification=target_is_classification
            )
        # Mutual information has no fixed "-1 to 1" scale, so we can't
        # reuse the same numeric threshold as correlation. Instead we
        # compare it against the target's own entropy-like scale --
        # simplified here to a high absolute score as a rough proxy.
            mi_score=mi_finding.evidence.get("mutual_information_score",0)
            if mi_score>= 0.7:
                statistical_suspicion=True
                statistical_evidence["mutual_evidence_score"]=mi_score
    is_suspicious = bool(suspicious_name_fragments) or statistical_suspicion

    if not is_suspicious:
        return ValidationFinding(
            check_name="target_leakage",
            passed=True,
            severity=Severity.LOW,
            message=f"'{feature_column}' shows no strong leakage indicators.",
        )

    reasons = []
    if suspicious_name_fragments:
        reasons.append(f"suspicious name pattern(s): {suspicious_name_fragments}")
    if statistical_suspicion:
        reasons.append(f"unusually strong statistical relationship with the target: {statistical_evidence}")

    # Severity reflects HOW MANY independent signals agree -- both name
    # AND statistics pointing the same way is much stronger evidence
    # than either alone.
    severity = Severity.CRITICAL if (suspicious_name_fragments and statistical_suspicion) else Severity.HIGH

    return ValidationFinding(
        check_name="target_leakage",
        passed=False,
        severity=severity,
        message=(
            f"'{feature_column}' shows possible target leakage: {'; '.join(reasons)}. "
            f"This is a SUSPICION based on patterns, not certainty -- please verify "
            f"whether this feature would genuinely be available before the outcome occurs."
        ),
        evidence={
            "suspicious_name_fragments": suspicious_name_fragments,
            "statistical_evidence": statistical_evidence,
        },
    )


def scan_all_features_for_leakage(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target_is_classification: bool,
) -> list[ValidationFinding]:
    """Runs the leakage check across every candidate feature column."""
    logger.info(f"Scanning {len(feature_columns)} feature(s) for target leakage...")

    findings = [
        check_feature_for_leakage(dataframe, feature_column, target_column, target_is_classification)
        for feature_column in feature_columns
    ]

    flagged_count = sum(1 for finding in findings if not finding.passed)
    logger.info(f"Leakage scan complete: {flagged_count} of {len(findings)} feature(s) flagged.")

    return findings
