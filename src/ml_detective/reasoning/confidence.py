"""
Estimates confidence per finding, based on the NATURE of how that
finding was produced -- never a flat/invented number. Rule:
  - Deterministic counts/measurements (missing %, VIF, duplicates) -> 1.0
  - Vote-based multi-method agreement (outliers) -> fraction of methods agreeing
  - p-value-based statistical tests -> scaled from p-value
  - Heuristic suspicions (leakage) -> capped moderate, higher if multiple signals agree
"""

from ml_detective.ingestion.findings import ValidationFinding

_DETERMINISTIC_CHECKS = {
    "missing_values", "duplicate_rows", "duplicate_columns",
    "constant_or_near_constant_columns", "mixed_type_columns",
    "rare_categories", "multicollinearity_vif",
    "high_cardinality_features", "low_variance_features", "feature_redundancy",
}

_PVALUE_CHECKS = {"normality_test", "pearson_correlation", "chi_square_test", "anova_test"}

_VOTE_BASED_CHECKS = {"statistical_outliers", "multivariate_outliers"}


def estimate_confidence(finding: ValidationFinding) -> float:
    if finding.passed:
        return 1.0 if finding.check_name in _DETERMINISTIC_CHECKS else 0.9

    if finding.check_name in _DETERMINISTIC_CHECKS:
        return 1.0

    if finding.check_name in _VOTE_BASED_CHECKS:
        vote_counts = finding.evidence.get("vote_counts", {})
        if not vote_counts:
            return 0.5
        max_possible_votes = 3  # 3 methods per engine (statistical or ML)
        avg_votes = sum(vote_counts.values()) / len(vote_counts)
        return round(min(avg_votes / max_possible_votes, 1.0), 2)

    if finding.check_name in _PVALUE_CHECKS:
        p_value = finding.evidence.get("p_value", 0.05)
        # Smaller p-value = stronger evidence against null = higher confidence
        # in the finding as stated. Scaled so p=0.05 -> 0.5, p->0 -> ~1.0
        confidence = 1.0 - min(p_value / 0.05, 1.0) * 0.5
        return round(confidence, 2)

    if finding.check_name == "target_leakage":
        has_name_match = bool(finding.evidence.get("suspicious_name_fragments"))
        has_stat_match = bool(finding.evidence.get("statistical_evidence"))
        if has_name_match and has_stat_match:
            return 0.85
        return 0.6  # single weak signal -- explicitly a suspicion, not certainty

    return 0.7  # fallback for any future unmapped check