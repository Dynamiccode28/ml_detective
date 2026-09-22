"""
Text templates for reasoning/business_impact/model_impact/recommended_fix
per check type. Kept as simple string templates now -- Phase 10 will let
an LLM rewrite these more fluently, but templates guarantee a sensible
baseline explanation exists even without any LLM involved.
"""

from ml_detective.ingestion.findings import ValidationFinding

_TEMPLATES = {
    "missing_values": {
        "reasoning": "Missing values reduce the amount of usable information per row and can bias statistics if missingness isn't random.",
        "business_impact": "Incomplete records may lead to unreliable downstream decisions or reports.",
        "model_impact": "Most ML algorithms cannot handle missing values directly and will error or silently drop rows.",
        "recommended_fix": "Impute (mean/median/mode) or drop rows/columns depending on missingness severity; investigate WHY values are missing first.",
    },
    "duplicate_rows": {
        "reasoning": "Identical rows can inflate perceived data volume and let a model 'memorize' repeated examples rather than generalize.",
        "business_impact": "May overstate the true number of unique events/customers/records.",
        "model_impact": "Risk of data leakage between train/test splits if duplicates land on both sides.",
        "recommended_fix": "Remove duplicate rows unless duplication is a genuine, expected pattern in the domain.",
    },
    "duplicate_columns": {
        "reasoning": "Two identical columns carry zero additional information beyond one of them.",
        "business_impact": "Wastes storage and confuses anyone reviewing the schema.",
        "model_impact": "Can distort feature importance and correlation-based analysis.",
        "recommended_fix": "Drop one column from each duplicate group.",
    },
    "constant_or_near_constant_columns": {
        "reasoning": "A column that barely varies carries little to no information for distinguishing rows.",
        "business_impact": "May indicate a broken data pipeline (e.g. a field that should vary but doesn't).",
        "model_impact": "Provides negligible predictive value; wastes model capacity.",
        "recommended_fix": "Drop the column, or investigate why it isn't varying as expected.",
    },
    "mixed_type_columns": {
        "reasoning": "A column meant to be numeric contains some non-numeric text, blocking correct type inference.",
        "business_impact": "Downstream calculations on this field may silently produce wrong results.",
        "model_impact": "The column may load as text, making it unusable as a numeric feature without cleaning.",
        "recommended_fix": "Clean non-numeric entries (e.g. strip '%' signs) and cast the column to numeric.",
    },
    "rare_categories": {
        "reasoning": "A category value appearing very rarely often signals a typo or data entry error.",
        "business_impact": "May fragment reporting/analysis into spurious extra categories.",
        "model_impact": "Rare categories are hard for models to learn from and may need grouping into 'Other'.",
        "recommended_fix": "Manually review rare values for typos; consider grouping genuinely rare-but-valid categories.",
    },
    "statistical_outliers": {
        "reasoning": "Values far outside the typical range were flagged by multiple independent detection methods.",
        "business_impact": "May represent genuine rare events, or data entry errors that need investigation.",
        "model_impact": "Outliers can disproportionately influence distance-based and linear models.",
        "recommended_fix": "Investigate flagged rows individually; cap, transform, or remove only if confirmed erroneous.",
    },
    "multivariate_outliers": {
        "reasoning": "These rows are unusual in COMBINATION across multiple features, even if individually normal.",
        "business_impact": "Same as single-column outliers, but harder to spot manually.",
        "model_impact": "Can distort models that assume typical multivariate relationships.",
        "recommended_fix": "Review flagged rows for plausibility; consider robust modeling techniques if they're genuine.",
    },
    "target_leakage": {
        "reasoning": "This feature shows patterns consistent with information that wouldn't be available before the outcome occurs.",
        "business_impact": "A model trained with this feature may look excellent in testing but fail in production.",
        "model_impact": "Inflates apparent model performance in a way that doesn't generalize to real predictions.",
        "recommended_fix": "Verify manually whether this feature is genuinely available at prediction time; remove if not.",
    },
    "high_cardinality_features": {
        "reasoning": "Nearly every value in this column is unique, behaving like an identifier rather than a category.",
        "business_impact": "Not directly harmful to the business, but signals the column isn't a real predictive feature.",
        "model_impact": "Cannot generalize -- a model can't learn from a value it will never see again.",
        "recommended_fix": "Exclude from modeling, or engineer a lower-cardinality derived feature from it.",
    },
    "low_variance_features": {
        "reasoning": "This numeric feature barely changes across rows relative to its scale.",
        "business_impact": "Minimal, but indicates the field may not be tracked/varying as expected.",
        "model_impact": "Provides little signal for a model to use.",
        "recommended_fix": "Consider dropping unless domain knowledge says small variation is still meaningful.",
    },
    "feature_redundancy": {
        "reasoning": "Two features are so highly correlated they likely carry the same information.",
        "business_impact": "Minimal directly, but adds unnecessary complexity to data collection/maintenance.",
        "model_impact": "Can destabilize coefficient estimates in linear models (multicollinearity).",
        "recommended_fix": "Keep one of the redundant pair and drop the other.",
    },
    "multicollinearity_vif": {
        "reasoning": "This feature is highly predictable from other features, indicating redundant information in the feature set.",
        "business_impact": "Minimal directly.",
        "model_impact": "Destabilizes coefficient estimates in linear/logistic regression; less of an issue for tree-based models.",
        "recommended_fix": "Remove or combine redundant features, or use regularization (Ridge/Lasso).",
    },
    "normality_test": {
        "reasoning": "This column's distribution significantly deviates from a normal (bell-curve) shape.",
        "business_impact": "Minimal directly.",
        "model_impact": "Models/tests assuming normality (e.g. linear regression's error assumptions) may be less reliable.",
        "recommended_fix": "Consider a transformation (log, Box-Cox) if using a normality-assuming method.",
    },
    "pearson_correlation": {
        "reasoning": "Two numeric columns move together in a strong straight-line relationship.",
        "business_impact": "May indicate redundant data collection.",
        "model_impact": "Contributes to multicollinearity if both are used as features.",
        "recommended_fix": "Consider dropping one, or explicitly model the relationship if both are needed.",
    },
    "chi_square_test": {
        "reasoning": "Two categorical columns show a statistically significant association.",
        "business_impact": "May reveal a genuine business relationship worth understanding.",
        "model_impact": "Related categorical features may carry overlapping information.",
        "recommended_fix": "No fix required necessarily -- informational; consider feature engineering if useful.",
    },
    "anova_test": {
        "reasoning": "A numeric column's average differs significantly across groups of a categorical column.",
        "business_impact": "The categorical grouping is meaningfully associated with the numeric outcome.",
        "model_impact": "This categorical feature likely carries real predictive signal for the numeric column.",
        "recommended_fix": "No fix required -- this is a positive signal the feature may be useful.",
    },
}

_DEFAULT_TEMPLATE = {
    "reasoning": "This check flagged a pattern worth reviewing.",
    "business_impact": "Impact not yet characterized for this check type.",
    "model_impact": "Impact not yet characterized for this check type.",
    "recommended_fix": "Manual review recommended.",
}


def get_explanation(finding: ValidationFinding) -> dict[str, str]:
    return _TEMPLATES.get(finding.check_name, _DEFAULT_TEMPLATE)