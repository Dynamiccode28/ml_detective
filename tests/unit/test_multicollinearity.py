"""
test_multicollinearity.py
"""

import numpy as np
import pandas as pd

from ml_detective.statistics.multicollinearity import calculate_vif


def test_vif_flags_near_duplicate_columns():
    np.random.seed(42)
    base_feature = np.random.normal(size=200)

    dataframe = pd.DataFrame({
        "feature_a": base_feature,
        "feature_b": base_feature + np.random.normal(0, 0.01, size=200),
        "feature_c": np.random.normal(size=200),
    })

    finding = calculate_vif(dataframe, numeric_columns=["feature_a", "feature_b", "feature_c"])

    assert finding.passed is False
    assert finding.evidence["vif_scores"]["feature_a"] > 10
    assert finding.evidence["vif_scores"]["feature_b"] > 10


def test_vif_passes_for_independent_columns():
    np.random.seed(42)
    dataframe = pd.DataFrame({
        "feature_a": np.random.normal(size=200),
        "feature_b": np.random.normal(size=200),
        "feature_c": np.random.normal(size=200),
    })

    finding = calculate_vif(dataframe, numeric_columns=["feature_a", "feature_b", "feature_c"])

    assert finding.passed is True