import numpy as np
import pandas as pd

from ml_detective.statistics.relationship_tests import (
    anova_test,
    calculate_mutual_information,
    calculate_pearson_correlation,
    chi_square_test,
)


def test_pearson_correlation_detects_perfect_linear_relationship():
    x = pd.Series(range(1, 101))
    y = pd.Series([value * 2 + 5 for value in x])

    finding = calculate_pearson_correlation(x, y)

    assert finding.passed is False
    assert finding.evidence["correlation"] > 0.99


def test_pearson_correlation_finds_no_relationship_in_random_data():
    np.random.seed(42)
    x = pd.Series(np.random.normal(size=200))
    y = pd.Series(np.random.normal(size=200))

    finding = calculate_pearson_correlation(x, y)

    assert finding.passed is True
    assert abs(finding.evidence["correlation"]) < 0.3


def test_chi_square_detects_dependent_categorical_columns():
    column_a = pd.Series(["X", "X", "X", "X", "Y", "Y", "Y", "Y"] * 10)
    column_b = pd.Series(["P", "P", "P", "P", "Q", "Q", "Q", "Q"] * 10)

    finding = chi_square_test(column_a, column_b)

    assert finding.passed is False
    assert finding.evidence["p_value"] < 0.05


def test_anova_detects_group_mean_differences():
    categories = pd.Series(["A"] * 50 + ["B"] * 50)
    values = pd.Series([100] * 50 + [10] * 50)

    finding = anova_test(values, categories)

    assert finding.passed is False
    assert finding.evidence["p_value"] < 0.05


def test_mutual_information_is_higher_for_informative_feature():
    np.random.seed(42)
    target = pd.Series(np.random.choice([0, 1], size=300))
    feature_informative = target + np.random.normal(0, 0.1, size=300)
    feature_random = pd.Series(np.random.normal(size=300))

    finding_informative = calculate_mutual_information(
        feature_informative, target, target_is_classification=True
    )
    finding_random = calculate_mutual_information(
        feature_random, target, target_is_classification=True
    )

    assert finding_informative.evidence["mutual_information_score"] > finding_random.evidence["mutual_information_score"]
