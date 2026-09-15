import math

from ml_detective.ingestion.loader import load_csv
from ml_detective.profiling.profiler import profile_dataset

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_profile_dataset_reports_correct_shape():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    profile = profile_dataset(dataframe)

    assert profile.n_rows == 200
    assert profile.n_cols == len(dataframe.columns)
    assert profile.memory_usage_mb > 0


def test_employee_id_has_near_maximum_cardinality_ratio():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    profile = profile_dataset(dataframe)

    assert profile.column_profiles["employee_id"].cardinality_ratio == round(199 / 200, 4)


def test_country_has_minimum_cardinality_ratio():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    profile = profile_dataset(dataframe)

    assert profile.column_profiles["country"].cardinality_ratio == round(1 / 200, 4)


def test_target_profile_computes_correct_imbalance_ratio_for_churned():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    profile = profile_dataset(dataframe, target_column="churned")

    counts = dataframe["churned"].value_counts()
    expected_ratio = round(counts.max() / counts.min(), 2)

    assert profile.target_profile.imbalance_ratio == expected_ratio
    assert profile.target_profile.task_type == "binary_classification"


def test_target_profile_entropy_is_zero_for_a_perfectly_constant_target():
    import pandas as pd
    from ml_detective.profiling.profiler import _calculate_entropy

    constant_class_counts = pd.Series([200])
    assert _calculate_entropy(constant_class_counts) == 0.0


def test_target_profile_entropy_is_maximum_for_a_perfectly_balanced_binary_target():
    import pandas as pd
    from ml_detective.profiling.profiler import _calculate_entropy

    balanced_class_counts = pd.Series([100, 100])
    entropy = _calculate_entropy(balanced_class_counts)
    assert math.isclose(entropy, 1.0, rel_tol=1e-9)


def test_regression_target_gets_summary_stats_not_class_distribution():
    import pandas as pd

    dataframe = pd.DataFrame({
        "feature": range(10),
        "price": [100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 120.0, 220.0, 310.0, 190.0],
    })
    profile = profile_dataset(dataframe, target_column="price")

    assert profile.target_profile.task_type == "regression"
    assert profile.target_profile.class_distribution is None
    assert profile.target_profile.summary_stats is not None
    assert "mean" in profile.target_profile.summary_stats