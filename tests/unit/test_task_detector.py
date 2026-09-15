"""
test_task_detector.py
"""

import pandas as pd

from ml_detective.ingestion.loader import load_csv
from ml_detective.ingestion.task_detector import (
    TaskType,
    detect_task_type,
    guess_target_column,
)

SAMPLE_CSV_PATH = "data/sample/employee_churn_synthetic.csv"


def test_guess_target_column_finds_churned_by_common_name():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    guessed_column = guess_target_column(dataframe)
    assert guessed_column == "churned"


def test_guess_target_column_falls_back_to_last_column_when_no_name_matches():
    dataframe = pd.DataFrame({
        "feature_one": [1, 2, 3],
        "feature_two": [4, 5, 6],
        "totally_unrelated_name": [0, 1, 0],
    })
    guessed_column = guess_target_column(dataframe)
    assert guessed_column == "totally_unrelated_name"


def test_detect_task_type_identifies_binary_classification_on_synthetic_data():
    dataframe = load_csv(SAMPLE_CSV_PATH)
    task_type = detect_task_type(dataframe, target_column="churned")
    assert task_type == TaskType.BINARY_CLASSIFICATION


def test_detect_task_type_identifies_regression_for_continuous_numbers():
    dataframe = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5],
        "house_price": [250000.50, 310250.75, 415900.20, 198000.00, 275300.10],
    })
    task_type = detect_task_type(dataframe, target_column="house_price")
    assert task_type == TaskType.REGRESSION


def test_detect_task_type_identifies_multiclass_for_text_categories():
    dataframe = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6],
        "risk_level": ["Low", "Medium", "High", "Low", "Medium", "High"],
    })
    task_type = detect_task_type(dataframe, target_column="risk_level")
    assert task_type == TaskType.MULTICLASS_CLASSIFICATION


def test_detect_task_type_identifies_multiclass_for_disguised_numeric_categories():
    dataframe = pd.DataFrame({
        "feature": range(10),
        "star_rating": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    })
    task_type = detect_task_type(dataframe, target_column="star_rating")
    assert task_type == TaskType.MULTICLASS_CLASSIFICATION