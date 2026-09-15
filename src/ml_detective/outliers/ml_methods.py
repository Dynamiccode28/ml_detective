"""
ml_methods.py

Two machine-learning-based, MULTI-COLUMN outlier detection methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)

_RANDOM_STATE = 42


def detect_outliers_isolation_forest(
    dataframe: pd.DataFrame, numeric_columns: list[str], contamination: float = 0.05
) -> set[int]:
    usable_data = dataframe[numeric_columns].dropna()

    if len(usable_data) < 10:
        return set()

    model = IsolationForest(contamination=contamination, random_state=_RANDOM_STATE)
    predictions = model.fit_predict(usable_data)

    outlier_positions = usable_data.index[predictions == -1]
    return set(outlier_positions)


def detect_outliers_local_outlier_factor(
    dataframe: pd.DataFrame, numeric_columns: list[str], n_neighbors: int = 20
) -> set[int]:
    usable_data = dataframe[numeric_columns].dropna()

    if len(usable_data) <= n_neighbors:
        return set()

    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    predictions = model.fit_predict(usable_data)

    outlier_positions = usable_data.index[predictions == -1]
    return set(outlier_positions)