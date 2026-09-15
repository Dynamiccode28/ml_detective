"""
clustering_methods.py

DBSCAN as a side-effect outlier detector via its "noise" label.
"""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def detect_outliers_dbscan(
    dataframe: pd.DataFrame, numeric_columns: list[str], eps: float = 0.5, min_samples: int = 5
) -> set[int]:
    usable_data = dataframe[numeric_columns].dropna()

    if len(usable_data) < min_samples:
        return set()

    scaled_data = StandardScaler().fit_transform(usable_data)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = model.fit_predict(scaled_data)

    outlier_positions = usable_data.index[cluster_labels == -1]
    return set(outlier_positions)