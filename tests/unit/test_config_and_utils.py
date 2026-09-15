"""
test_config_and_utils.py

Tests config loading, thresholds, exceptions, logging, timing, and
file helpers.
"""

import pytest

from ml_detective.config.settings import AppEnvironment, settings
from ml_detective.config.thresholds import get_thresholds
from ml_detective.utils.exceptions import (
    EmptyDatasetError,
    IngestionError,
    MLDetectiveError,
)
from ml_detective.utils.file_helpers import get_file_size_mb
from ml_detective.utils.logger import get_logger
from ml_detective.utils.timing import timeit


def test_settings_load_with_expected_defaults():
    assert settings.app_env in list(AppEnvironment)
    assert settings.database_url.startswith("sqlite") or settings.database_url.startswith("mysql")


def test_thresholds_yaml_loads_and_has_expected_keys():
    thresholds = get_thresholds()
    assert "outliers" in thresholds
    assert "leakage" in thresholds
    assert thresholds["outliers"]["z_score_threshold"] == 3.0


def test_health_score_weights_add_up_to_one():
    thresholds = get_thresholds()
    weights = thresholds["health_score"]["weights"]
    assert round(sum(weights.values()), 5) == 1.0


def test_custom_exception_hierarchy():
    with pytest.raises(IngestionError):
        raise EmptyDatasetError("The uploaded file has no rows.")

    with pytest.raises(MLDetectiveError):
        raise EmptyDatasetError("Still catchable via the base error class.")


def test_logger_returns_usable_logger_object():
    logger = get_logger(__name__)
    logger.info("This is a test log message from test_logger_returns_usable_logger_object.")
    assert logger.name == __name__


def test_timeit_decorator_does_not_change_function_behavior():
    @timeit
    def add_numbers(a: int, b: int) -> int:
        return a + b

    assert add_numbers(2, 3) == 5


def test_get_file_size_mb_on_a_real_temp_file(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("x" * 1024 * 1024)

    size_mb = get_file_size_mb(test_file)
    assert size_mb == pytest.approx(1.0, abs=0.01)