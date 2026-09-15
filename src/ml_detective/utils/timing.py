"""
timing.py

A small, reusable tool for measuring how long a function takes to run.
"""

import functools
import time
from typing import Any, Callable

from ml_detective.utils.logger import get_logger

logger = get_logger(__name__)


def timeit(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_seconds = time.perf_counter() - start_time
        logger.debug(f"{func.__name__} took {elapsed_seconds:.4f} seconds")
        return result

    return wrapper