"""
thresholds.py

Loads the "judgment call" values from default.yaml.
"""

from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent
_DEFAULT_YAML_PATH = _CONFIG_DIR / "default.yaml"


@lru_cache(maxsize=1)
def get_thresholds() -> dict:
    with open(_DEFAULT_YAML_PATH, "r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)