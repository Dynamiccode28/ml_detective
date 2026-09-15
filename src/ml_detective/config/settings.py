"""
settings.py

Defines the "Settings" class: a single, validated source of truth
for every environment-specific value the project needs.
"""

from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    database_url: str = "sqlite:///./ml_detective_dev.db"
    llm_provider: str = ""
    llm_api_key: str = ""
    log_level: str = "INFO"


settings = Settings()