"""Configuration: everything that differs between environments comes from env vars.

12-factor rule: the same Docker image runs in local, dev, staging and prod.
Only the environment (env vars, mounted secrets) changes.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    app_env: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"

    database_url: str = "sqlite:///./adlens.db"
    redis_url: str | None = None
    cache_ttl_s: int = 86_400

    llm_provider: Literal["fake", "openai_compat"] = "fake"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("not-needed")  # SecretStr: never printed in logs/reprs
    llm_model: str = "llama3.1:8b"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    prompt_version: Literal["v1", "v2"] = "v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
