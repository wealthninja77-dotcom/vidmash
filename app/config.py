"""
Application configuration using pydantic-settings.

Loads settings from environment variables and .env file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    GEMINI_API_KEY: str = ""
    OUTPUT_DIR: str = "./output"
    MAX_VIDEO_DURATION: int = 14400  # 4 hours in seconds
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DB_PATH: str = "./vidmash.db"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()
