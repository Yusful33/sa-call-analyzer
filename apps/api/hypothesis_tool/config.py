"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # BigQuery
    bq_project_id: str = "mkt-analytics-268801"
    google_application_credentials: str | None = None

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # Web search
    brave_api_key: str = ""

    # Arize Observability
    arize_space_id: str = ""
    arize_api_key: str = ""
    arize_project_name: str = "ae-hypothesis-agent"
    # Plan-research prompt: fetched via GraphQL (app.arize.com/graphql). Use prompt (container) ID.
    arize_plan_research_prompt_id: str = "UHJvbXB0OjMwNTI3Om9BbWo="

    # Database
    database_url: str = "sqlite:///feedback.db"

    # App settings
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
