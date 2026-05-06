"""
Application configuration using pydantic-settings.

This module provides validated configuration management with environment variable support.
All configuration is type-checked at startup, providing clear error messages for missing
or invalid values.

Usage:
    from config import settings
    
    print(settings.arize_api_key)
    print(settings.is_production)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation.
    
    Environment variables are automatically loaded. Use a .env file for local development.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ============================================================================
    # API Keys
    # ============================================================================
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key for Claude")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    brave_api_key: Optional[str] = Field(default=None, description="Brave Search API key")
    
    # ============================================================================
    # Arize / Observability
    # ============================================================================
    arize_api_key: Optional[str] = Field(default=None, description="Arize platform API key")
    arize_space_id: Optional[str] = Field(default=None, description="Arize space identifier")
    arize_trace_enabled: bool = Field(default=True, description="Enable Arize tracing")
    arize_trace_debug: bool = Field(default=False, description="Enable verbose trace debugging")
    
    # ============================================================================
    # GCP / BigQuery
    # ============================================================================
    gcp_project_id: Optional[str] = Field(default=None, description="GCP project ID")
    gcp_credentials_base64: Optional[str] = Field(
        default=None, 
        description="Base64-encoded GCP service account JSON"
    )
    bigquery_dataset: str = Field(default="arize_gtm", description="BigQuery dataset name")
    
    # ============================================================================
    # External Services
    # ============================================================================
    gong_mcp_url: Optional[str] = Field(default=None, description="Gong MCP server URL")
    
    # ============================================================================
    # Application
    # ============================================================================
    api_service_mode: Literal["full", "light", "crew"] = Field(
        default="full",
        description="API service mode: full (all routes), light (no crew), crew (crew only)"
    )
    port: int = Field(default=8080, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")
    debug: bool = Field(default=False, description="Debug mode")
    
    # ============================================================================
    # URLs
    # ============================================================================
    stillness_web_url: Optional[str] = Field(
        default=None,
        description="URL of the Stillness web app for redirects"
    )
    public_web_app_url: Optional[str] = Field(
        default=None,
        description="Alternative URL for web app redirects"
    )
    
    # ============================================================================
    # Deployment
    # ============================================================================
    vercel: Optional[str] = Field(default=None, description="Set when running on Vercel")
    railway_environment: Optional[str] = Field(default=None, description="Railway environment name")
    
    # ============================================================================
    # Computed Properties
    # ============================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in a production environment."""
        return bool(self.vercel or self.railway_environment)
    
    @property
    def is_vercel(self) -> bool:
        """Check if running on Vercel."""
        return bool(self.vercel)
    
    @property
    def has_llm_api_key(self) -> bool:
        """Check if any LLM API key is configured."""
        return bool(self.anthropic_api_key or self.openai_api_key)
    
    @property
    def has_arize_config(self) -> bool:
        """Check if Arize is fully configured."""
        return bool(self.arize_api_key and self.arize_space_id)
    
    @property
    def web_app_redirect_url(self) -> Optional[str]:
        """Get the URL to redirect API root requests to."""
        url = self.stillness_web_url or self.public_web_app_url
        return url.rstrip("/") if url else None
    
    @field_validator("api_service_mode", mode="before")
    @classmethod
    def normalize_service_mode(cls, v: str) -> str:
        """Normalize service mode to lowercase."""
        return str(v).lower() if v else "full"
    
    @field_validator("arize_trace_debug", mode="before")
    @classmethod
    def parse_bool_env(cls, v) -> bool:
        """Parse boolean from environment variable."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes")
        return False


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Settings are loaded once and cached for the application lifetime.
    """
    return Settings()


# Convenience export for direct import
settings = get_settings()
