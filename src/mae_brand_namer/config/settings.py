"""Configuration management for the brand naming workflow.

This module handles configuration for the LangGraph-based brand naming workflow,
including LangSmith integration, model settings, and database connections.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuration settings for the brand naming workflow."""
    
    # LangSmith configuration
    langchain_api_key: Optional[str] = None
    langchain_project: str = "mae-brand-namer"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_tracing_v2: bool = True
    
    # Model configuration
    model_name: str = "gemini-1.5-pro"
    model_temperature: float = 0.7
    gemini_api_key: Optional[str] = None
    
    # Add google_api_key to point to gemini_api_key for compatibility
    @property
    def google_api_key(self) -> Optional[str]:
        """Return the Gemini API key to maintain compatibility with google_api_key references."""
        return self.gemini_api_key
    
    # Supabase configuration
    supabase_url: Optional[str] = None
    supabase_service_key: Optional[str] = None
    supabase_timeout: int = 10
    
    # Report configuration
    template_dir: str = "./templates"
    report_template: str = "report_template.md"
    output_format: str = "docx"
    output_dir: str = "./reports"
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: int = 1
    retry_backoff: int = 2
    retry_max_delay: int = 60
    
    # Graph configuration
    graph_name: str = "brand-naming-workflow"
    graph_description: str = "Brand naming workflow using expert agents"
    
    # Helper methods
    def get_langsmith_callbacks(self):
        """Return LangSmith callbacks if tracing is enabled."""
        if self.langchain_tracing_v2:
            from langchain_core.tracers import LangChainTracer
            try:
                return [LangChainTracer(project_name=self.langchain_project)]
            except:
                return None
        return None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Global settings instance
settings = Settings() 