"""Configuration management for the brand naming workflow."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
import json

class LangGraphConfig(BaseSettings):
    """LangGraph configuration settings."""
    
    # LangSmith configuration
    langsmith_api_key: str = Field(env="LANGCHAIN_API_KEY")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_project: str = Field(default="mae-brand-namer")
    tracing_enabled: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    
    # Model configuration
    model_name: str = Field(default="gemini-1.5-pro", env="MODEL_NAME")
    model_temperature: float = Field(default=0.7)
    google_api_key: str = Field(env="GEMINI_API_KEY")
    
    # Supabase configuration
    supabase_url: str = Field(env="SUPABASE_URL")
    supabase_key: str = Field(env="SUPABASE_SERVICE_KEY")
    
    # Retry configuration
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: int = Field(default=1, env="RETRY_DELAY")
    retry_backoff: int = Field(default=2, env="RETRY_BACKOFF")
    retry_max_delay: int = Field(default=60, env="RETRY_MAX_DELAY")
    
    # Report configuration
    report_template_path: str = Field(default="./templates/report_template.md")
    report_output_format: str = Field(default="docx")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True

    @property
    def get_langsmith_client(self) -> Client:
        """Get a configured LangSmith client."""
        from langsmith import Client
        return Client(
            api_url=self.langsmith_endpoint,
            api_key=self.langsmith_api_key,
            project_name=self.langsmith_project
        )

# Global configuration instance
settings = LangGraphConfig()

def load_langgraph_config() -> Dict[str, Any]:
    """Load LangGraph configuration from langgraph.json."""
    try:
        with open("langgraph.json", "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading LangGraph configuration: {str(e)}")

# Load LangGraph configuration
langgraph_config = load_langgraph_config() 