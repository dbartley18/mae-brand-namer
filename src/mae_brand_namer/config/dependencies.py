"""Dependency injection configuration."""

import os
from typing import Optional
from dataclasses import dataclass
from supabase import create_client, Client as SupabaseClient
from langsmith import Client as LangSmithClient

from .settings import settings
from ..models.app_config import AppConfig

@dataclass
class Dependencies:
    """Container for application dependencies."""
    
    supabase: SupabaseClient
    langsmith: Optional[LangSmithClient]
    app_config: AppConfig

def create_dependencies() -> Dependencies:
    """Create and configure application dependencies."""
    
    # Initialize Supabase client
    supabase = create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key  # This is already configured to use SUPABASE_SERVICE_KEY
    )
    
    # Initialize LangSmith client if tracing is enabled
    langsmith = None
    if settings.langchain_tracing_v2:
        langsmith = LangSmithClient(
            api_url=settings.langchain_endpoint,
            api_key=settings.langchain_api_key,
            project_name=settings.langchain_project
        )
    
    # Initialize AppConfig
    app_config = AppConfig()
    
    return Dependencies(
        supabase=supabase,
        langsmith=langsmith,
        app_config=app_config
    ) 