"""Dependency injection configuration."""

from typing import Optional
from dataclasses import dataclass
from supabase import create_client, Client as SupabaseClient
from langsmith import Client as LangSmithClient

from .settings import settings

@dataclass
class Dependencies:
    """Container for application dependencies."""
    
    supabase: SupabaseClient
    langsmith: Optional[LangSmithClient]

def create_dependencies() -> Dependencies:
    """Create and configure application dependencies."""
    
    # Initialize Supabase client
    supabase = create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key  # This is already configured to use SUPABASE_SERVICE_KEY
    )
    
    # Initialize LangSmith client if enabled
    langsmith = None
    if settings.tracing_enabled:
        langsmith = LangSmithClient(
            api_url=settings.langsmith_endpoint,
            api_key=settings.langsmith_api_key,
            project_name=settings.langsmith_project
        )
    
    return Dependencies(
        supabase=supabase,
        langsmith=langsmith
    ) 