"""Helper utilities for working with agents."""

import os
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from ..config.settings import settings
from ..models.app_config import AppConfig

# Type variable for generic agent class
T = TypeVar('T')

def configure_agent_llm(agent_name: str, app_config: Optional[AppConfig] = None) -> ChatGoogleGenerativeAI:
    """Configure an LLM for a specific agent with the appropriate temperature.
    
    Args:
        agent_name (str): The name of the agent to configure the LLM for
        app_config (Optional[AppConfig], optional): App configuration. Defaults to None.
        
    Returns:
        ChatGoogleGenerativeAI: A configured LLM instance
    """
    # Initialize AppConfig if not provided
    config = app_config or AppConfig()
    
    # Get the temperature for this agent
    temperature = config.get_temperature_for_agent(agent_name)
    
    # Initialize LangSmith tracer if enabled
    tracer = None
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        from langchain_core.tracers import LangChainTracer
        tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT", "mae-brand-namer")
        )
    
    # Initialize and return the LLM
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        temperature=temperature,
        google_api_key=os.getenv("GEMINI_API_KEY") or settings.google_api_key,
        convert_system_message_to_human=True,
        callbacks=[tracer] if tracer else None
    )

def create_agent(agent_class: Type[T], agent_name: str, **kwargs) -> T:
    """Create an agent with the appropriate configuration.
    
    Args:
        agent_class (Type[T]): The class of the agent to create
        agent_name (str): The name of the agent (for temperature configuration)
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        T: An instance of the specified agent class
    """
    # Initialize AppConfig
    app_config = kwargs.get('app_config') or AppConfig()
    kwargs['app_config'] = app_config
    
    # Get the agent-specific temperature
    temperature = app_config.get_temperature_for_agent(agent_name)
    
    # Log the agent creation with its temperature
    logging.getLogger(__name__).info(
        f"Creating agent {agent_name} with temperature {temperature}",
        extra={"agent": agent_name, "temperature": temperature}
    )
    
    # Create and return the agent
    return agent_class(**kwargs) 