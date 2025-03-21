"""
Custom LangGraph API server initialization.

This module provides functions to initialize and configure the LangGraph API server,
including health check endpoints for container orchestration.
"""

import logging
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI

logger = logging.getLogger(__name__)

def on_startup():
    """Function to run on server startup."""
    logger.info("Server starting up")
    # Add any additional startup logic here

def on_shutdown():
    """Function to run on server shutdown."""
    logger.info("Server shutting down")
    # Add any cleanup logic here

def init_app() -> FastAPI:
    """
    Initialize and configure the FastAPI application.
    
    This function will be called by LangGraph API server during startup.
    
    Returns:
        FastAPI: The configured FastAPI application
    """
    # Create a new FastAPI application
    app = FastAPI()
    
    # Add health check endpoints
    @app.get("/health")
    async def health():
        """Simple health check endpoint that always returns OK."""
        logger.debug("Health check request received")
        return {"status": "ok"}
    
    @app.get("/ready")
    async def ready():
        """Readiness check endpoint that returns OK when the server is ready to accept requests."""
        logger.debug("Readiness check request received")
        return {"status": "ready"}
    
    # Set up event handlers
    app.add_event_handler("startup", on_startup)
    app.add_event_handler("shutdown", on_shutdown)
    
    # Log initialization
    logger.info("Custom server initialization complete")
    
    return app

def get_server_config() -> Dict[str, Any]:
    """
    Get server configuration for LangGraph.
    
    Returns:
        Dict[str, Any]: Configuration dictionary for the server
    """
    return {
        "app_init": init_app,
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "8000")),
        "workers": int(os.getenv("WORKERS", "1")),
        "log_level": os.getenv("LOG_LEVEL", "info"),
        "timeout": int(os.getenv("TIMEOUT", "120")),
    } 