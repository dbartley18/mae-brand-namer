"""Structured logging configuration for the brand naming workflow."""

import sys
import logging
from typing import Any, Dict, Optional
import structlog
from structlog.stdlib import BoundLogger
import os

def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Set up structured logging configuration.
    
    Args:
        level (str): Log level (DEBUG, INFO, WARNING, ERROR)
        json_format (bool): Whether to output logs in JSON format
        log_file (Optional[str]): Path to log file, if file logging is desired
    """
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )
    
    # Add file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        BoundLogger: Configured structured logger
    """
    return structlog.get_logger(name)

# Set up logging with default configuration
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=os.getenv("LOG_JSON", "true").lower() == "true",
    log_file=os.getenv("LOG_FILE")
) 