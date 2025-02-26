# Mae Brand Namer Style Guide

This document outlines the coding and documentation standards for the Mae Brand Namer project. Following these guidelines ensures consistency, readability, and maintainability across the codebase.

## Code Style

### Import Organization (PEP 8)

Organize imports in the following order, with a blank line between each group:

1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

```python
# Standard library imports
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Third-party imports
from supabase import create_client
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Local application imports
from ..config.settings import settings
from ..utils.logging import get_logger
```

### Line Length

- Maximum line length is 88 characters (extended from PEP 8's 79 characters)
- For long Field declarations, use a multi-line format:

```python
# Instead of this:
field_name: str = Field(..., description="This is a very long description that exceeds the line length limit")

# Do this:
field_name: str = Field(
    ..., 
    description="This is a very long description that exceeds the line length limit"
)
```

### Naming Conventions

- `snake_case` for variables, functions, and methods
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Avoid single-letter variable names except for counters and iterators

### Type Annotations

- All function parameters and return values should have type annotations
- Use `Optional[Type]` for parameters that can be None
- Use `Any` sparingly and only when necessary
- Use `TypedDict` for dictionary types with known structure

## Documentation Standards

### Module Docstrings

Every module should have a docstring that describes its purpose:

```python
"""
Module for handling brand context extraction and analysis.

This module provides functionality to extract and analyze brand context
from user prompts, including brand identity, values, and target audience.
"""
```

### Class Docstrings

Class docstrings should include:
- General description of the class
- Attributes section listing all class attributes
- Examples section showing basic usage

```python
class BrandContextExpert:
    """
    Expert in analyzing and extracting brand context from user input.
    
    This class provides functionality to extract structured brand information
    from unstructured text, including brand values, personality traits,
    target audience, and other key brand elements.
    
    Attributes:
        supabase: Connection manager for Supabase storage
        system_prompt: System prompt for the LLM
        extraction_prompt: Prompt template for extracting brand context
        llm: LLM instance configured for context extraction
    
    Example:
        ```python
        expert = BrandContextExpert()
        brand_context = await expert.extract_brand_context(user_prompt, run_id)
        print(brand_context["brand_values"])
        ```
    """
```

### Function Docstrings

Function docstrings should include:
- Description of the function
- Args section listing all parameters
- Returns section describing return values
- Raises section listing exceptions
- Examples section for complex functions

```python
def create_app_state(user_prompt: str, run_id: Optional[str] = None) -> AppState:
    """
    Initialize a new AppState instance with default values.
    
    Args:
        user_prompt: The user's original prompt text
        run_id: Unique identifier for this workflow run. If None, a new UUID is generated
        
    Returns:
        A new AppState instance with initialized fields
        
    Raises:
        ValueError: If user_prompt is empty
        
    Example:
        ```python
        state = create_app_state("Create a brand name for a tech startup")
        print(state["run_id"])  # Prints the generated UUID
        ```
    """
```

### Error Handling Documentation

Document error handling patterns and recovery strategies:

```python
try:
    # Operation that might fail
    result = potentially_failing_function()
except SpecificError as e:
    # Document recovery strategy in a comment
    # Example: Retry the operation with default parameters
    logger.error(
        "Operation failed",
        extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "action": "Using default parameters"
        }
    )
    result = fallback_function()
```

## Common Code Patterns

### Structured Logging

Use structured logging with the `extra` parameter:

```python
logger.info(
    "Processing brand context",
    extra={
        "run_id": run_id,
        "user_prompt_length": len(user_prompt)
    }
)
```

### State Management

Use immutable state updates:

```python
def update_app_state(app_state: AppState, updates: Dict[str, Any]) -> AppState:
    """Update AppState with new values."""
    return {**app_state, **updates}
```

### Error Handling

Use specific exception types and handle them explicitly:

```python
try:
    # Operation code
except APIError as e:
    logger.error(
        "Supabase API error",
        extra={
            "run_id": run_id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "status_code": getattr(e, "code", None)
        }
    )
    raise
except Exception as e:
    logger.error(
        "Unexpected error",
        extra={
            "run_id": run_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    )
    raise ValueError(f"Operation failed: {str(e)}")
```

## Testing

- Write tests for all public functions and methods
- Use descriptive test names that explain the scenario and expected outcome
- Mock external dependencies 