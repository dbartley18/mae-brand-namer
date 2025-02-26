# Error Handling Template for Mae Brand Namer Agents

This template provides standardized error handling patterns to be applied consistently across all agent files in the Mae Brand Namer application.

## Imports to Add

```python
from postgrest import APIError
from ..utils.logging import get_logger

logger = get_logger(__name__)
```

## Method Template with Error Handling

```python
async def method_name(self, run_id: str, other_params):
    """
    Method description.
    
    Args:
        run_id: Unique identifier for this workflow run
        other_params: Description of other parameters
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: If the operation fails
        APIError: If there's an error with the Supabase API
    """
    try:
        logger.info(
            "Starting operation",
            extra={"run_id": run_id}
        )
        
        # Method implementation here
        
        logger.info(
            "Operation completed successfully",
            extra={
                "run_id": run_id,
                # Add other relevant info
            }
        )
        
        return result
        
    except APIError as e:
        logger.error(
            "Supabase API error",
            extra={
                "run_id": run_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status_code": getattr(e, "code", None),
                "details": getattr(e, "details", None)
            }
        )
        raise
        
    except Exception as e:
        logger.error(
            "Error in operation",
            extra={
                "run_id": run_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise ValueError(f"Operation failed: {str(e)}")
```

## LLM Call Error Handling

```python
try:
    # LLM call preparation
    formatted_prompt = prompt_template.format(...)
    
    # LLM invocation
    response = await self.llm.ainvoke([...])
    
    # Response parsing
    parsed_result = self.output_parser.parse(response.content)
    
    logger.info(
        "LLM call successful",
        extra={
            "run_id": run_id,
            # Add other relevant info
        }
    )
    
except Exception as e:
    logger.error(
        "Error in LLM call",
        extra={
            "run_id": run_id,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
    )
    raise ValueError(f"LLM call failed: {str(e)}")
```

## Supabase Storage Error Handling

```python
async def _store_in_supabase(self, run_id: str, data: Dict[str, Any]) -> None:
    """
    Store data in Supabase.
    
    Args:
        run_id: Unique identifier for this workflow run
        data: Data to store
        
    Raises:
        APIError: If there's an error with the Supabase API
        ValueError: If there's an error with data validation
    """
    try:
        # Prepare data for storage
        supabase_data = {
            "run_id": run_id,
            # Other fields here
        }
        
        # Store in Supabase
        await supabase.execute_with_retry(
            operation="insert",
            table="table_name",
            data=supabase_data
        )
        
        logger.info(
            "Data stored in Supabase",
            extra={
                "run_id": run_id,
                # Other relevant info
            }
        )
        
    except KeyError as e:
        logger.error(
            "Missing key in data",
            extra={
                "run_id": run_id,
                "error_type": "KeyError",
                "missing_key": str(e)
            }
        )
        raise ValueError(f"Missing required field: {str(e)}")
        
    except (TypeError, ValueError) as e:
        logger.error(
            "Invalid data type",
            extra={
                "run_id": run_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise ValueError(f"Invalid data: {str(e)}")
        
    except APIError as e:
        logger.error(
            "Supabase API error",
            extra={
                "run_id": run_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status_code": getattr(e, "code", None),
                "details": getattr(e, "details", None)
            }
        )
        raise
        
    except Exception as e:
        logger.error(
            "Unexpected error storing data",
            extra={
                "run_id": run_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise
```

## Implementation Guidelines

1. Always use structured logging with the `extra` parameter
2. Log both start and successful completion of operations
3. Catch specific exception types before general exceptions
4. Always include `run_id` in log messages for traceability
5. Re-raise APIErrors to allow proper handling up the call stack
6. For other exceptions, consider whether to raise a custom error or re-raise
7. Include both error type and message in log entries

## Example Updated Agent

```python
"""Agent description."""

import os
from typing import Dict, Any
from datetime import datetime
from postgrest import APIError

from ..utils.logging import get_logger
from ..utils.supabase_utils import supabase

logger = get_logger(__name__)

class ExampleAgent:
    """Agent description."""
    
    def __init__(self):
        """Initialize the agent."""
        try:
            # Initialization code
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(
                "Failed to initialize agent",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    async def process_data(self, run_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data.
        
        Args:
            run_id: Unique identifier for this workflow run
            data: Input data to process
            
        Returns:
            Processed data
            
        Raises:
            ValueError: If processing fails
        """
        try:
            logger.info(
                "Starting data processing",
                extra={"run_id": run_id}
            )
            
            # Processing code
            
            result = {"status": "success", "data": processed_data}
            
            logger.info(
                "Data processing completed",
                extra={"run_id": run_id}
            )
            
            return result
            
        except APIError as e:
            logger.error(
                "Supabase API error in data processing",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            raise
            
        except Exception as e:
            logger.error(
                "Error processing data",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise ValueError(f"Data processing failed: {str(e)}") 