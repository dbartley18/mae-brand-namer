"""
Utility functions for working with state objects in the Mae Brand Namer application.

This module provides helper functions for creating, manipulating, and converting
between different state object types used in the application. It facilitates:
- Creating new application state instances with proper initialization
- Converting between state object types for backward compatibility
- Immutably updating state with new values

Example:
    ```python
    # Create a new application state
    state = create_app_state(user_prompt="Create a brand name for a tech startup")
    
    # Update state with new values
    updated_state = update_app_state(
        state, 
        {
            "brand_values": ["innovation", "reliability", "transparency"],
            "brand_personality": ["confident", "innovative", "friendly"]
        }
    )
    
    # Access state values
    run_id = updated_state["run_id"]
    brand_values = updated_state["brand_values"]
    ```
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from ..models.state import AppState, BrandNameGenerationState

def create_app_state(user_prompt: str, run_id: Optional[str] = None, client: Optional[Any] = None) -> AppState:
    """
    Create a new AppState instance with initialized fields.
    
    This function initializes an AppState TypedDict with all required fields.
    If no run_id is provided, a unique ID is automatically generated.
    
    Args:
        user_prompt: The user's initial prompt for brand naming
        run_id: Optional run ID, generated if not provided
        client: Optional LangGraph/LangSmith client for tracing and async operations
        
    Returns:
        AppState: An initialized AppState instance with empty collections
        
    Example:
        ```python
        # Create state with auto-generated run_id
        state = create_app_state("Create a brand name for a luxury watch company")
        
        # Create state with specific run_id
        state = create_app_state(
            user_prompt="Create a brand name for a fitness app",
            run_id="brand-custom-id"
        )
        
        # Create state with LangGraph client
        from langsmith import Client
        client = Client()
        state = create_app_state(
            user_prompt="Create a brand name for a tech startup",
            client=client
        )
        ```
    """
    if not run_id:
        run_id = f"brand-{uuid4().hex[:8]}"
        
    return {
        "run_id": run_id,
        "user_prompt": user_prompt,
        "errors": [],
        "client": client,
        
        # Initialize empty collections
        "brand_values": [],
        "brand_personality": [],
        "customer_needs": [],
        "industry_trends": [],
        "generated_names": [],
        "naming_categories": [],
        "brand_personality_alignments": [],
        "brand_promise_alignments": [],
        "target_audience_relevance": [],
        "market_differentiation": [],
        "memorability_scores": [],
        "pronounceability_scores": [],
        "visual_branding_potential": [],
        "name_rankings": [],
        "shortlisted_names": [],
        
        # Initialize empty dictionaries
        "task_statuses": {},
        "linguistic_analysis_results": {},
        "cultural_analysis_results": {},
        "evaluation_results": {},
        "competitor_analysis_results": {},
    }

def convert_to_brand_name_generation_state(app_state: AppState) -> BrandNameGenerationState:
    """
    Convert an AppState to a BrandNameGenerationState for compatibility with existing code.
    
    This function transforms the newer TypedDict-based AppState into the Pydantic
    BrandNameGenerationState model to maintain compatibility with components that
    expect the Pydantic model.
    
    Args:
        app_state (AppState): The AppState instance to convert
        
    Returns:
        BrandNameGenerationState: A Pydantic model instance with data from the AppState
        
    Raises:
        ValueError: If required fields are missing from the AppState
        
    Example:
        ```python
        # Create a new state
        app_state = create_app_state("Create a brand name for a coffee shop")
        
        # Convert to legacy Pydantic model for compatibility
        pydantic_state = convert_to_brand_name_generation_state(app_state)
        
        # Now you can use methods of the Pydantic model
        pydantic_state_dict = pydantic_state.model_dump()
        ```
    """
    # Check required fields
    if not app_state.get("run_id"):
        raise ValueError("Missing required field 'run_id' in AppState")
    if not app_state.get("user_prompt"):
        raise ValueError("Missing required field 'user_prompt' in AppState")
    
    # Convert errors from TypedDict to dict
    errors = [dict(error) for error in app_state.get("errors", [])]
    
    # Create BrandNameGenerationState instance
    return BrandNameGenerationState(
        run_id=app_state["run_id"],
        user_prompt=app_state["user_prompt"],
        errors=errors,
        client=app_state.get("client"),
        # Add other fields as needed...
    )

def update_app_state(app_state: AppState, updates: Dict[str, Any]) -> AppState:
    """
    Update an AppState with new values in an immutable way.
    
    This function creates a new AppState instance by merging the original state
    with the provided updates, following the immutable data pattern where state
    is never modified in-place.
    
    Args:
        app_state (AppState): The original AppState instance
        updates (Dict[str, Any]): Dictionary of field updates to apply
        
    Returns:
        AppState: A new AppState instance with updated values
        
    Example:
        ```python
        # Initial state
        state = create_app_state("Create a brand name for an eco-friendly product")
        
        # Update state with brand context
        state = update_app_state(state, {
            "brand_identity_brief": "An eco-conscious brand focused on sustainability",
            "brand_values": ["sustainability", "transparency", "quality"]
        })
        
        # Later, update with generated names
        state = update_app_state(state, {
            "generated_names": ["EcoVital", "GreenLife", "NaturePure"]
        })
        ```
    """
    # Create a new dict by merging the old state with updates
    return {**app_state, **updates} 