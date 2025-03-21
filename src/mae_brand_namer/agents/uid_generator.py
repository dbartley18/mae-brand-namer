from typing import Dict, List, Tuple, Optional
from uuid import uuid4
from datetime import datetime
import asyncio

from langchain_core.messages import HumanMessage

from ..models.state import BrandNameGenerationState
from ..utils.logging import get_logger

logger = get_logger(__name__)

class UIDGeneratorAgent:
    """Agent responsible for generating unique run IDs for the workflow."""
    
    def __init__(self):
        """Initialize the UIDGeneratorAgent."""
        self.role = "Unique ID Generator"
        self.goal = "Generate unique identifiers for workflow tracking"
    
    @staticmethod
    def generate_run_id(prefix: str = "brand") -> str:
        """Generate a unique run ID with timestamp and UUID.
        
        Args:
            prefix (str, optional): Prefix for the run ID. Defaults to "brand".
            
        Returns:
            str: A unique run ID in the format "{prefix}_{timestamp}_{unique_id}"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_id}"

    async def generate_uid(self) -> str:
        """Generate a unique run ID.
        
        Returns:
            str: A unique run ID generated by generate_run_id method.
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            return self.generate_run_id()
        except Exception as e:
            logger.error(
                "Error generating unique run ID",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    async def invoke(self, state: BrandNameGenerationState) -> Tuple[BrandNameGenerationState, List[Dict]]:
        """
        Generate a unique run ID and update the state.
        
        Args:
            state (BrandNameGenerationState): Current workflow state
            
        Returns:
            Tuple[BrandNameGenerationState, List[Dict]]: Updated state and any messages/logs
        """
        try:
            # Generate the run ID using generate_uid method which includes error handling
            prefix = getattr(state, "prefix", "brand")
            # Temporarily store the prefix in the static method's default parameter
            self.generate_run_id.__defaults__ = (prefix,)
            run_id = await self.generate_uid()
            
            # Update state
            state.run_id = run_id
            state.current_step = "brand_context"
            
            # Log the generated run ID
            logger.info(
                "Generated unique run ID",
                extra={"run_id": run_id}
            )
            
            # Return updated state and success message
            return state, [{
                "role": "assistant",
                "content": f"Successfully generated run ID: {run_id}"
            }]
            
        except Exception as e:
            # Log the error with detailed information
            logger.error(
                "Error in UIDGeneratorAgent",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            # Add error to state
            error_msg = f"Error generating run ID: {str(e)}"
            state.errors.append({
                "step": "generate_uid",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            # Re-raise the exception after logging
            raise 