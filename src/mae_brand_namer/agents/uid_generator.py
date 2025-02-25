from typing import Dict, List, Tuple, Optional
from uuid import uuid4
from datetime import datetime

from langchain.agents import AgentExecutor
from langchain_core.agents import AgentFinish
from langchain_core.messages import HumanMessage

from ..models.state import BrandNameGenerationState

class UIDGeneratorAgent:
    """Agent responsible for generating unique run IDs for the workflow."""
    
    def __init__(self):
        """Initialize the UIDGeneratorAgent."""
        self.role = "Unique ID Generator"
        self.goal = "Generate unique identifiers for workflow tracking"
    
    @staticmethod
    def generate_run_id(prefix: str = "brand") -> str:
        """Generate a unique run ID with timestamp and UUID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_id}"
    
    def invoke(self, state: BrandNameGenerationState) -> Tuple[BrandNameGenerationState, List[Dict]]:
        """
        Generate a unique run ID and update the state.
        
        Args:
            state (BrandNameGenerationState): Current workflow state
            
        Returns:
            Tuple[BrandNameGenerationState, List[Dict]]: Updated state and any messages/logs
        """
        try:
            # Generate the run ID
            run_id = self.generate_run_id()
            
            # Update state
            state.run_id = run_id
            state.current_step = "brand_context"
            
            # Return updated state and success message
            return state, [{
                "role": "assistant",
                "content": f"Successfully generated run ID: {run_id}"
            }]
            
        except Exception as e:
            # Handle any errors
            error_msg = f"Error generating run ID: {str(e)}"
            state.errors.append({
                "step": "generate_uid",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            return state, [{
                "role": "assistant",
                "content": error_msg
            }] 