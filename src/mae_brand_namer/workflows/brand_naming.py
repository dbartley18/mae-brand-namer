"""Brand naming workflow using LangGraph."""

from typing import Dict, Any, List, Optional, TypedDict, Tuple, Union, Callable, TypeVar
from datetime import datetime
import asyncio
from functools import partial
import json
import os
import uuid
import traceback

from langgraph.graph import StateGraph
from langsmith import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import SystemMessage, HumanMessage
from postgrest import APIError as PostgrestError
from postgrest import APIError  # Add direct import of APIError for compatibility
from langgraph.constants import Send
from pydantic import BaseModel, Field, ConfigDict

from mae_brand_namer.agents import (
    UIDGeneratorAgent,
    BrandContextExpert,
    BrandNameCreationExpert,
    LinguisticsExpert,
    SemanticAnalysisExpert,
    CulturalSensitivityExpert,
    TranslationAnalysisExpert,
    DomainAnalysisExpert,
    SEOOnlineDiscoveryExpert,
    CompetitorAnalysisExpert,
    SurveySimulationExpert,
    MarketResearchExpert,
    ReportCompiler,
    ReportStorer,
    ProcessSupervisor,
    BrandNameEvaluator
)
from mae_brand_namer.utils.logging import get_logger
from mae_brand_namer.utils.supabase_utils import SupabaseManager
from mae_brand_namer.config.settings import settings
from mae_brand_namer.config.dependencies import Dependencies, create_dependencies
from mae_brand_namer.models.state import BrandNameGenerationState

logger = get_logger(__name__)

# Mapping of node names to agent types and task names for process monitoring
node_to_agent_task = {
    "generate_uid": ("UIDGeneratorAgent", "Generate_UID"),
    "understand_brand_context": ("BrandContextExpert", "Understand_Brand_Context"),
    "generate_brand_names": ("BrandNameCreationExpert", "Generate_Brand_Names"),
    "process_linguistics": ("LinguisticsExpert", "Analyze_Linguistics"),
    "process_cultural_sensitivity": ("CulturalSensitivityExpert", "Analyze_Cultural_Sensitivity"),
    "process_analyses": ("AnalysisCoordinator", "Process_Analyses"),
    "process_evaluation": ("BrandNameEvaluator", "Evaluate_Brand_Names"),
    "process_market_research": ("MarketResearchExpert", "Analyze_Market_Research"),
    "compile_report": ("ReportCompiler", "Compile_Report"),
    "store_report": ("ReportStorer", "Store_Report")
}

class ProcessSupervisorCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler that uses ProcessSupervisor to track node execution.
    This replaces the deprecated pre/post-processor and interrupt handler methods.
    """
    
    def __init__(self, supervisor: ProcessSupervisor = None, langsmith_client: Optional[Client] = None):
        """Initialize the callback handler with a ProcessSupervisor instance."""
        super().__init__()
        self.supervisor = supervisor or ProcessSupervisor()
        self.node_start_times = {}
        self.node_to_agent_task = node_to_agent_task
        self.langsmith_client = langsmith_client
        self.current_run_id = None
        self.current_node = None
        self.error_nodes = set()
    
    def _get_agent_task_info(self, node_name: str) -> Tuple[str, str]:
        """Get agent type and task name for a node."""
        return self.node_to_agent_task.get(node_name, ("Unknown", "Unknown"))
    
    def _extract_node_name(self, serialized: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        """Extract node name from metadata using multiple fallback methods."""
        # Method 1: From kwargs tags
        node_name = kwargs.get("tags", {}).get("node_name")
        
        # Method 2: From serialized tags
        if not node_name and serialized and isinstance(serialized, dict):
            tags = serialized.get("tags", {})
            if isinstance(tags, dict):
                node_name = tags.get("node_name")
        
        # Method 3: From serialized name (LangGraph specific)
        if not node_name and serialized and isinstance(serialized, dict):
            name = serialized.get("name", "")
            if isinstance(name, str) and ":" in name:
                # Extract node name from format like "StateGraph:node_name"
                parts = name.split(":")
                if len(parts) > 1:
                    node_name = parts[1]
        
        return node_name
    
    def _extract_run_id(self, inputs: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        """Extract run_id using multiple fallback methods."""
        # Method 1: Direct from inputs dict
        run_id = None
        if inputs and isinstance(inputs, dict):
            run_id = inputs.get("run_id")
        
        # Method 2: From a nested state object in different formats
        if not run_id and inputs and isinstance(inputs, dict):
            # Check for standard state dict
            state = inputs.get("state")
            if isinstance(state, dict):
                run_id = state.get("run_id")
            
            # Check for state that might be a Pydantic model
            state_obj = inputs.get("state")
            if not run_id and state_obj:
                # Try as attribute
                try:
                    if hasattr(state_obj, "run_id"):
                        run_id = getattr(state_obj, "run_id")
                except:
                    pass
                
                # Try as dict-like access
                try:
                    if "run_id" in state_obj:
                        run_id = state_obj["run_id"]
                except:
                    pass
        
        # Method 3: From the current instance variable if all else fails
        if not run_id:
            run_id = self.current_run_id
            
        # If we still don't have a run_id and this is a new execution, generate one
        if not run_id and not self.current_run_id:
            # Generate a fallback run_id
            run_id = f"generated_{uuid.uuid4().hex[:8]}"
            self.current_run_id = run_id
            logger.warning(f"Generated fallback run_id: {run_id}")
            
        return run_id
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log task start when a node starts execution."""
        try:
            # Extract node name and run_id from the metadata
            node_name = self._extract_node_name(serialized, **kwargs)
            run_id = self._extract_run_id(inputs, **kwargs)
            
            if not node_name:
                logger.debug("Could not extract node name in on_chain_start")
                return
                
            self.current_node = node_name
            
            if run_id:
                self.current_run_id = run_id
            
            # Store the start time for duration calculation
            self.node_start_times[node_name] = datetime.now()
            
            # Get agent type and task name
            agent_type, task_name = self._get_agent_task_info(node_name)
            
            logger.debug(
                f"Node execution starting",
                extra={
                    "node_name": node_name,
                    "run_id": run_id,
                    "agent_type": agent_type, 
                    "task_name": task_name
                }
            )
            
            # Log task start
            if run_id:
                await self.supervisor.log_task_start(
                    run_id=run_id,
                    agent_type=agent_type,
                    task_name=task_name
                )
                
            # Add LangSmith metadata if available
            if self.langsmith_client and run_id:
                try:
                    # Add metadata to LangSmith run if possible
                    self.langsmith_client.update_run(
                        run_id,
                        metadata={
                            "node_name": node_name,
                            "agent_type": agent_type,
                            "task_name": task_name,
                            "status": "in_progress"
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error updating LangSmith run: {e}")
                
        except Exception as e:
            logger.error(f"Error in on_chain_start callback: {str(e)}")
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log task completion when a node completes execution."""
        try:
            # Try to get the node name from kwargs, fallback to the current_node
            node_name = kwargs.get("tags", {}).get("node_name") or self.current_node
            
            # Try to get run_id from outputs, fallback to the current_run_id
            run_id = None
            if outputs and isinstance(outputs, dict):
                run_id = outputs.get("run_id")
                
                # Check if run_id is in a nested state
                if not run_id and "state" in outputs and isinstance(outputs["state"], dict):
                    run_id = outputs["state"].get("run_id")
            
            # Fallback to current_run_id if not found
            if not run_id:
                run_id = self.current_run_id
            
            if not node_name or not run_id:
                logger.debug("Missing node_name or run_id in on_chain_end")
                return
                
            # Get agent type and task name
            agent_type, task_name = self._get_agent_task_info(node_name)
            
            logger.debug(
                f"Node execution completed",
                extra={
                    "node_name": node_name,
                    "run_id": run_id,
                    "agent_type": agent_type, 
                    "task_name": task_name
                }
            )
            
            # Log task completion
            await self.supervisor.log_task_completion(
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name
            )
            
            # Add LangSmith metadata if available
            if self.langsmith_client and run_id:
                try:
                    # Add metadata to LangSmith run
                    self.langsmith_client.update_run(
                        run_id,
                        metadata={
                            "node_name": node_name,
                            "agent_type": agent_type,
                            "task_name": task_name,
                            "status": "completed"
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error updating LangSmith run: {e}")
                
        except Exception as e:
            logger.error(f"Error in on_chain_end callback: {str(e)}")
    
    async def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Log task error and determine if it should be retried."""
        try:
            # Try to get the node name from kwargs, fallback to the current_node
            node_name = kwargs.get("tags", {}).get("node_name") or self.current_node
            
            # Get inputs if available
            inputs = kwargs.get("inputs", {})
            
            # Try to get run_id from inputs, fallback to the current_run_id
            run_id = self._extract_run_id(inputs, **kwargs)
            
            if not node_name or not run_id:
                logger.debug("Missing node_name or run_id in on_chain_error")
                return
                
            # Add to set of error nodes
            self.error_nodes.add(node_name)
                
            # Get agent type and task name
            agent_type, task_name = self._get_agent_task_info(node_name)
            
            logger.error(
                f"Node execution error",
                extra={
                    "node_name": node_name,
                    "run_id": run_id,
                    "agent_type": agent_type, 
                    "task_name": task_name,
                    "error": str(error)
                }
            )
            
            # Log task error
            should_retry = await self.supervisor.log_task_error(
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name,
                error=error
            )
            
            # Add LangSmith metadata if available
            if self.langsmith_client and run_id:
                try:
                    # Add metadata to LangSmith run
                    self.langsmith_client.update_run(
                        run_id,
                        metadata={
                            "node_name": node_name,
                            "agent_type": agent_type,
                            "task_name": task_name,
                            "status": "error",
                            "error": str(error),
                            "should_retry": should_retry
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error updating LangSmith run: {e}")
            
            # If we should retry, log the retry information
            if should_retry:
                retry_count = self.supervisor._get_retry_count(run_id, agent_type, task_name)
                retry_delay = self.supervisor._calculate_retry_delay(retry_count)
                logger.info(
                    f"Scheduling retry in {retry_delay} seconds",
                    extra={
                        "node_name": node_name,
                        "run_id": run_id,
                        "agent_type": agent_type,
                        "task_name": task_name,
                        "retry_count": retry_count,
                        "retry_delay": retry_delay
                    }
                )
        except Exception as e:
            logger.error(f"Error in on_chain_error callback: {str(e)}")

async def throttle_step(delay_seconds: float = 1.0, label: str = "Unknown step"):
    """
    Simple utility function to throttle execution between workflow steps.
    
    Args:
        delay_seconds: Number of seconds to delay execution
        label: Label for the step being throttled (for logging)
    """
    if delay_seconds <= 0:
        return
        
    logger.info(f"Throttling before step: {label} (delay: {delay_seconds}s)")
    await asyncio.sleep(delay_seconds)
    logger.info(f"Continuing with step: {label}")

def create_workflow(
    config: dict = None
) -> StateGraph:
    """
    Create and configure the brand naming workflow graph.
    
    This function sets up the workflow state graph with all nodes, edges, and process monitoring.
    It uses callback handlers for task logging and error management.
    
    Args:
        config: RunnableConfig object containing configuration parameters.
            May include 'langsmith_client', 'default_step_delay', and 'step_delays'.
        
    Returns:
        Configured StateGraph for the brand naming workflow
    """
    # Extract parameters from config if provided
    if config is None:
        config = {}
    
    # Get configuration values with defaults
    configurable = config.get("configurable", {})
    langsmith_client = configurable.get("langsmith_client")
    default_step_delay = configurable.get("default_step_delay", 2.0)
    step_delays = configurable.get("step_delays")
    
    # Create a single Supabase manager instance
    supabase_manager = SupabaseManager()
    
    # Initialize supervisor for process monitoring
    supervisor = ProcessSupervisor(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client))
    
    # Initialize step_delays if None
    if step_delays is None:
        step_delays = {
            # Fast steps
            "generate_uid": 0.5,
            # Medium steps
            "understand_brand_context": 2.0,
            "compile_report": 2.0,
            "store_report": 2.0,
            # Slow steps that need more throttling
            "generate_brand_names": 3.0,
            "process_analyses": 5.0,  # This is a heavy step with multiple experts
            "process_evaluation": 3.0,
            "process_market_research": 3.0,
        }
    
    # Create workflow state graph
    workflow = StateGraph(BrandNameGenerationState)
    
    # Helper function to wrap async functions so they are properly handled by LangGraph
    def wrap_async_node(func, step_name: str = "unknown"):
        """Wraps an async function to ensure it's properly awaited before returning"""
        async def wrapper(state):
            # Get the appropriate delay for this step (or use default)
            step_delay = step_delays.get(step_name, default_step_delay)
            # Add throttling delay before executing the step
            await throttle_step(step_delay, step_name)
            # Execute the original function
            result = await func(state)
            return result
        return wrapper
    
    # Define agent nodes with proper async handling
    workflow.add_node("generate_uid", 
        wrap_async_node(process_uid, "generate_uid")
    )
    
    # Wrap process_brand_context and other async functions
    workflow.add_node("understand_brand_context", 
        wrap_async_node(lambda state: process_brand_context(state, BrandContextExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "understand_brand_context")
    )
    
    workflow.add_node("generate_brand_names", 
        wrap_async_node(lambda state: process_brand_names(state, BrandNameCreationExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "generate_brand_names")
    )
    
    # Use the original process_analyses function signature but with throttling
    workflow.add_node("process_analyses", 
        wrap_async_node(lambda state: process_analyses(state, [
            SemanticAnalysisExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)),
            LinguisticsExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)),
            CulturalSensitivityExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)),
            TranslationAnalysisExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)),
            DomainAnalysisExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)),
            SEOOnlineDiscoveryExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)),
            CompetitorAnalysisExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)),
            SurveySimulationExpert(dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client))
        ]), "process_analyses")
    )
    
    workflow.add_node("process_evaluation", 
        wrap_async_node(lambda state: process_evaluation(state, BrandNameEvaluator(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_evaluation")
    )
    
    workflow.add_node("process_market_research", 
        wrap_async_node(lambda state: process_market_research(state, MarketResearchExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_market_research")
    )
    
    workflow.add_node("compile_report", 
        wrap_async_node(lambda state: process_report(state, ReportCompiler(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "compile_report")
    )
    
    workflow.add_node("store_report", 
        wrap_async_node(lambda state: process_report_storage(state, ReportStorer()), "store_report")
    )
    
    # Define conditional edges
    workflow.add_edge("generate_uid", "understand_brand_context")
    workflow.add_edge("understand_brand_context", "generate_brand_names")
    workflow.add_edge("generate_brand_names", "process_analyses")
    workflow.add_edge("process_analyses", "process_evaluation")
    workflow.add_edge("process_evaluation", "process_market_research")
    workflow.add_edge("process_market_research", "compile_report")
    workflow.add_edge("compile_report", "store_report")
    
    # Define entry point
    workflow.set_entry_point("generate_uid")
    
    return workflow

async def process_uid(state: BrandNameGenerationState) -> Dict[str, Any]:
    """
    Generate a unique run ID for the workflow execution.
    
    Args:
        state: The current workflow state
        
    Returns:
        Dictionary with state updates including run_id, start_time, and status
    """
    try:
        # Try to get the running event loop, create one if it doesn't exist
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        agent = UIDGeneratorAgent()
        
        # Use tracing_v2_enabled without tags
        with tracing_v2_enabled():
            # Use getattr instead of .get() since state is a Pydantic model
            prefix = getattr(state, "prefix", "mae")
            
            # Use the static method directly with the prefix
            run_id = agent.generate_run_id(prefix=prefix)
            
            logger.info(f"Generated run_id: {run_id}")
            
            # Return state updates
            return {
                "run_id": run_id,
                "start_time": datetime.now().isoformat(),
                "status": "initialized"
            }
    except Exception as e:
        error_msg = f"Error in process_uid: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Return error updates that match the expected schema (using "step" instead of "task")
        return {
            "errors": [{
                "step": "generate_uid",  # Use "step" instead of "task"
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_brand_context(state: BrandNameGenerationState, agent: BrandContextExpert) -> Dict[str, Any]:
    try:
        with tracing_v2_enabled():
            # Call the agent to extract brand context and capture the result
            brand_context = await agent.extract_brand_context(
                user_prompt=state.user_prompt,
                run_id=state.run_id
            )
            
            # IMPORTANT: These field names must exactly match the output_keys defined in tasks.yaml
            # for the Understand_Brand_Context task to ensure proper data flow through the workflow
            return {
                "run_id": brand_context.get("run_id", state.run_id),
                "brand_promise": brand_context.get("brand_promise", ""),
                "brand_values": brand_context.get("brand_values", []),
                "brand_personality": brand_context.get("brand_personality", []),
                "brand_tone_of_voice": brand_context.get("brand_tone_of_voice", ""),
                "brand_purpose": brand_context.get("brand_purpose", ""),
                "brand_mission": brand_context.get("brand_mission", ""),
                "target_audience": brand_context.get("target_audience", ""),
                "customer_needs": brand_context.get("customer_needs", []),
                "market_positioning": brand_context.get("market_positioning", ""),
                "competitive_landscape": brand_context.get("competitive_landscape", ""),
                "industry_focus": brand_context.get("industry_focus", ""),
                "industry_trends": brand_context.get("industry_trends", []),
                "brand_identity_brief": brand_context.get("brand_identity_brief", {})
            }
    except Exception as e:
        logger.error(f"Error in process_brand_context: {str(e)}")
        # Return error information as a dictionary update
        return {
            "errors": [{
                "step": "understand_brand_context",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_brand_names(state: BrandNameGenerationState, agent: BrandNameCreationExpert) -> Dict[str, Any]:
    try:
        with tracing_v2_enabled():
            # Create brand_context dictionary with consistent use of getattr() for all fields
            brand_context = {
                "brand_promise": getattr(state, "brand_promise", ""),
                "brand_personality": getattr(state, "brand_personality", []),
                "brand_tone_of_voice": getattr(state, "brand_tone_of_voice", ""),
                "target_audience": getattr(state, "target_audience", ""),
                "customer_needs": getattr(state, "customer_needs", []),
                "market_positioning": getattr(state, "market_positioning", ""),
                "competitive_landscape": getattr(state, "competitive_landscape", ""),
                "industry_focus": getattr(state, "industry_focus", ""),
                "industry_trends": getattr(state, "industry_trends", []),
                "brand_identity_brief": getattr(state, "brand_identity_brief", {})
            }
            
            # Extract purpose from brand_purpose field or fall back to brand_promise
            purpose = getattr(state, "brand_purpose", getattr(state, "brand_promise", ""))
            
            # Extract key_attributes from brand_personality
            key_attributes = []
            brand_personality = getattr(state, "brand_personality", [])
            if isinstance(brand_personality, list):
                key_attributes = brand_personality
            elif isinstance(brand_personality, str):
                key_attributes = [attr.strip() for attr in brand_personality.split(',') if attr.strip()]
            else:
                # Default fallback
                key_attributes = ["Innovative", "Professional", "Trustworthy"]
            
            # Get brand values as a list
            brand_values = []
            brand_values_attr = getattr(state, "brand_values", [])
            if isinstance(brand_values_attr, list):
                brand_values = brand_values_attr
            elif isinstance(brand_values_attr, str):
                brand_values = [val.strip() for val in brand_values_attr.split(',') if val.strip()]
            
            # Call generate_brand_names with exactly the parameters it expects
            brand_names = await agent.generate_brand_names(
                run_id=state.run_id,
                brand_context=brand_context,
                brand_values=brand_values,
                purpose=purpose,
                key_attributes=key_attributes
            )
            
            # IMPORTANT: These field names must exactly match the output_keys defined in tasks.yaml
            # for the Generate_Brand_Name_Ideas task to ensure proper data flow
            if brand_names:
                # Store the list of generated names
                return {
                    "generated_names": brand_names,
                    # The following fields will be derived from the first brand name for state tracking
                    # These match the output_keys in tasks.yaml
                    "brand_name": brand_names[0].get("brand_name", ""),
                    "naming_category": brand_names[0].get("naming_category", ""),
                    "brand_personality_alignment": brand_names[0].get("brand_personality_alignment", ""),
                    "brand_promise_alignment": brand_names[0].get("brand_promise_alignment", ""),
                    "target_audience_relevance_score": brand_names[0].get("target_audience_relevance_score", 5.0),
                    "target_audience_relevance_details": brand_names[0].get("target_audience_relevance_details", ""),
                    "market_differentiation_score": brand_names[0].get("market_differentiation_score", 5.0),
                    "market_differentiation_details": brand_names[0].get("market_differentiation_details", ""),
                    "memorability_score": brand_names[0].get("memorability_score", 5.0),
                    "memorability_score_details": brand_names[0].get("memorability_score_details", ""),
                    "pronounceability_score": brand_names[0].get("pronounceability_score", 5.0),
                    "pronounceability_score_details": brand_names[0].get("pronounceability_score_details", ""),
                    "visual_branding_potential_score": brand_names[0].get("visual_branding_potential_score", 5.0),
                    "visual_branding_potential_details": brand_names[0].get("visual_branding_potential_details", ""),
                    "name_generation_methodology": brand_names[0].get("name_generation_methodology", ""),
                    "timestamp": brand_names[0].get("timestamp", ""),
                    "rank": brand_names[0].get("rank", 5.0)
                }
            else:
                # Return empty state if no names were generated
                return {"generated_names": []}
    except Exception as e:
        logger.error(f"Error in process_brand_names: {str(e)}")
        return {
            "errors": [{
                "step": "generate_brand_names",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_linguistics(state: BrandNameGenerationState, agent: LinguisticsExpert) -> Dict[str, Any]:
    try:
        with tracing_v2_enabled():
            linguistics_analysis = await agent.analyze_names(
                brand_names=[name["brand_name"] for name in state.generated_names],
                run_id=state.run_id
            )
            
            # Create a deep copy of generated names to update
            generated_names = [dict(name) for name in state.generated_names]
            
            # Update with linguistics analysis
            for i, name_data in enumerate(generated_names):
                if i < len(linguistics_analysis):
                    name_data["linguistic_analysis"] = linguistics_analysis[i]
            
            # IMPORTANT: Match the output_keys defined in tasks.yaml
            # Create a flattened version of the first analysis for state tracking
            if linguistics_analysis and len(linguistics_analysis) > 0:
                first_analysis = linguistics_analysis[0]
                return {
                    "generated_names": generated_names,
                    "run_id": state.run_id,
                    "brand_name": first_analysis.get("brand_name", ""),
                    "pronunciation_ease": first_analysis.get("pronunciation_ease", ""),
                    "euphony_vs_cacophony": first_analysis.get("euphony_vs_cacophony", ""),
                    "rhythm_and_meter": first_analysis.get("rhythm_and_meter", ""),
                    "phoneme_frequency_distribution": first_analysis.get("phoneme_frequency_distribution", ""),
                    "sound_symbolism": first_analysis.get("sound_symbolism", ""),
                    "word_class": first_analysis.get("word_class", ""),
                    "morphological_transparency": first_analysis.get("morphological_transparency", ""),
                    "grammatical_gender": first_analysis.get("grammatical_gender", ""),
                    "inflectional_properties": first_analysis.get("inflectional_properties", ""),
                    "ease_of_marketing_integration": first_analysis.get("ease_of_marketing_integration", ""),
                    "naturalness_in_collocations": first_analysis.get("naturalness_in_collocations", ""),
                    "homophones_homographs": first_analysis.get("homophones_homographs", False),
                    "semantic_distance_from_competitors": first_analysis.get("semantic_distance_from_competitors", ""),
                    "neologism_appropriateness": first_analysis.get("neologism_appropriateness", ""),
                    "overall_readability_score": first_analysis.get("overall_readability_score", ""),
                    "notes": first_analysis.get("notes", ""),
                    "rank": first_analysis.get("rank", 0)
                }
            else:
                return {
                    "generated_names": generated_names
                }
    except Exception as e:
        logger.error(f"Error in process_linguistics: {str(e)}")
        return {
            "errors": [{
                "step": "analyze_linguistics",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_cultural_sensitivity(state: BrandNameGenerationState, agent: CulturalSensitivityExpert) -> Dict[str, Any]:
    try:
        with tracing_v2_enabled():
            cultural_analysis = await agent.analyze_names(
                brand_names=[name["brand_name"] for name in state.generated_names],
                run_id=state.run_id
            )
            
            # Create a deep copy of generated names to update
            generated_names = [dict(name) for name in state.generated_names]
            
            # Update with cultural sensitivity analysis
            for i, name_data in enumerate(generated_names):
                if i < len(cultural_analysis):
                    name_data["cultural_analysis"] = cultural_analysis[i]
            
            # IMPORTANT: Match the output_keys defined in tasks.yaml
            # Create a flattened version of the first analysis for state tracking
            if cultural_analysis and len(cultural_analysis) > 0:
                first_analysis = cultural_analysis[0]
                return {
                    "generated_names": generated_names,
                    "run_id": state.run_id,
                    "brand_name": first_analysis.get("brand_name", ""),
                    "cultural_connotations": first_analysis.get("cultural_connotations", ""),
                    "symbolic_meanings": first_analysis.get("symbolic_meanings", ""),
                    "alignment_with_cultural_values": first_analysis.get("alignment_with_cultural_values", ""),
                    "religious_sensitivities": first_analysis.get("religious_sensitivities", ""),
                    "social_political_taboos": first_analysis.get("social_political_taboos", ""),
                    "body_part_bodily_function_connotations": first_analysis.get("body_part_bodily_function_connotations", False),
                    "age_related_connotations": first_analysis.get("age_related_connotations", ""),
                    "gender_connotations": first_analysis.get("gender_connotations", ""),
                    "regional_variations": first_analysis.get("regional_variations", ""),
                    "historical_meaning": first_analysis.get("historical_meaning", ""),
                    "current_event_relevance": first_analysis.get("current_event_relevance", ""),
                    "overall_risk_rating": first_analysis.get("overall_risk_rating", ""),
                    "notes": first_analysis.get("notes", ""),
                    "rank": first_analysis.get("rank", 0)
                }
            else:
                return {
                    "generated_names": generated_names
                }
    except Exception as e:
        logger.error(f"Error in process_cultural_sensitivity: {str(e)}")
        return {
            "errors": [{
                "step": "analyze_cultural_sensitivity",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_evaluation(state: BrandNameGenerationState, agent: BrandNameEvaluator) -> Dict[str, Any]:
    """
    Evaluate brand names based on analyses and select top candidates.
    
    Args:
        state: The current workflow state
        agent: The BrandNameEvaluator agent
        
    Returns:
        Dictionary with state updates including evaluation results and shortlisted names
    """
    try:
        with tracing_v2_enabled():
            # Get input data
            brand_names = state.generated_names
            
            # Extract relevant analyses from the state
            semantic_analyses = [name.get("semantic_analysis") for name in brand_names]
            linguistic_analyses = [name.get("linguistic_analysis") for name in brand_names]
            cultural_analyses = [name.get("cultural_analysis") for name in brand_names]
            
            # Evaluate brand names
            evaluation_results = await agent.evaluate_brand_names(
                brand_names=[name["brand_name"] for name in brand_names],
                semantic_analyses=semantic_analyses,
                linguistic_analyses=linguistic_analyses, 
                cultural_analyses=cultural_analyses,
                run_id=state.run_id
            )
            
            # Shortlist top brand names
            shortlisted_names = [result for result in evaluation_results if result.get("shortlist_status") is True]
            
            logger.info(f"Evaluated {len(brand_names)} brand names; shortlisted {len(shortlisted_names)}")
            
            # IMPORTANT: Match the output_keys defined in tasks.yaml
            # Create a flattened version of the first evaluation for state tracking
            if evaluation_results and len(evaluation_results) > 0:
                first_result = evaluation_results[0]
                return {
                    "evaluation_results": evaluation_results,
                    "shortlisted_names": shortlisted_names,
                    "run_id": state.run_id,
                    "brand_name": first_result.get("brand_name", ""),
                    "strategic_alignment_score": first_result.get("strategic_alignment_score", 0),
                    "distinctiveness_score": first_result.get("distinctiveness_score", 0),
                    "competitive_advantage": first_result.get("competitive_advantage", ""),
                    "brand_fit_score": first_result.get("brand_fit_score", 0),
                    "positioning_strength": first_result.get("positioning_strength", ""),
                    "memorability_score": first_result.get("memorability_score", 0),
                    "pronounceability_score": first_result.get("pronounceability_score", 0),
                    "meaningfulness_score": first_result.get("meaningfulness_score", 0),
                    "phonetic_harmony": first_result.get("phonetic_harmony", ""),
                    "visual_branding_potential": first_result.get("visual_branding_potential", ""),
                    "storytelling_potential": first_result.get("storytelling_potential", ""),
                    "domain_viability_score": first_result.get("domain_viability_score", 0),
                    "overall_score": first_result.get("overall_score", 0),
                    "shortlist_status": first_result.get("shortlist_status", False),
                    "evaluation_comments": first_result.get("evaluation_comments", ""),
                    "rank": first_result.get("rank", 0)
                }
            else:
                return {
                    "evaluation_results": [],
                    "shortlisted_names": []
                }
    except Exception as e:
        logger.error(f"Error in process_evaluation: {str(e)}")
        return {
            "errors": [{
                "step": "evaluate_brand_names",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_analyses(state: BrandNameGenerationState, analyzers: List[Any]) -> Dict[str, Any]:
    """Run multiple analysis agents concurrently using LangGraph map."""
    try:
        with tracing_v2_enabled():
            # Prepare input data for map
            input_data = [
                {
                    "run_id": state.run_id,
                    "brand_name": name["brand_name"],
                    "brand_context": getattr(state, "brand_context", {})
                }
                for name in state.generated_names
            ]
            
            # Run analyses concurrently using client.map
            # Use the client from dependencies if available
            if hasattr(state, "client") and state.client:
                # Use the client from state for map operation with callbacks
                client = state.client
            elif hasattr(state, "deps") and state.deps and state.deps.get("langsmith"):
                # Use the client from dependencies
                client = state.deps["langsmith"]
                logger.info("Using LangSmith client from dependencies")
            else:
                # If no client is provided, create one for this operation
                from langsmith import Client
                logger.info("Creating new LangGraph client for map operation")
                client = Client()
            
            # Use the client for map operation
            results = await client.map(
                analyzers,
                input_data,
                lambda agent, data: agent.analyze_brand_name(**data),
                config={"callbacks": [client]}
            )
            
            # Return dictionary of state updates
            return {
                "analysis_results": {
                    "results": results
                }
            }
            
    except Exception as e:
        logger.error(f"Error in process_analyses: {str(e)}")
        return {
            "errors": [{
                "step": "process_analyses",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_market_research(state: BrandNameGenerationState, agent: MarketResearchExpert) -> Dict[str, Any]:
    """Process market research analysis with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            market_research = await agent.analyze_market_potential(
                run_id=state.run_id,
                brand_names=[name["brand_name"] for name in state.generated_names],
                brand_context=getattr(state, "brand_context", {})
            )
            
            # IMPORTANT: Match the output_keys defined in tasks.yaml for market research
            # If we have results, extract the first one for state tracking
            if market_research and len(market_research) > 0:
                first_result = market_research[0]
                return {
                    "market_research_results": market_research,
                    "run_id": state.run_id,
                    "brand_name": first_result.get("brand_name", ""),
                    "market_size": first_result.get("market_size", ""),
                    "growth_potential": first_result.get("growth_potential", ""),
                    "competitor_analysis": first_result.get("competitor_analysis", ""),
                    "target_market_fit": first_result.get("target_market_fit", ""),
                    "market_readiness": first_result.get("market_readiness", "")
                }
            else:
                return {
                    "market_research_results": []
                }
            
    except Exception as e:
        logger.error(f"Error in process_market_research: {str(e)}")
        return {
            "errors": [{
                "step": "conduct_market_research",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_report(state: BrandNameGenerationState, agent: ReportCompiler) -> Dict[str, Any]:
    """Process report compilation with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            report = await agent.compile_report(
                run_id=state.run_id,
                state=state
            )
            
            # IMPORTANT: Match the output_keys defined in tasks.yaml
            current_time = datetime.now().isoformat()
            return {
                "compiled_report": report,
                "run_id": state.run_id,
                "report_url": report.get("report_url", ""),
                "version": report.get("version", 1),
                "created_at": current_time,
                "last_updated": current_time,
                "format": report.get("format", "pdf"),
                "file_size_kb": report.get("file_size_kb", 0),
                "notes": report.get("notes", "")
            }
            
    except Exception as e:
        logger.error(f"Error in process_report: {str(e)}")
        return {
            "errors": [{
                "step": "compile_report",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_report_storage(state: BrandNameGenerationState, agent: ReportStorer) -> Dict[str, Any]:
    """Process report storage with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            storage_result = await agent.store_report(
                run_id=state.run_id,
                report_data=state.compiled_report
            )
            
            # IMPORTANT: Match the output_keys defined in tasks.yaml
            current_time = datetime.now().isoformat()
            return {
                "report_storage_metadata": storage_result,
                "run_id": state.run_id,
                "report_url": storage_result.get("report_url", ""),
                "status": storage_result.get("status", "completed"),
                "created_at": current_time,
                "last_updated": current_time
            }
            
    except Exception as e:
        logger.error(f"Error in process_report_storage: {str(e)}")
        return {
            "errors": [{
                "step": "store_report",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

def create_langsmith_tracer() -> Optional[LangChainTracer]:
    """Create LangSmith tracer if enabled."""
    if settings.langsmith_enabled:
        return LangChainTracer(
            project_name=settings.langsmith_project,
            tags={
                "application": "mae_brand_namer",
                "version": settings.version,
                "environment": settings.environment
            }
        )
    return None 

# Public API function
async def run_brand_naming_workflow(
    user_prompt: str,
    client: Optional[Any] = None,
    dependencies: Optional[Dependencies] = None
) -> Dict[str, Any]:
    """
    Run the brand naming workflow with the given prompt.
    
    This is the main public API function for the brand naming workflow.
    
    Args:
        user_prompt (str): The user's prompt describing the brand naming requirements
        client (Optional[Any]): LangGraph client for async operations and tracing
        dependencies (Optional[Dependencies]): Optional container for application dependencies
        
    Returns:
        Dict[str, Any]: The final state of the workflow containing all results
    """
    # Setup event loop if not available
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    # If client wasn't provided, create one
    use_client = client
    if use_client is None:
        from langsmith import Client
        use_client = Client()
    
    # Get LangSmith client if available in dependencies
    langsmith_client = None
    if dependencies and hasattr(dependencies, "langsmith"):
        langsmith_client = dependencies.langsmith
    
    # Create the workflow with configuration
    workflow_config = {
        "configurable": {
            "langsmith_client": langsmith_client,
            "default_step_delay": 2.0,
            "step_delays": None  # Use default step delays
        }
    }
    workflow = create_workflow(workflow_config)
    
    # Create process supervisor callback handler with the LangSmith client
    supervisor_handler = ProcessSupervisorCallbackHandler(langsmith_client=langsmith_client)
    
    try:
        # Invoke the workflow with client and supervisor callback for monitoring
        result = await workflow.ainvoke(
            {"user_prompt": user_prompt, "client": use_client},
            config={"callbacks": [use_client, supervisor_handler]}
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Error running brand naming workflow",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "user_prompt": user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt
            }
        )
        raise

# Example usage
async def main():
    """
    Example function showing how to invoke the brand naming workflow asynchronously.
    
    This demonstrates the proper pattern for using the workflow with full async support.
    """
    import asyncio
    from langsmith import Client
    
    try:
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Create the LangGraph client for async operations
        client = Client()
        
        # Create the workflow
        workflow = create_workflow(langsmith_client=client)
        
        # Create process supervisor callback handler with LangSmith client
        supervisor_handler = ProcessSupervisorCallbackHandler(langsmith_client=client)
        
        print("Starting brand naming workflow...")
        
        # Invoke the workflow with the client and supervisor handler
        result = await workflow.ainvoke(
            {
                "user_prompt": "global b2b consultancy specializing in digital transformation", 
                "client": client
            }, 
            config={"callbacks": [client, supervisor_handler]}
        )
        
        print("Workflow execution completed successfully.")
        print(f"Generated {len(result.get('shortlisted_names', []))} shortlisted brand names")
        print(f"Run ID: {result.get('run_id')}")
        
        if result.get("report_url"):
            print(f"Report available at: {result.get('report_url')}")
            
        return result
        
    except Exception as e:
        print(f"Error executing workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import asyncio
    
    # Set up and run the async event loop
    asyncio.run(main()) 