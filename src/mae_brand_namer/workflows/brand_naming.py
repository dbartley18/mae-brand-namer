"""Brand naming workflow using LangGraph."""

from typing import Dict, Any, List, Optional, TypedDict, Tuple, Union, Callable, TypeVar
from datetime import datetime
import asyncio
from functools import partial
import json
import os
import uuid
import traceback

from langgraph.graph import StateGraph, Graph
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

from mae_brand_namer.utils.logging import get_logger
from mae_brand_namer.agents import (
    UIDGeneratorAgent,
    BrandContextExpert,
    BrandNameCreationExpert,
    LinguisticsExpert,
    SemanticAnalysisExpert,
    CulturalSensitivityExpert,
    DomainAnalysisExpert,
    SEOOnlineDiscoveryExpert,
    CompetitorAnalysisExpert,
    SurveySimulationExpert,
    MarketResearchExpert,
    ReportCompiler,
    
    ProcessSupervisor,
    BrandNameEvaluator
)
from mae_brand_namer.utils.supabase_utils import SupabaseManager
from mae_brand_namer.config.settings import settings
from mae_brand_namer.config.dependencies import Dependencies
from mae_brand_namer.agents.language_expert_factory import get_language_expert, get_language_display_name
from mae_brand_namer.models.state import BrandNameGenerationState

# Initialize logger
logger = get_logger(__name__)

# Mapping of node names to agent types and task names for process monitoring
NODE_AGENT_TASK_MAPPING = {
    "generate_uid": ("UIDGenerator", "Generate_UID"),
    "understand_brand_context": ("BrandContextExpert", "Understand_Brand_Context"),
    "generate_brand_names": ("BrandNameCreationExpert", "Generate_Brand_Names"),
    "process_semantic_analysis": ("SemanticAnalysisExpert", "Analyze_Semantics"),
    "process_linguistic_analysis": ("LinguisticsExpert", "Analyze_Linguistics"),
    "process_cultural_analysis": ("CulturalSensitivityExpert", "Analyze_Cultural_Sensitivity"),
    "process_translation_analysis": ("LanguageTranslationExperts", "Analyze_Translations"),
    "process_evaluation": ("BrandNameEvaluator", "Evaluate_Names"),
    "process_market_research": ("MarketResearchExpert", "Research_Market"),
    "process_domain_analysis": ("DomainAnalysisExpert", "Analyze_Domain"),
    "process_seo_analysis": ("SEOOnlineDiscoveryExpert", "Analyze_SEO"),
    "process_competitor_analysis": ("CompetitorAnalysisExpert", "Analyze_Competition"),
    "process_survey_simulation": ("SurveySimulationExpert", "Simulate_Survey"),
    "compile_report": ("ReportCompiler", "Compile_Report"),
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
        self.node_to_agent_task = NODE_AGENT_TASK_MAPPING
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
            
        # Only generate a fallback run_id for the generate_uid node
        # Extract the current node name
        node_name = self._extract_node_name(kwargs.get("serialized", {}), **kwargs)
        
        # If we still don't have a run_id and this is specifically the generate_uid node
        if not run_id and node_name == "generate_uid":
            # Generate a fallback run_id
            run_id = f"generated_{uuid.uuid4().hex[:8]}"
            self.current_run_id = run_id
            logger.warning(f"Generated fallback run_id: {run_id} for generate_uid node")
            
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
            
            # Only use the current_run_id, don't extract from outputs to avoid LangGraph conflicts
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

def create_workflow(config: dict) -> StateGraph:
    """
    Create the brand naming workflow state graph with nodes, edges, and process monitoring.
    
    Args:
        config: A dictionary or RunnableConfig containing configuration parameters
            - langsmith_client: Optional LangSmith client for tracing
            - default_step_delay: Default delay between steps in seconds (0.0 for no delay)
            - step_delays: Optional dict mapping step names to specific delay times
        
    Returns:
        StateGraph: The workflow graph
    """
    # Extract config parameters from the config object
    configurable = getattr(config, "configurable", config) if config else {}
    
    # Extract parameters with defaults
    langsmith_client = configurable.get("langsmith_client", None)
    default_step_delay = configurable.get("default_step_delay", 0.0)
    step_delays = configurable.get("step_delays") or {
            "understand_brand_context": 2.0,
            "generate_brand_names": 3.0,
            "process_analyses": 5.0,  # This is a heavy step with multiple experts
            "process_evaluation": 3.0,
            "process_market_research": 3.0,
        "compile_report": 3.0
    }
    
    # Create SupabaseManager for database management
    supabase_manager = SupabaseManager()
    
    # Create process supervisor for monitoring
    process_supervisor = ProcessSupervisor()
    
    # Define wrapper for async functions to ensure proper execution order and delay
    def wrap_async_node(fn, node_name):
        # Get the task_name from mapping
        agent_type, task_name = NODE_AGENT_TASK_MAPPING.get(node_name, (None, None))
        
        async def wrapped_fn(state, config=None):
            # Get run_id and validate it's not unknown
            run_id = getattr(state, "run_id", "unknown")
            if run_id == "unknown" and node_name != "generate_uid":
                logger.warning(
                    f"Missing run_id in node {node_name}. This could cause tracking issues. "
                    f"Ensure process_uid executes first and generates a valid run_id."
                )
            
            # Log task/node start
            if process_supervisor and agent_type and task_name:
                await process_supervisor.log_task_start(
                    run_id=run_id,
                    agent_type=agent_type,
                    task_name=task_name
                )
            
            # Apply delay if configured
            delay = step_delays.get(node_name, default_step_delay)
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Call the original function
            try:
                updates = await fn(state)
                
                # Apply updates to state
                for k, v in updates.items():
                    setattr(state, k, v)
                
                # Check if run_id has been updated by this function
                updated_run_id = getattr(state, "run_id", "unknown")
                if run_id != "unknown" and updated_run_id != run_id and node_name != "generate_uid":
                    logger.warning(
                        f"run_id changed during execution of {node_name} from {run_id} to {updated_run_id}. "
                        f"This could cause consistency issues."
                    )
                
                # Log task/node completion
                if process_supervisor and agent_type and task_name:
                    await process_supervisor.log_task_completion(
                        run_id=updated_run_id if node_name == "generate_uid" else run_id,
                        agent_type=agent_type,
                        task_name=task_name
                    )
                    
                return state
            except Exception as e:
                logger.error(f"Error in {node_name}: {str(e)}")
                # Log task/node error
                if process_supervisor and agent_type and task_name:
                    await process_supervisor.log_task_error(
                        run_id=run_id,
                        agent_type=agent_type,
                        task_name=task_name,
                        error=e
                    )
                # Add error to state errors list
                state.errors.append({
                    "step": node_name,
                    "error": str(e)
                })
                # Reraise the exception for graph error handling
                raise
        
        return wrapped_fn
    
    # Create the workflow
    workflow = StateGraph(BrandNameGenerationState)
    
    # Define nodes for each step in the workflow
    workflow.add_node("generate_uid", wrap_async_node(process_uid, "generate_uid"))
    workflow.add_node("understand_brand_context",
        wrap_async_node(lambda state: process_brand_context(state, BrandContextExpert()), "understand_brand_context")
    )
    workflow.add_node("generate_brand_names",
        wrap_async_node(lambda state: process_brand_names(state, BrandNameCreationExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "generate_brand_names")
    )
    
    # Add individual analysis nodes for ungrouped analysis
    workflow.add_node("process_semantic_analysis",
        wrap_async_node(lambda state: process_semantic_analysis(state, SemanticAnalysisExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_semantic_analysis")
    )
    
    workflow.add_node("process_linguistic_analysis", 
        wrap_async_node(lambda state: process_linguistic_analysis(state, LinguisticsExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_linguistic_analysis")
    )
    
    workflow.add_node("process_cultural_analysis", 
        wrap_async_node(lambda state: process_cultural_analysis(state, CulturalSensitivityExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_cultural_analysis")
    )
    
    workflow.add_node("process_translation_analysis", 
        wrap_async_node(lambda state: process_multi_language_translation(
            state, 
            language_codes=["es", "fr", "de", "zh", "ja", "ar"],
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        ), "process_translation_analysis")
    )
    
    workflow.add_node("process_evaluation", 
        wrap_async_node(lambda state: process_evaluation(state, BrandNameEvaluator(
            supabase=supabase_manager,
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_evaluation")
    )
    
    workflow.add_node("process_market_research", 
        wrap_async_node(lambda state: process_market_research(state, MarketResearchExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_market_research")
    )
    
    workflow.add_node("process_domain_analysis", 
        wrap_async_node(lambda state: process_domain_analysis(state, DomainAnalysisExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_domain_analysis")
    )
    
    workflow.add_node("process_seo_analysis", 
        wrap_async_node(lambda state: process_seo_analysis(state, SEOOnlineDiscoveryExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_seo_analysis")
    )
    
    workflow.add_node("process_competitor_analysis", 
        wrap_async_node(lambda state: process_competitor_analysis(state, CompetitorAnalysisExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_competitor_analysis")
    )
    
    workflow.add_node("process_survey_simulation", 
        wrap_async_node(lambda state: process_survey_simulation(state, SurveySimulationExpert(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "process_survey_simulation")
    )
    
    workflow.add_node("compile_report", 
        wrap_async_node(lambda state: process_report(state, ReportCompiler(
            dependencies=Dependencies(supabase=supabase_manager, langsmith=langsmith_client)
        )), "compile_report")
    )
    
    # Add edges to connect nodes in the workflow
    workflow.add_edge("generate_uid", "understand_brand_context")
    workflow.add_edge("understand_brand_context", "generate_brand_names")
    
    # Change from parallel to sequential execution for analysis steps
    workflow.add_edge("generate_brand_names", "process_semantic_analysis")
    workflow.add_edge("process_semantic_analysis", "process_linguistic_analysis")
    workflow.add_edge("process_linguistic_analysis", "process_cultural_analysis")
    workflow.add_edge("process_cultural_analysis", "process_evaluation")
    
    # Move translation after evaluation so it only processes shortlisted names
    workflow.add_edge("process_evaluation", "process_translation_analysis")
    workflow.add_edge("process_translation_analysis", "process_market_research")
    
    workflow.add_edge("process_market_research", "process_domain_analysis")
    workflow.add_edge("process_domain_analysis", "process_seo_analysis")
    workflow.add_edge("process_seo_analysis", "process_competitor_analysis")
    workflow.add_edge("process_competitor_analysis", "process_survey_simulation")
    workflow.add_edge("process_survey_simulation", "compile_report")
    
    # Define entry point
    workflow.set_entry_point("generate_uid")
    
    return workflow

async def process_uid(state: BrandNameGenerationState) -> Dict[str, Any]:
    """
    Generate a unique run ID for the workflow execution.
    This is the ONLY function that should set the run_id field.
    
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
        
        # Check if run_id is already set
        existing_run_id = getattr(state, "run_id", None)
        if existing_run_id:
            logger.warning(
                f"process_uid found existing run_id: {existing_run_id}. "
                f"This should not happen as process_uid is the only function that should set run_id."
            )
        
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
            try:
                brand_names = await agent.generate_brand_names(
                    run_id=state.run_id,
                    brand_context=brand_context,
                    brand_values=brand_values,
                    purpose=purpose,
                    key_attributes=key_attributes
                )
            except Exception as e:
                logger.error(f"Error generating brand names: {str(e)}")
                # Create fallback brand names to allow workflow to continue
                brand_names = [
                    {
                        "brand_name": f"BrandSuggestion{i+1}",
                        "naming_category": "Fallback",
                        "brand_personality_alignment": "Generated after error",
                        "brand_promise_alignment": "Error fallback",
                        "target_audience_relevance": 5.0,
                        "market_differentiation": 5.0,
                        "visual_branding_potential": 5.0,
                        "memorability_score": 5.0,
                        "pronounceability_score": 5.0,
                        "target_audience_relevance_details": "Error fallback data",
                        "market_differentiation_details": "Error fallback data",
                        "visual_branding_potential_details": "Error fallback data",
                        "memorability_score_details": "Error fallback data",
                        "pronounceability_score_details": "Error fallback data",
                        "name_generation_methodology": "Fallback after error",
                        "timestamp": datetime.now().isoformat(),
                        "rank": i + 1
                    } for i in range(3)  # Generate 3 fallback names
                ]
                
                # Add warning to state
                if "warnings" not in state or not state.warnings:
                    state.warnings = []
                state.warnings.append({
                    "step": "generate_brand_names",
                    "warning": f"Using fallback brand names due to error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
            
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
                    
                    # Split fields - scores
                    "target_audience_relevance": brand_names[0].get("target_audience_relevance", 0),
                    "market_differentiation": brand_names[0].get("market_differentiation", 0),
                    "visual_branding_potential": brand_names[0].get("visual_branding_potential", 0),
                    "memorability_score": brand_names[0].get("memorability_score", 0),
                    "pronounceability_score": brand_names[0].get("pronounceability_score", 0),
                    
                    # Split fields - details
                    "target_audience_relevance_details": brand_names[0].get("target_audience_relevance_details", ""),
                    "market_differentiation_details": brand_names[0].get("market_differentiation_details", ""),
                    "visual_branding_potential_details": brand_names[0].get("visual_branding_potential_details", ""),
                    "memorability_score_details": brand_names[0].get("memorability_score_details", ""),
                    "pronounceability_score_details": brand_names[0].get("pronounceability_score_details", ""),
                    
                    # Other fields
                    "name_generation_methodology": brand_names[0].get("name_generation_methodology", ""),
                    "timestamp": brand_names[0].get("timestamp", ""),
                    "rank": brand_names[0].get("rank", 0)
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

async def process_semantic_analysis(state: BrandNameGenerationState, agent: SemanticAnalysisExpert) -> Dict[str, Any]:
    """Run semantic analysis on each brand name."""
    try:
        with tracing_v2_enabled():
            # Prepare results container
            semantic_results = []
            
            # Log the start of analysis
            num_names = len(state.generated_names) if state.generated_names else 0
            logger.info(f"Starting semantic analysis for {num_names} brand names")
            
            # Process each brand name in sequence
            for brand_name_data in state.generated_names:
                try:
                    # Skip empty names
                    if not brand_name_data.get("brand_name"):
                        logger.warning("Skipping empty brand name in semantic analysis")
                        continue
                        
                    # Run analysis
                    result = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name_data["brand_name"]
                    )
                    
                    # Fix boolean fields to ensure they're proper booleans
                    boolean_fields = ["ambiguity", "irony_or_paradox", "humor_playfulness", 
                                     "rhyme_rhythm", "alliteration_assonance"]
                    
                    for field in boolean_fields:
                        if field in result:
                            if isinstance(result[field], str):
                                # Convert string to boolean based on content
                                # If it starts with "No" or "Not", it's False, otherwise True
                                text = result[field].lower()
                                result[field] = not (
                                    text.startswith("no") or 
                                    text.startswith("not") or 
                                    "no " in text[:15] or
                                    "none" in text[:15] or
                                    "not " in text[:15]
                                )
                            elif result[field] is None:
                                # Default to False if None
                                result[field] = False
                    
                    semantic_results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing brand name {brand_name_data.get('brand_name', '[unknown]')}: {str(e)}")
                    # Create a fallback result for this name
                    semantic_results.append({
                        "brand_name": brand_name_data.get("brand_name", "[unknown]"),
                        "task_name": "semantic_analysis",
                        "denotative_meaning": f"Error analyzing '{brand_name_data.get('brand_name', '[unknown]')}'",
                        "etymology": "Unknown due to analysis error",
                        "descriptiveness": 5,
                        "brand_personality": "Error during analysis",
                        "memorability_score": 5,
                        "pronunciation_ease": 5,
                        "brand_fit_relevance": 5,
                        "error": str(e)
                    })
            
            # Return dictionary of state updates
            return {
                "semantic_analysis_results": semantic_results
            }
            
    except Exception as e:
        logger.error(f"Error in process_semantic_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "process_semantic_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_linguistic_analysis(state: BrandNameGenerationState, agent: LinguisticsExpert) -> Dict[str, Any]:
    """Run linguistic analysis on each brand name."""
    try:
        with tracing_v2_enabled():
            # Prepare results container - change from list to dictionary
            linguistic_results = {}
            
            # Log the start of analysis
            num_names = len(state.generated_names) if state.generated_names else 0
            logger.info(f"Starting linguistic analysis for {num_names} brand names")
            
            # Process each brand name in sequence
            for brand_name_data in state.generated_names:
                try:
                    # Skip empty names
                    if not brand_name_data.get("brand_name"):
                        logger.warning("Skipping empty brand name in linguistic analysis")
                        continue
                    
                    brand_name = brand_name_data["brand_name"]
                        
                    # Run analysis 
                    result = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name
                    )
                    
                    # Fix the homophones_homographs field to ensure it's a boolean
                    if "homophones_homographs" in result:
                        if isinstance(result["homophones_homographs"], str):
                            # Convert string to boolean based on content
                            # If it starts with "No" or "Not", it's False, otherwise True
                            text = result["homophones_homographs"].lower()
                            result["homophones_homographs"] = not (
                                text.startswith("no") or 
                                text.startswith("not") or 
                                "no " in text[:15] or
                                "none" in text[:15] or
                                "not " in text[:15]
                            )
                        elif result["homophones_homographs"] is None:
                            # Default to False if None
                            result["homophones_homographs"] = False
                    
                    # Store result with brand name as key
                    linguistic_results[brand_name] = result
                except Exception as e:
                    logger.error(f"Error analyzing brand name {brand_name_data.get('brand_name', '[unknown]')}: {str(e)}")
                    # Store error result with brand name as key
                    linguistic_results[brand_name_data.get("brand_name", "[unknown]")] = {
                        "brand_name": brand_name_data.get("brand_name", "[unknown]"),
                        "error": str(e)
                    }
            
            # Return dictionary of state updates
            return {
                "linguistic_analysis_results": linguistic_results
            }
            
    except Exception as e:
        logger.error(f"Error in process_linguistic_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "process_linguistic_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_cultural_analysis(state: BrandNameGenerationState, agent: CulturalSensitivityExpert) -> Dict[str, Any]:
    """Run cultural sensitivity analysis on each brand name."""
    try:
        with tracing_v2_enabled():
            # Prepare results container - change from list to dictionary
            cultural_results = {}
            
            # Log the start of analysis
            num_names = len(state.generated_names) if state.generated_names else 0
            logger.info(f"Starting cultural sensitivity analysis for {num_names} brand names")
            
            # Process each brand name in sequence
            for brand_name_data in state.generated_names:
                try:
                    # Skip empty names
                    if not brand_name_data.get("brand_name"):
                        logger.warning("Skipping empty brand name in cultural analysis")
                        continue
                    
                    brand_name = brand_name_data["brand_name"]
                        
                    # Run analysis 
                    result = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name
                    )
                    
                    # Fix the body_part_bodily_function_connotations field to ensure it's a boolean
                    if "body_part_bodily_function_connotations" in result:
                        if isinstance(result["body_part_bodily_function_connotations"], str):
                            # Convert string to boolean based on content
                            # If it starts with "No" or "Not", it's False, otherwise True
                            text = result["body_part_bodily_function_connotations"].lower()
                            result["body_part_bodily_function_connotations"] = not (
                                text.startswith("no") or 
                                text.startswith("not") or 
                                "no " in text[:15] or
                                "none" in text[:15] or
                                "not " in text[:15]
                            )
                        elif result["body_part_bodily_function_connotations"] is None:
                            # Default to False if None
                            result["body_part_bodily_function_connotations"] = False
                    
                    # Store result with brand name as key
                    cultural_results[brand_name] = result
                except Exception as e:
                    logger.error(f"Error analyzing brand name {brand_name_data.get('brand_name', '[unknown]')}: {str(e)}")
                    # Store error result with brand name as key
                    cultural_results[brand_name_data.get("brand_name", "[unknown]")] = {
                        "brand_name": brand_name_data.get("brand_name", "[unknown]"),
                        "error": str(e)
                    }
            
            # Return dictionary of state updates
            return {
                "cultural_analysis_results": cultural_results
            }
            
    except Exception as e:
        logger.error(f"Error in process_cultural_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "process_cultural_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_multi_language_translation(
    state: BrandNameGenerationState, 
    language_codes: List[str], 
    dependencies: Dependencies
) -> Dict[str, Any]:
    """Run translation analysis for multiple languages on shortlisted brand names.
    
    Args:
        state: Current workflow state
        language_codes: List of language codes to analyze (e.g., ["es", "fr", "de", "zh", "ja", "ar"])
        dependencies: Container for dependencies like Supabase
        
    Returns:
        Dictionary of state updates with translation results
    """
    try:
        # Import the language expert factory functions
        from mae_brand_namer.agents.language_expert_factory import get_language_expert, get_language_display_name
        
        with tracing_v2_enabled():
            # Prepare results container
            all_translation_results = []
            
            # Check if we have shortlisted_names in the state
            if not hasattr(state, "shortlisted_names") or not state.shortlisted_names:
                logger.warning("No shortlisted names found for translation analysis")
                return {
                    "translation_analysis_results": [],
                    "errors": [{
                        "step": "process_translation_analysis",
                        "error": "No shortlisted brand names available for translation",
                        "timestamp": datetime.now().isoformat()
                    }]
                }
            
            # Log the start of analysis
            num_names = len(state.shortlisted_names)
            num_languages = len(language_codes)
            logger.info(f"Starting multi-language translation analysis for {num_names} shortlisted brand names across {num_languages} languages")
            
            # Process each shortlisted brand name in sequence
            for brand_name in state.shortlisted_names:
                # Skip empty names
                if not brand_name:
                    logger.warning("Skipping empty brand name in translation analysis")
                    continue
                
                # Analyze each language for this brand name
                for language_code in language_codes:
                    try:
                        # Get the language expert for this language
                        language_expert = get_language_expert(language_code, dependencies=dependencies)
                        
                        if not language_expert:
                            logger.error(f"Could not create language expert for {language_code}")
                            all_translation_results.append({
                                "brand_name": brand_name,  # REQUIRED - NOT NULL in database
                                "target_language": get_language_display_name(language_code),  # REQUIRED - NOT NULL in database
                                "task_name": "translation_analysis",
                                "direct_translation": f"Error: No language expert available for {language_code}",
                                "semantic_shift": "Error in analysis",
                                "pronunciation_difficulty": "Unknown due to error",
                                "phonetic_retention": "Unknown due to error",
                                "global_consistency_vs_localization": "Unknown due to error",
                                "notes": f"Error: Could not create language expert for {language_code}",
                                "rank": 0.0
                            })
                            continue
                        
                        # Log the analysis start
                        language_name = get_language_display_name(language_code)
                        logger.info(f"Analyzing {language_name} translation for shortlisted brand name: '{brand_name}'")
                        
                        # Run analysis
                        result = await language_expert.analyze_brand_name(
                            run_id=state.run_id,
                            brand_name=brand_name
                        )
                        
                        # Add to results
                        all_translation_results.append(result)
                        
                    except Exception as e:
                        language_name = get_language_display_name(language_code)
                        logger.error(f"Error analyzing {language_name} translation for brand name '{brand_name}': {str(e)}")
                        
                        # Create a fallback result
                        all_translation_results.append({
                            "brand_name": brand_name,  # REQUIRED - NOT NULL in database
                            "target_language": language_name,  # REQUIRED - NOT NULL in database
                            "task_name": "translation_analysis",
                            "direct_translation": "Error in analysis",
                            "semantic_shift": "Error in analysis",
                            "pronunciation_difficulty": "Unknown due to error",
                            "phonetic_similarity_undesirable": False,
                            "phonetic_retention": "Unknown due to error",
                            "cultural_acceptability": "Unknown due to error",
                            "adaptation_needed": False,
                            "proposed_adaptation": "N/A - Error in analysis",
                            "brand_essence_preserved": "Unknown due to error",
                            "global_consistency_vs_localization": "Unknown due to error",
                            "notes": f"Error in analysis: {str(e)}",
                            "rank": 5.0
                        })
            
            # Return dictionary of state updates
            return {
                "translation_analysis_results": all_translation_results
            }
            
    except Exception as e:
        logger.error(f"Error in process_multi_language_translation: {str(e)}")
        return {
            "errors": [{
                "step": "process_translation_analysis",
                "error": str(e)
            }]
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
            # In the sequential flow, we can assume all analyses are complete
            # since this node only runs after all analysis nodes have completed
            
            # Get input data
            brand_names = state.generated_names
            
            # Extract results from all analyses
            semantic_analyses_list = state.semantic_analysis_results if hasattr(state, "semantic_analysis_results") else []
            linguistic_analyses_raw = state.linguistic_analysis_results if hasattr(state, "linguistic_analysis_results") else {}
            cultural_analyses_raw = state.cultural_analysis_results if hasattr(state, "cultural_analysis_results") else {}
            
            # Convert semantic_analyses from list to dictionary
            semantic_analyses = {}
            for analysis in semantic_analyses_list:
                if "brand_name" in analysis:
                    brand_name = analysis["brand_name"]
                    semantic_analyses[brand_name] = {
                        "analysis": analysis
                    }
            
            # Convert linguistic_analyses to the format expected by the evaluator
            linguistic_analyses = {}
            for brand_name, analysis in linguistic_analyses_raw.items():
                # Convert the LinguisticAnalysisResult object to a dictionary
                if hasattr(analysis, "dict"):
                    analysis_dict = analysis.dict()
                else:
                    # If it's already a dict or has another structure, use it as is
                    analysis_dict = analysis
                
                linguistic_analyses[brand_name] = {
                    "analysis": analysis_dict
                }
            
            # Convert cultural_analyses to the format expected by the evaluator
            cultural_analyses = {}
            for brand_name, analysis in cultural_analyses_raw.items():
                # Convert the CulturalAnalysisResult object to a dictionary
                if hasattr(analysis, "dict"):
                    analysis_dict = analysis.dict()
                else:
                    # If it's already a dict or has another structure, use it as is
                    analysis_dict = analysis
                
                cultural_analyses[brand_name] = {
                    "analysis": analysis_dict
                }
            
            # Create brand context from state for the evaluator
            # Even though analyzers don't use brand_context anymore, the evaluator still needs it
            brand_context = {
                "brand_identity_brief": getattr(state, "brand_identity_brief", ""),
                "brand_promise": getattr(state, "brand_promise", ""),
                "brand_values": getattr(state, "brand_values", []),
                "brand_personality": getattr(state, "brand_personality", []),
                "brand_tone_of_voice": getattr(state, "brand_tone_of_voice", ""),
                "brand_purpose": getattr(state, "brand_purpose", ""),
                "brand_mission": getattr(state, "brand_mission", ""),
                "target_audience": getattr(state, "target_audience", ""),
                "customer_needs": getattr(state, "customer_needs", []),
                "market_positioning": getattr(state, "market_positioning", ""),
                "competitive_landscape": getattr(state, "competitive_landscape", ""),
                "industry_focus": getattr(state, "industry_focus", ""),
                "industry_trends": getattr(state, "industry_trends", [])
            }
            
            # Evaluate brand names
            evaluation_results_list = await agent.evaluate_brand_names(
                brand_names=[name["brand_name"] for name in brand_names],
                semantic_analyses=semantic_analyses,
                linguistic_analyses=linguistic_analyses, 
                cultural_analyses=cultural_analyses,
                run_id=state.run_id,
                brand_context=brand_context
            )
            
            # Convert evaluation_results from list to dictionary and ensure all required fields are present
            evaluation_results = {}
            for result in evaluation_results_list:
                if "brand_name" in result:
                    brand_name = result["brand_name"]
                    
                    # Ensure all required fields are present
                    required_fields = {
                        "positioning_strength": "Not evaluated",
                        "phonetic_harmony": "Not evaluated",
                        "visual_branding_potential": "Not evaluated",
                        "storytelling_potential": "Not evaluated",
                        "rank": 0.0
                    }
                    
                    for field, default_value in required_fields.items():
                        if field not in result or result[field] is None:
                            result[field] = default_value
                    
                    # Ensure all score fields are integers between 1-10
                    integer_score_fields = [
                        "strategic_alignment_score", "distinctiveness_score", "brand_fit_score",
                        "memorability_score", "pronounceability_score", "meaningfulness_score",
                        "domain_viability_score", "overall_score"
                    ]
                    
                    for field in integer_score_fields:
                        if field in result and result[field] is not None:
                            try:
                                # Convert to integer
                                value = int(float(result[field]))
                                # Constrain to range 1-10
                                result[field] = max(1, min(10, value))
                            except (ValueError, TypeError):
                                # Default to middle value if conversion fails
                                result[field] = 5
                        else:
                            # Set default value if missing
                            result[field] = 5
                    
                    # Ensure shortlist_status is a boolean for database storage
                    # but convert to string for the state model
                    if "shortlist_status" in result:
                        if isinstance(result["shortlist_status"], str):
                            is_shortlisted = result["shortlist_status"].lower() in ["true", "yes", "1", "t", "y"]
                            # Store the boolean value for later use
                            result["_shortlist_status_bool"] = is_shortlisted
                            # Convert to string for state model
                            result["shortlist_status"] = "Yes" if is_shortlisted else "No"
                        elif isinstance(result["shortlist_status"], bool):
                            # Store the boolean value for later use
                            result["_shortlist_status_bool"] = result["shortlist_status"]
                            # Convert to string for state model
                            result["shortlist_status"] = "Yes" if result["shortlist_status"] else "No"
                        else:
                            # Default to False if not a recognized type
                            result["_shortlist_status_bool"] = False
                            result["shortlist_status"] = "No"
                    else:
                        result["_shortlist_status_bool"] = False
                        result["shortlist_status"] = "No"
                    
                    evaluation_results[brand_name] = result
            
            # Shortlist top brand names (using the stored boolean value)
            shortlisted_names = [
                result["brand_name"] for result in evaluation_results_list 
                if result.get("_shortlist_status_bool", False) or 
                   (isinstance(result.get("shortlist_status"), bool) and result["shortlist_status"]) or
                   (isinstance(result.get("shortlist_status"), str) and 
                    result["shortlist_status"].lower() in ["true", "yes", "1", "t", "y"])
            ]
            
            logger.info(f"Evaluated {len(brand_names)} brand names; shortlisted {len(shortlisted_names)}")
            
            # IMPORTANT: Match the output_keys defined in tasks.yaml
            # Create a flattened version of the first evaluation for state tracking
            if evaluation_results_list and len(evaluation_results_list) > 0:
                first_result = evaluation_results_list[0]
                
                # Ensure all required fields are present in the first result
                required_fields = {
                    "strategic_alignment_score": 0,
                    "distinctiveness_score": 0,
                    "competitive_advantage": "",
                    "brand_fit_score": 0,
                    "positioning_strength": "Not evaluated",
                    "memorability_score": 0,
                    "pronounceability_score": 0,
                    "meaningfulness_score": 0,
                    "phonetic_harmony": "Not evaluated",
                    "visual_branding_potential": "Not evaluated",
                    "storytelling_potential": "Not evaluated",
                    "domain_viability_score": 0,
                    "overall_score": 0,
                    "shortlist_status": "No",
                    "evaluation_comments": "",
                    "rank": 0.0
                }
                
                for field, default_value in required_fields.items():
                    if field not in first_result or first_result[field] is None:
                        first_result[field] = default_value
                
                # Convert shortlist_status to string for the state
                if "shortlist_status" in first_result and isinstance(first_result["shortlist_status"], bool):
                    first_result["shortlist_status"] = "Yes" if first_result["shortlist_status"] else "No"
                
                return {
                    "evaluation_results": evaluation_results,
                    "shortlisted_names": shortlisted_names,
                    "brand_name": first_result.get("brand_name", ""),
                    "strategic_alignment_score": first_result.get("strategic_alignment_score", 0),
                    "distinctiveness_score": first_result.get("distinctiveness_score", 0),
                    "competitive_advantage": first_result.get("competitive_advantage", ""),
                    "brand_fit_score": first_result.get("brand_fit_score", 0),
                    "positioning_strength": first_result.get("positioning_strength", "Not evaluated"),
                    "memorability_score": first_result.get("memorability_score", 0),
                    "pronounceability_score": first_result.get("pronounceability_score", 0),
                    "meaningfulness_score": first_result.get("meaningfulness_score", 0),
                    "phonetic_harmony": first_result.get("phonetic_harmony", "Not evaluated"),
                    "visual_branding_potential": first_result.get("visual_branding_potential", "Not evaluated"),
                    "storytelling_potential": first_result.get("storytelling_potential", "Not evaluated"),
                    "domain_viability_score": first_result.get("domain_viability_score", 0),
                    "overall_score": first_result.get("overall_score", 0),
                    "shortlist_status": first_result.get("shortlist_status", "No"),
                    "evaluation_comments": first_result.get("evaluation_comments", ""),
                    "rank": first_result.get("rank", 0.0)
                }
            else:
                # Return empty results if no evaluations were generated
                return {
                    "evaluation_results": {},
                    "shortlisted_names": [],
                    "brand_name": "",
                    "strategic_alignment_score": 0,
                    "distinctiveness_score": 0,
                    "competitive_advantage": "",
                    "brand_fit_score": 0,
                    "positioning_strength": "Not evaluated",
                    "memorability_score": 0,
                    "pronounceability_score": 0,
                    "meaningfulness_score": 0,
                    "phonetic_harmony": "Not evaluated",
                    "visual_branding_potential": "Not evaluated",
                    "storytelling_potential": "Not evaluated",
                    "domain_viability_score": 0,
                    "overall_score": 0,
                    "shortlist_status": "No",
                    "evaluation_comments": "",
                    "rank": 0.0
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

async def process_market_research(state: BrandNameGenerationState, agent: MarketResearchExpert) -> Dict[str, Any]:
    """Process market research analysis with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            # Extract only the shortlisted names from evaluation results
            shortlisted_names = []
            
            # Check if we have evaluation results and shortlisted_names in the state
            if hasattr(state, "shortlisted_names") and state.shortlisted_names:
                # Use the explicitly shortlisted names if available
                shortlisted_names = state.shortlisted_names
            elif hasattr(state, "evaluation_results") and state.evaluation_results:
                # Extract shortlisted names from evaluation results
                for brand_name, eval_data in state.evaluation_results.items():
                    if eval_data.get("shortlist_status", False):
                        shortlisted_names.append(brand_name)
            
            # If no shortlisted names found through normal means, use a fallback approach
            if not shortlisted_names and hasattr(state, "generated_names") and state.generated_names:
                logger.warning("No shortlisted names found, using top 3 names from evaluation results if available")
                
                # Try to get top 3 based on overall_score if evaluations exist
                if hasattr(state, "evaluation_results") and state.evaluation_results:
                    sorted_names = sorted(
                        state.evaluation_results.items(),
                        key=lambda x: x[1].get("overall_score", 0),
                        reverse=True
                    )
                    shortlisted_names = [name for name, _ in sorted_names[:3]]
                else:
                    # Last resort: just take the first 3 generated names
                    logger.warning("No evaluation results found, using first 3 generated names")
                    shortlisted_names = [name["brand_name"] for name in state.generated_names[:3]]
            
            logger.info(f"Conducting market research on {len(shortlisted_names)} shortlisted names: {shortlisted_names}")
            
            # Only proceed if we have shortlisted names to analyze
            if shortlisted_names:
                # Extract brand context for the market research
                brand_context = {}
                if hasattr(state, "brand_identity_brief"):
                    brand_context["brand_identity_brief"] = state.brand_identity_brief
                if hasattr(state, "brand_values"):
                    brand_context["brand_values"] = state.brand_values
                if hasattr(state, "brand_personality"):
                    brand_context["brand_personality"] = state.brand_personality
                if hasattr(state, "target_audience"):
                    brand_context["target_audience"] = state.target_audience
                if hasattr(state, "industry_focus"):
                    brand_context["industry_focus"] = state.industry_focus
                
                market_research = await agent.analyze_market_potential(
                    run_id=state.run_id,
                    brand_names=shortlisted_names,
                    brand_context=brand_context
                )
                
                # Ensure market_research is a list
                if not isinstance(market_research, list):
                    logger.warning(f"Expected market_research to be a list, got {type(market_research)}. Converting to empty list.")
                    market_research = []
                
                # Create a new dictionary with only the fields defined in the state model
                results = {}
                results["run_id"] = state.run_id
                
                # Explicitly handle the market_research_results field to avoid field access error
                results["market_research_results"] = market_research
                
                return results
            else:
                logger.error("No brand names available for market research analysis")
                
                # Create a new dictionary with only the fields defined in the state model
                results = {}
                
                # Ensure the run_id is passed through
                if hasattr(state, "run_id"):
                    results["run_id"] = state.run_id
                    
                # Initialize empty market research results
                results["market_research_results"] = []
                
                # Add error information
                results["errors"] = [{
                    "step": "conduct_market_research",
                    "error": "No shortlisted brand names available for analysis",
                    "timestamp": datetime.now().isoformat()
                }]
                results["status"] = "error"
                
                return results
            
    except Exception as e:
        logger.error(f"Error in process_market_research: {str(e)}")
        
        # Create a new dictionary with only the fields defined in the state model
        results = {}
        
        # Ensure the run_id is passed through
        if hasattr(state, "run_id"):
            results["run_id"] = state.run_id
            
        # Initialize empty market research results
        results["market_research_results"] = []
        
        # Add error information
        results["errors"] = [{
            "step": "conduct_market_research",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }]
        results["status"] = "error"
        
        return results

async def process_domain_analysis(state: BrandNameGenerationState, agent: DomainAnalysisExpert) -> Dict[str, Any]:
    """Process domain analysis with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            # Extract only the shortlisted names from the state
            shortlisted_names = []
            
            # Check if we have shortlisted_names in the state
            if hasattr(state, "shortlisted_names") and state.shortlisted_names:
                shortlisted_names = state.shortlisted_names
            else:
                logger.warning("No shortlisted names found for domain analysis")
                return {
                    "domain_analysis_results": [],
                    "run_id": state.run_id,
                    "errors": [{
                        "step": "process_domain_analysis",
                        "error": "No shortlisted brand names available for analysis",
                        "timestamp": datetime.now().isoformat()
                    }],
                    "status": "error"
                }
            
            # Extract brand context for domain analysis
            brand_context = {}
            if hasattr(state, "brand_identity_brief"):
                brand_context["brand_identity_brief"] = state.brand_identity_brief
            if hasattr(state, "brand_values"):
                brand_context["brand_values"] = state.brand_values
            if hasattr(state, "brand_personality"):
                brand_context["brand_personality"] = state.brand_personality
            if hasattr(state, "target_audience"):
                brand_context["target_audience"] = state.target_audience
            if hasattr(state, "industry_focus"):
                brand_context["industry_focus"] = state.industry_focus
            
            # Process each shortlisted name for domain analysis
            domain_analyses = []
            for brand_name in shortlisted_names:
                logger.info(f"Analyzing domain for shortlisted name: {brand_name}")
                try:
                    # Try to analyze domain but handle any exceptions
                    analysis = await agent.analyze_domain(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context
                    )
                    domain_analyses.append({
                        "brand_name": brand_name,
                        **analysis
                    })
                except Exception as e:
                    logger.error(f"Error analyzing domain for {brand_name}: {str(e)}")
                    # Continue with next name instead of breaking the entire process
                    domain_analyses.append({
                        "brand_name": brand_name,
                        "error": str(e),
                        "status": "error"
                    })
            
            # Create a new dictionary with only the fields defined in the state model
            results = {}
            results["run_id"] = state.run_id
            results["domain_analysis_results"] = domain_analyses
            
            return results
            
    except Exception as e:
        logger.error(f"Error in process_domain_analysis: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback: {traceback_str}")
        return {
            "domain_analysis_results": [],
            "run_id": state.run_id if hasattr(state, "run_id") else "unknown",
            "errors": [{
                "step": "process_domain_analysis",
                "error": str(e),
                "traceback": traceback_str,
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_seo_analysis(state: BrandNameGenerationState, agent: SEOOnlineDiscoveryExpert) -> Dict[str, Any]:
    """Process SEO and online discovery analysis with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            # Extract only the shortlisted names from the state
            shortlisted_names = []
            
            # Check if we have shortlisted_names in the state
            if hasattr(state, "shortlisted_names") and state.shortlisted_names:
                shortlisted_names = state.shortlisted_names
            else:
                logger.warning("No shortlisted names found for SEO analysis")
                return {
                    "seo_analysis_results": [],
                    "run_id": state.run_id,
                    "errors": [{
                        "step": "process_seo_analysis",
                        "error": "No shortlisted brand names available for analysis",
                        "timestamp": datetime.now().isoformat()
                    }],
                    "status": "error"
                }
            
            # Extract brand context for SEO analysis
            brand_context = {}
            if hasattr(state, "brand_identity_brief"):
                brand_context["brand_identity_brief"] = state.brand_identity_brief
            if hasattr(state, "brand_values"):
                brand_context["brand_values"] = state.brand_values
            if hasattr(state, "brand_personality"):
                brand_context["brand_personality"] = state.brand_personality
            if hasattr(state, "target_audience"):
                brand_context["target_audience"] = state.target_audience
            if hasattr(state, "industry_focus"):
                brand_context["industry_focus"] = state.industry_focus
            
            # Process each shortlisted name for SEO analysis
            seo_analyses = []
            for brand_name in shortlisted_names:
                logger.info(f"Analyzing SEO for shortlisted name: {brand_name}")
                try:
                    analysis = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context
                    )
                    seo_analyses.append({
                        "brand_name": brand_name,
                        **analysis
                    })
                except Exception as e:
                    logger.error(f"Error analyzing SEO for {brand_name}: {str(e)}")
                    seo_analyses.append({
                        "brand_name": brand_name,
                        "error": str(e),
                        "status": "error"
                    })
            
            # Create a new dictionary with only the fields defined in the state model
            results = {}
            results["run_id"] = state.run_id
            results["seo_analysis_results"] = seo_analyses
            
            return results
            
    except Exception as e:
        logger.error(f"Error in process_seo_analysis: {str(e)}")
        return {
            "seo_analysis_results": [],
            "run_id": state.run_id if hasattr(state, "run_id") else "unknown",
            "errors": [{
                "step": "process_seo_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_competitor_analysis(state: BrandNameGenerationState, agent: CompetitorAnalysisExpert) -> Dict[str, Any]:
    """Process competitor analysis with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            # Extract only the shortlisted names from the state
            shortlisted_names = []
            
            # Check if we have shortlisted_names in the state
            if hasattr(state, "shortlisted_names") and state.shortlisted_names:
                shortlisted_names = state.shortlisted_names
            else:
                logger.warning("No shortlisted names found for competitor analysis")
                return {
                    "competitor_analysis_results": [],
                    "run_id": state.run_id,
                    "errors": [{
                        "step": "process_competitor_analysis",
                        "error": "No shortlisted brand names available for analysis",
                        "timestamp": datetime.now().isoformat()
                    }],
                    "status": "error"
                }
            
            # Extract brand context for competitor analysis
            brand_context = {}
            if hasattr(state, "brand_identity_brief"):
                brand_context["brand_identity_brief"] = state.brand_identity_brief
            if hasattr(state, "brand_values"):
                brand_context["brand_values"] = state.brand_values
            if hasattr(state, "brand_personality"):
                brand_context["brand_personality"] = state.brand_personality
            if hasattr(state, "target_audience"):
                brand_context["target_audience"] = state.target_audience
            if hasattr(state, "industry_focus"):
                brand_context["industry_focus"] = state.industry_focus
            if hasattr(state, "competitors"):
                brand_context["competitors"] = state.competitors
            
            # Process each shortlisted name for competitor analysis
            competitor_analyses = []
            for brand_name in shortlisted_names:
                logger.info(f"Analyzing competitive position for shortlisted name: {brand_name}")
                try:
                    analysis = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context,
                        user_prompt=state.user_prompt
                    )
                    # Check if the analysis contains a competitors array
                    if "competitors" in analysis and isinstance(analysis["competitors"], list):
                        # Handle multiple competitors returned
                        competitor_analyses.append({
                            "brand_name": brand_name,
                            "competitors": analysis["competitors"],
                            "competitor_count": analysis.get("competitor_count", len(analysis["competitors"]))
                        })
                    else:
                        # Handle single competitor or error case
                        competitor_analyses.append({
                            "brand_name": brand_name,
                            **analysis
                        })
                except Exception as e:
                    logger.error(f"Error analyzing competitive position for {brand_name}: {str(e)}")
                    competitor_analyses.append({
                        "brand_name": brand_name,
                        "error": str(e),
                        "status": "error"
                    })
            
            # Create a new dictionary with only the fields defined in the state model
            results = {}
            results["run_id"] = state.run_id
            results["competitor_analysis_results"] = competitor_analyses
            
            return results
            
    except Exception as e:
        logger.error(f"Error in process_competitor_analysis: {str(e)}")
        return {
            "competitor_analysis_results": [],
            "run_id": state.run_id if hasattr(state, "run_id") else "unknown",
            "errors": [{
                "step": "process_competitor_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_survey_simulation(state: BrandNameGenerationState, agent: SurveySimulationExpert) -> Dict[str, Any]:
    """Process survey simulation with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            # Extract only the shortlisted names from the state
            shortlisted_names = []
            
            # Check if we have shortlisted_names in the state
            if hasattr(state, "shortlisted_names") and state.shortlisted_names:
                shortlisted_names = state.shortlisted_names
            else:
                logger.warning("No shortlisted names found for survey simulation")
                return {
                    "survey_simulation_results": [],
                    "run_id": state.run_id,
                    "errors": [{
                        "step": "process_survey_simulation",
                        "error": "No shortlisted brand names available for simulation",
                        "timestamp": datetime.now().isoformat()
                    }],
                    "status": "error"
                }
            
            # Extract brand context for survey simulation
            brand_context = {}
            if hasattr(state, "brand_identity_brief"):
                brand_context["brand_identity_brief"] = state.brand_identity_brief
            if hasattr(state, "brand_values"):
                brand_context["brand_values"] = state.brand_values
            if hasattr(state, "brand_personality"):
                brand_context["brand_personality"] = state.brand_personality
            if hasattr(state, "target_audience"):
                brand_context["target_audience"] = state.target_audience
            if hasattr(state, "industry_focus"):
                brand_context["industry_focus"] = state.industry_focus
            
            # Process each shortlisted name for survey simulation
            survey_simulations = []
            for brand_name in shortlisted_names:
                logger.info(f"Simulating survey for shortlisted name: {brand_name}")
                try:
                    # Extract target audience as a list
                    target_audience = []
                    if hasattr(state, "target_audience") and state.target_audience:
                        if isinstance(state.target_audience, list):
                            target_audience = state.target_audience
                        elif isinstance(state.target_audience, str):
                            target_audience = [segment.strip() for segment in state.target_audience.split(',')]
                    
                    # Extract brand values as a list
                    brand_values = []
                    if hasattr(state, "brand_values") and state.brand_values:
                        if isinstance(state.brand_values, list):
                            brand_values = state.brand_values
                        elif isinstance(state.brand_values, str):
                            brand_values = [value.strip() for value in state.brand_values.split(',')]
                    
                    # Get competitive analysis if available
                    competitive_analysis = {}
                    if hasattr(state, "competitor_analysis_results") and state.competitor_analysis_results:
                        competitive_analysis = state.competitor_analysis_results
                    
                    # Call the simulate_survey method with all available parameters
                    simulation = await agent.simulate_survey(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context,
                        target_audience=target_audience,
                        brand_values=brand_values,
                        competitive_analysis=competitive_analysis
                    )
                    
                    # Add the simulation results to the list
                    survey_simulations.append({
                        "brand_name": brand_name,
                        **simulation
                    })
                    
                    logger.info(f"Successfully simulated survey for {brand_name} with {len(simulation.get('individual_personas', []))} individual personas")
                    
                except Exception as e:
                    logger.error(f"Error simulating survey for {brand_name}: {str(e)}")
                    survey_simulations.append({
                        "brand_name": brand_name,
                        "error": str(e),
                        "status": "error"
                    })
            
            # Create a new dictionary with only the fields defined in the state model
            results = {}
            results["run_id"] = state.run_id
            results["survey_simulation_results"] = survey_simulations
            
            return results
            
    except Exception as e:
        logger.error(f"Error in process_survey_simulation: {str(e)}")
        return {
            "errors": [{
                "step": "process_survey_simulation",
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

            
    except Exception as e:
        logger.error(f"Error in process_report_storage: {str(e)}")
        return {
            "errors": [{
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
        # Initialize state with just the user_prompt
        # The run_id will be generated by the process_uid function
        initial_state = {"user_prompt": user_prompt, "client": use_client}
        
        # Invoke the workflow with client and supervisor callback for monitoring
        result = await workflow.ainvoke(
            initial_state,
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
        
        # Create the workflow with proper config object
        workflow_config = {
            "configurable": {
                "langsmith_client": client,
                "default_step_delay": 2.0,
                "step_delays": None  # Use default step delays
            }
        }
        workflow = create_workflow(workflow_config)
        
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

# Special entry point for LangGraph API
def graph_factory(config=None):
    """
    Entry point function for LangGraph API.
    
    This function can optionally take a config argument as required by the LangGraph API.
    
    Args:
        config: Optional configuration object containing configuration parameters
    
    Returns:
        StateGraph: The configured workflow graph
    """
    # Use provided config or create a default one
    workflow_config = config or {
        "configurable": {
            "langsmith_client": None,
            "default_step_delay": 2.0,
            "step_delays": None  # Use default step delays
        }
    }
    return create_workflow(workflow_config) 