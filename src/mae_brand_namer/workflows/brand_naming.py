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
from mae_brand_namer.models.app_config import AppConfig

logger = get_logger(__name__)

# Map workflow node names to agent and task names
node_to_agent_task = {
    "process_uid": ("UIDGeneratorAgent", "Generate_UID"),
    "process_brand_context": ("BrandContextExpert", "Understand_Brand_Context"),
    "process_brand_names": ("BrandNameCreationExpert", "Generate_Brand_Name_Ideas"),
    "process_semantic_analysis": ("SemanticAnalysisExpert", "Conduct_Semantic_Analysis"),
    "process_linguistic_analysis": ("LinguisticsExpert", "Conduct_Linguistic_Analysis"),
    "process_cultural_analysis": ("CulturalSensitivityExpert", "Conduct_Cultural_Sensitivity_Analysis"),
    "process_translation_analysis": ("TranslationAnalysisExpert", "Conduct_Translation_Analysis"),
    "process_analyses": ("AnalysisCoordinator", "Conduct_All_Analyses"),
    "process_evaluation": ("BrandNameEvaluator", "Evaluate_Brand_Name_Ideas"),
    "process_domain_analysis": ("DomainAnalysisExpert", "Conduct_Domain_Search"),
    "process_seo_analysis": ("SEOOnlineDiscoveryExpert", "Analyze_SEO_and_Online_Discoverability"),
    "process_competitor_analysis": ("CompetitorAnalysisExpert", "Analyze_Competitor_Names"),
    "process_survey_simulation": ("SurveySimulationExpert", "Conduct_Survey_Simulation"),
    "process_market_research": ("MarketResearchExpert", "Analyze_Market_Research"),
    "compile_report": ("ReportCompiler", "Compile_And_Format_Report"),
    "store_report": ("ReportStorer", "Store_Report"),
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
    Create the brand naming workflow.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        A StateGraph instance representing the workflow
    """
    app_config = AppConfig.get_instance()
    
    # Create agents
    uid_generator_agent = UIDGeneratorAgent()
    brand_context_expert = BrandContextExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    brand_name_creation_expert = BrandNameCreationExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    semantic_expert = SemanticAnalysisExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    linguistic_expert = LinguisticsExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    cultural_sensitivity_expert = CulturalSensitivityExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    translation_expert = TranslationAnalysisExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    domain_analysis_expert = DomainAnalysisExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    seo_online_discovery_expert = SEOOnlineDiscoveryExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    competitor_analysis_expert = CompetitorAnalysisExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    survey_simulation_expert = SurveySimulationExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    market_research_expert = MarketResearchExpert(dependencies=Dependencies(supabase=SupabaseManager()))
    report_compiler = ReportCompiler(dependencies=Dependencies(supabase=SupabaseManager()))
    report_storer = ReportStorer(dependencies=Dependencies(supabase=SupabaseManager()))
    brand_name_evaluator = BrandNameEvaluator(dependencies=Dependencies(supabase=SupabaseManager()))
    
    # Get step delays from config or use defaults
    step_delays = config.get("step_delays", None)
    if step_delays is None:
        step_delays = {
            # Fast steps
            "process_uid": 0.1,
            "process_brand_context": 1.0,
            # Analysis steps
            "process_semantic_analysis": 2.0,
            "process_linguistic_analysis": 2.0,
            "process_cultural_analysis": 2.0,
            "process_translation_analysis": 2.0,
            "process_domain_analysis": 2.0,
            "process_seo_analysis": 2.0,
            "process_competitor_analysis": 2.0,
            "process_survey_simulation": 2.0,
            # Legacy combined analysis step
            "process_analyses": 5.0,
            # Medium steps
            "process_brand_names": 3.0,
            "process_evaluation": 3.0,
            "process_market_research": 3.0,
            # Slow steps
            "compile_report": 5.0,
            "store_report": 1.0,
        }
    
    # Create workflow state graph
    workflow = StateGraph(BrandNameGenerationState)
    
    # Helper function to wrap async functions so they are properly handled by LangGraph
    def wrap_async_node(func, step_name: str = "unknown"):
        """Wraps an async function to ensure it's properly awaited before returning"""
        async def wrapper(state):
            # Get the appropriate delay for this step (or use default)
            step_delay = step_delays.get(step_name, 2.0)
            # Add throttling delay before executing the step
            await throttle_step(step_delay, step_name)
            # Execute the original function
            result = await func(state)
            return result
        return wrapper
    
    # Add nodes for each step in the workflow
    workflow.add_node("process_uid", wrap_async_node(process_uid, "process_uid"))
    workflow.add_node("process_brand_context", wrap_async_node(
        lambda state: process_brand_context(state, brand_context_expert),
        "process_brand_context"
    ))
    workflow.add_node("process_brand_names", wrap_async_node(
        lambda state: process_brand_names(state, brand_name_creation_expert),
        "process_brand_names"
    ))
    
    # Add separate nodes for each analysis type
    workflow.add_node("process_semantic_analysis", wrap_async_node(
        lambda state: process_semantic_analysis(state, semantic_expert),
        "process_semantic_analysis"
    ))
    workflow.add_node("process_linguistic_analysis", wrap_async_node(
        lambda state: process_linguistic_analysis(state, linguistic_expert),
        "process_linguistic_analysis"
    ))
    workflow.add_node("process_cultural_analysis", wrap_async_node(
        lambda state: process_cultural_analysis(state, cultural_sensitivity_expert),
        "process_cultural_analysis"
    ))
    workflow.add_node("process_translation_analysis", wrap_async_node(
        lambda state: process_translation_analysis(state, translation_expert),
        "process_translation_analysis"
    ))
    
    # Keep the legacy combined node for backward compatibility
    workflow.add_node("process_analyses", wrap_async_node(
        lambda state: process_analyses(state, [
            semantic_expert,
            linguistic_expert, 
            cultural_sensitivity_expert,
            translation_expert
        ]),
        "process_analyses"
    ))
    
    workflow.add_node("process_evaluation", wrap_async_node(
        lambda state: process_evaluation(state, brand_name_evaluator),
        "process_evaluation"
    ))
    workflow.add_node("process_domain_analysis", wrap_async_node(
        lambda state: process_domain_analysis(state, domain_analysis_expert),
        "process_domain_analysis"
    ))
    workflow.add_node("process_seo_analysis", wrap_async_node(
        lambda state: process_seo_analysis(state, seo_online_discovery_expert),
        "process_seo_analysis"
    ))
    workflow.add_node("process_competitor_analysis", wrap_async_node(
        lambda state: process_competitor_analysis(state, competitor_analysis_expert),
        "process_competitor_analysis"
    ))
    workflow.add_node("process_survey_simulation", wrap_async_node(
        lambda state: process_survey_simulation(state, survey_simulation_expert),
        "process_survey_simulation"
    ))
    workflow.add_node("process_market_research", wrap_async_node(
        lambda state: process_market_research(state, market_research_expert),
        "process_market_research"
    ))
    workflow.add_node("compile_report", wrap_async_node(
        lambda state: process_report(state, report_compiler),
        "compile_report"
    ))
    workflow.add_node("store_report", wrap_async_node(
        lambda state: process_report_storage(state, report_storer),
        "store_report"
    ))
    
    # Set the entry point
    workflow.set_entry_point("process_uid")
    
    # Define edges
    workflow.add_edge("process_uid", "process_brand_context")
    workflow.add_edge("process_brand_context", "process_brand_names")
    
    # Connect brand name generation to individual analysis agents
    workflow.add_edge("process_brand_names", "process_semantic_analysis")
    workflow.add_edge("process_brand_names", "process_linguistic_analysis")
    workflow.add_edge("process_brand_names", "process_cultural_analysis")  
    workflow.add_edge("process_brand_names", "process_translation_analysis")
    
    # Connect all analysis agents to evaluation
    workflow.add_edge("process_semantic_analysis", "process_evaluation")
    workflow.add_edge("process_linguistic_analysis", "process_evaluation")
    workflow.add_edge("process_cultural_analysis", "process_evaluation")
    workflow.add_edge("process_translation_analysis", "process_evaluation")
    
    # Connect the evaluation to market research steps
    workflow.add_edge("process_evaluation", "process_domain_analysis")
    workflow.add_edge("process_evaluation", "process_seo_analysis")
    workflow.add_edge("process_evaluation", "process_competitor_analysis")
    workflow.add_edge("process_evaluation", "process_survey_simulation")
    
    # Connect market research to report compilation
    workflow.add_edge("process_domain_analysis", "process_market_research")
    workflow.add_edge("process_seo_analysis", "process_market_research")
    workflow.add_edge("process_competitor_analysis", "process_market_research")
    workflow.add_edge("process_survey_simulation", "process_market_research")
    
    # Connect report compilation to storage
    workflow.add_edge("process_market_research", "compile_report")
    workflow.add_edge("compile_report", "store_report")
    
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
                # Safely convert values to float, defaulting to 0.0 if conversion fails
                try:
                    target_audience_relevance = float(brand_names[0].get("target_audience_relevance", 0))
                except (ValueError, TypeError):
                    target_audience_relevance = 0.0
                    
                try:
                    market_differentiation = float(brand_names[0].get("market_differentiation", 0))
                except (ValueError, TypeError):
                    market_differentiation = 0.0
                    
                try:
                    visual_branding_potential = float(brand_names[0].get("visual_branding_potential", 0))
                except (ValueError, TypeError):
                    visual_branding_potential = 0.0
                    
                return {
                    "generated_names": brand_names,
                    # The following fields will be derived from the first brand name for state tracking
                    # These match the output_keys in tasks.yaml
                    "brand_name": brand_names[0].get("brand_name", ""),
                    "naming_category": brand_names[0].get("naming_category", ""),
                    "brand_personality_alignment": brand_names[0].get("brand_personality_alignment", ""),
                    "brand_promise_alignment": brand_names[0].get("brand_promise_alignment", ""),
                    # Convert floats to strings and wrap in lists to match BrandNameGenerationState model
                    "target_audience_relevance": [str(target_audience_relevance)],
                    "market_differentiation": [str(market_differentiation)],
                    "memorability_score": brand_names[0].get("memorability_score", 0),
                    "pronounceability_score": brand_names[0].get("pronounceability_score", 0),
                    "visual_branding_potential": [str(visual_branding_potential)],
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
    """
    Processes semantic analysis for each brand name.
    
    Args:
        state: The current workflow state
        agent: The semantic analysis expert agent
        
    Returns:
        Dict with semantic_analysis_output containing the analysis results
    """
    try:
        logger.info(f"Running semantic analysis...")
        
        # Extract required input from state
        run_id = state.run_id
        brand_names = [name["brand_name"] for name in state.generated_names]
        brand_context = getattr(state, "brand_context", {})
        brand_values = getattr(state, "brand_values", [])
        
        # Run semantic analysis
        semantic_analyses = []
        for brand_name in brand_names:
            try:
                # Call the agent to perform semantic analysis
                analysis = await agent.analyze_brand_name(
                    run_id=run_id,
                    brand_name=brand_name,
                    brand_context=brand_context,
                    brand_values=brand_values
                )
                
                # Ensure all required fields are present and have appropriate data types
                # Required fields (is_nullable=NO)
                analysis["run_id"] = run_id
                analysis["task_name"] = "Conduct_Semantic_Analysis"
                analysis["brand_name"] = brand_name
                
                # Optional fields with appropriate data types
                analysis.setdefault("denotative_meaning", "")
                analysis.setdefault("etymology", "")
                analysis.setdefault("descriptiveness", 0.0)  # numeric
                analysis.setdefault("concreteness", 0.0)  # numeric
                analysis.setdefault("emotional_valence", "")
                analysis.setdefault("brand_personality", "")
                analysis.setdefault("sensory_associations", "")
                analysis.setdefault("figurative_language", "")
                analysis.setdefault("ambiguity", False)  # boolean
                analysis.setdefault("irony_or_paradox", False)  # boolean
                analysis.setdefault("humor_playfulness", False)  # boolean
                analysis.setdefault("phoneme_combinations", "")
                analysis.setdefault("sound_symbolism", "")
                analysis.setdefault("rhyme_rhythm", False)  # boolean
                analysis.setdefault("alliteration_assonance", False)  # boolean
                analysis.setdefault("word_length_syllables", 0)  # integer
                analysis.setdefault("compounding_derivation", "")
                analysis.setdefault("brand_name_type", "")
                analysis.setdefault("memorability_score", 0.0)  # numeric
                analysis.setdefault("pronunciation_ease", 0.0)  # numeric
                analysis.setdefault("clarity_understandability", 0.0)  # numeric
                analysis.setdefault("uniqueness_differentiation", 0.0)  # numeric
                analysis.setdefault("brand_fit_relevance", 0.0)  # numeric
                analysis.setdefault("semantic_trademark_risk", "")
                analysis.setdefault("rank", 0)  # numeric
                
                semantic_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error in semantic analysis for {brand_name}: {str(e)}")
                # Create a minimal analysis entry with required fields for failed analyses
                semantic_analyses.append({
                    "run_id": run_id,
                    "task_name": "Conduct_Semantic_Analysis",
                    "brand_name": brand_name,
                    "denotative_meaning": f"Error: {str(e)}"
                })
        
        return {
            "semantic_analysis_output": semantic_analyses
        }
        
    except Exception as e:
        logger.error(f"Error in process_semantic_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "semantic_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }],
            "semantic_analysis_output": []
        }

async def process_linguistic_analysis(state: BrandNameGenerationState, agent: LinguisticsExpert) -> Dict[str, Any]:
    """
    Processes linguistic analysis for each brand name.
    
    Args:
        state: The current workflow state
        agent: The linguistics expert agent
        
    Returns:
        Dict with linguistic_analysis_output containing the analysis results
    """
    try:
        logger.info(f"Running linguistic analysis...")
        
        # Extract required input from state
        run_id = state.run_id
        brand_names = [name["brand_name"] for name in state.generated_names]
        brand_context = getattr(state, "brand_context", {})
        brand_values = getattr(state, "brand_values", [])
        
        # Run linguistic analysis
        linguistic_analyses = []
        for brand_name in brand_names:
            try:
                # Call the agent to perform linguistic analysis
                analysis = await agent.analyze_brand_name(
                    run_id=run_id,
                    brand_name=brand_name,
                    brand_context=brand_context,
                    brand_values=brand_values
                )
                
                # Ensure all required fields are present and have appropriate data types
                # Required fields (is_nullable=NO)
                analysis["run_id"] = run_id
                analysis["brand_name"] = brand_name
                
                # Optional fields with appropriate data types
                analysis.setdefault("pronunciation_ease", "")
                analysis.setdefault("euphony_vs_cacophony", "")
                analysis.setdefault("rhythm_and_meter", "")
                analysis.setdefault("phoneme_frequency_distribution", "")
                analysis.setdefault("sound_symbolism", "")
                analysis.setdefault("word_class", "")
                analysis.setdefault("morphological_transparency", "")
                analysis.setdefault("grammatical_gender", "")
                analysis.setdefault("inflectional_properties", "")
                analysis.setdefault("ease_of_marketing_integration", "")
                analysis.setdefault("naturalness_in_collocations", "")
                analysis.setdefault("homophones_homographs", False)  # boolean
                analysis.setdefault("semantic_distance_from_competitors", "")
                analysis.setdefault("neologism_appropriateness", "")
                analysis.setdefault("overall_readability_score", "")
                analysis.setdefault("notes", "")
                analysis.setdefault("rank", 0)  # numeric
                
                linguistic_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error in linguistic analysis for {brand_name}: {str(e)}")
                # Create a minimal analysis entry with required fields for failed analyses
                linguistic_analyses.append({
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "pronunciation_ease": f"Error: {str(e)}",
                    "notes": f"Analysis failed: {str(e)}"
                })
        
        return {
            "linguistic_analysis_output": linguistic_analyses
        }
        
    except Exception as e:
        logger.error(f"Error in process_linguistic_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "linguistic_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }],
            "linguistic_analysis_output": []
        }

async def process_cultural_analysis(state: BrandNameGenerationState, agent: CulturalSensitivityExpert) -> Dict[str, Any]:
    """
    Processes cultural sensitivity analysis for each brand name.
    
    Args:
        state: The current workflow state
        agent: The cultural sensitivity expert agent
        
    Returns:
        Dict with cultural_sensitivity_output containing the analysis results
    """
    try:
        logger.info(f"Running cultural sensitivity analysis...")
        
        # Extract required input from state
        run_id = state.run_id
        brand_names = [name["brand_name"] for name in state.generated_names]
        brand_context = getattr(state, "brand_context", {})
        brand_values = getattr(state, "brand_values", [])
        
        # Run cultural sensitivity analysis
        cultural_analyses = []
        for brand_name in brand_names:
            try:
                # Call the agent to perform cultural analysis
                analysis = await agent.analyze_brand_name(
                    run_id=run_id,
                    brand_name=brand_name,
                    brand_context=brand_context,
                    brand_values=brand_values
                )
                
                # Ensure all required fields are present and have appropriate data types
                # Required fields (is_nullable=NO)
                analysis["run_id"] = run_id
                analysis["brand_name"] = brand_name
                
                # Optional fields with appropriate data types
                analysis.setdefault("cultural_connotations", "")
                analysis.setdefault("symbolic_meanings", "")
                analysis.setdefault("alignment_with_cultural_values", "")
                analysis.setdefault("religious_sensitivities", "")
                analysis.setdefault("social_political_taboos", "")
                analysis.setdefault("body_part_bodily_function_connotations", False)  # boolean
                analysis.setdefault("age_related_connotations", "")
                analysis.setdefault("gender_connotations", "")
                analysis.setdefault("regional_variations", "")
                analysis.setdefault("historical_meaning", "")
                analysis.setdefault("current_event_relevance", "")
                analysis.setdefault("overall_risk_rating", "")
                analysis.setdefault("notes", "")
                analysis.setdefault("rank", 0)  # numeric
                
                cultural_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error in cultural sensitivity analysis for {brand_name}: {str(e)}")
                # Create a minimal analysis entry with required fields for failed analyses
                cultural_analyses.append({
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "cultural_connotations": f"Error: {str(e)}",
                    "notes": f"Analysis failed: {str(e)}"
                })
        
        return {
            "cultural_sensitivity_output": cultural_analyses
        }
        
    except Exception as e:
        logger.error(f"Error in process_cultural_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "cultural_sensitivity_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }],
            "cultural_sensitivity_output": []
        }

async def process_translation_analysis(state: BrandNameGenerationState, agent: TranslationAnalysisExpert) -> Dict[str, Any]:
    """
    Processes translation analysis for each brand name.
    
    Args:
        state: The current workflow state
        agent: The translation analysis expert agent
        
    Returns:
        Dict with translation_analysis_output containing the analysis results
    """
    try:
        logger.info(f"Running translation analysis...")
        
        # Extract required input from state
        run_id = state.run_id
        brand_names = [name["brand_name"] for name in state.generated_names]
        
        # Run translation analysis
        translation_analyses = []
        for brand_name in brand_names:
            try:
                # Call the agent to perform translation analysis
                analysis = await agent.analyze_brand_name(
                    run_id=run_id,
                    brand_name=brand_name
                )
                
                # Ensure all required fields are present and have appropriate data types
                # Required fields (is_nullable=NO)
                analysis["run_id"] = run_id
                analysis["brand_name"] = brand_name
                analysis["target_language"] = analysis.get("target_language", "multiple")
                
                # Optional fields with appropriate data types
                analysis.setdefault("direct_translation", "")
                analysis.setdefault("semantic_shift", "")
                analysis.setdefault("pronunciation_difficulty", "")
                analysis.setdefault("phonetic_similarity_undesirable", False)  # boolean
                analysis.setdefault("phonetic_retention", "")
                analysis.setdefault("cultural_acceptability", "")
                analysis.setdefault("adaptation_needed", False)  # boolean
                analysis.setdefault("proposed_adaptation", "")
                analysis.setdefault("brand_essence_preserved", "")
                analysis.setdefault("global_consistency_vs_localization", "")
                analysis.setdefault("notes", "")
                analysis.setdefault("rank", 0)  # numeric
                
                translation_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error in translation analysis for {brand_name}: {str(e)}")
                # Create a minimal analysis entry with required fields for failed analyses
                translation_analyses.append({
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "target_language": "multiple",
                    "direct_translation": f"Error: {str(e)}",
                    "notes": f"Analysis failed: {str(e)}"
                })
        
        return {
            "translation_analysis_output": translation_analyses
        }
        
    except Exception as e:
        logger.error(f"Error in process_translation_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "translation_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }],
            "translation_analysis_output": []
        }

async def process_domain_analysis(state: BrandNameGenerationState, agent: DomainAnalysisExpert) -> Dict[str, Any]:
    """Process domain name analysis with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            # Prepare brand names for analysis
            brand_names = [name["brand_name"] for name in state.generated_names]
            brand_context = getattr(state, "brand_context", {})
            
            # Run domain analysis
            domain_analyses = []
            for brand_name in brand_names:
                try:
                    analysis = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context
                    )
                    domain_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in domain analysis for {brand_name}: {str(e)}")
                    # Continue with other brand names even if one fails
            
            # Create a deep copy of generated names to update
            generated_names = [dict(name) for name in state.generated_names]
            
            # Update with domain analysis results
            for i, name_data in enumerate(generated_names):
                brand_name = name_data["brand_name"]
                for analysis in domain_analyses:
                    if analysis.get("brand_name") == brand_name:
                        name_data["domain_analysis"] = analysis
                        break
            
            return {
                "generated_names": generated_names,
                "domain_analyses": domain_analyses
            }
            
    except Exception as e:
        logger.error(f"Error in process_domain_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "domain_analysis",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_seo_analysis(state: BrandNameGenerationState, agent: SEOOnlineDiscoveryExpert) -> Dict[str, Any]:
    """Process SEO analysis with error handling and tracing."""
    try:
        with tracing_v2_enabled():
            # Prepare brand names for analysis
            brand_names = [name["brand_name"] for name in state.generated_names]
            brand_context = getattr(state, "brand_context", {})
            
            # Run SEO analysis
            seo_analyses = []
            for brand_name in brand_names:
                try:
                    analysis = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context
                    )
                    seo_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in SEO analysis for {brand_name}: {str(e)}")
                    # Continue with other brand names even if one fails
            
            # Create a deep copy of generated names to update
            generated_names = [dict(name) for name in state.generated_names]
            
            # Update with SEO analysis results
            for i, name_data in enumerate(generated_names):
                brand_name = name_data["brand_name"]
                for analysis in seo_analyses:
                    if analysis.get("brand_name") == brand_name:
                        name_data["seo_analysis"] = analysis
                        break
            
            return {
                "generated_names": generated_names,
                "seo_analyses": seo_analyses
            }
            
    except Exception as e:
        logger.error(f"Error in process_seo_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "seo_analysis",
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
            # Prepare brand names for analysis
            brand_names = [name["brand_name"] for name in state.generated_names]
            brand_context = getattr(state, "brand_context", {})
            
            # Run competitor analysis
            competitor_analyses = []
            for brand_name in brand_names:
                try:
                    analysis = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context
                    )
                    competitor_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in competitor analysis for {brand_name}: {str(e)}")
                    # Continue with other brand names even if one fails
            
            # Create a deep copy of generated names to update
            generated_names = [dict(name) for name in state.generated_names]
            
            # Update with competitor analysis results
            for i, name_data in enumerate(generated_names):
                brand_name = name_data["brand_name"]
                for analysis in competitor_analyses:
                    if analysis.get("brand_name") == brand_name:
                        name_data["competitor_analysis"] = analysis
                        break
            
            return {
                "generated_names": generated_names,
                "competitor_analyses": competitor_analyses
            }
            
    except Exception as e:
        logger.error(f"Error in process_competitor_analysis: {str(e)}")
        return {
            "errors": [{
                "step": "competitor_analysis",
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
            # Prepare brand names for analysis
            brand_names = [name["brand_name"] for name in state.generated_names]
            brand_context = getattr(state, "brand_context", {})
            
            # Run survey simulation
            survey_analyses = []
            for brand_name in brand_names:
                try:
                    analysis = await agent.analyze_brand_name(
                        run_id=state.run_id,
                        brand_name=brand_name,
                        brand_context=brand_context
                    )
                    survey_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in survey simulation for {brand_name}: {str(e)}")
                    # Continue with other brand names even if one fails
            
            # Create a deep copy of generated names to update
            generated_names = [dict(name) for name in state.generated_names]
            
            # Update with survey simulation results
            for i, name_data in enumerate(generated_names):
                brand_name = name_data["brand_name"]
                for analysis in survey_analyses:
                    if analysis.get("brand_name") == brand_name:
                        name_data["survey_analysis"] = analysis
                        break
            
            return {
                "generated_names": generated_names,
                "survey_analyses": survey_analyses
            }
            
    except Exception as e:
        logger.error(f"Error in process_survey_simulation: {str(e)}")
        return {
            "errors": [{
                "step": "survey_simulation",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }],
            "status": "error"
        }

async def process_analyses(state: BrandNameGenerationState, analyzers: List[Any]) -> Dict[str, Any]:
    """
    Legacy function that processes all analyses in one step for backward compatibility.
    
    Args:
        state: The current workflow state
        analyzers: List of analyzer agents
        
    Returns:
        Dict with all analysis outputs
    """
    try:
        logger.info(f"Running combined analyses for backward compatibility...")
        
        # Extract required input from state
        run_id = state.run_id
        brand_names = [name["brand_name"] for name in state.generated_names]
        brand_context = getattr(state, "brand_context", {})
        brand_values = getattr(state, "brand_values", [])
        
        # Initialize outputs
        semantic_analyses = []
        linguistic_analyses = []
        cultural_analyses = []
        translation_analyses = []
        
        # Get agents by type
        semantic_agent = next((a for a in analyzers if isinstance(a, SemanticAnalysisExpert)), None)
        linguistic_agent = next((a for a in analyzers if isinstance(a, LinguisticsExpert)), None)
        cultural_agent = next((a for a in analyzers if isinstance(a, CulturalSensitivityExpert)), None)
        translation_agent = next((a for a in analyzers if isinstance(a, TranslationAnalysisExpert)), None)
        
        # Process analyses concurrently
        async def process_brand_name(brand_name):
            results = {}
            
            # Semantic analysis
            if semantic_agent:
                try:
                    analysis = await semantic_agent.analyze_brand_name(
                        run_id=run_id,
                        brand_name=brand_name,
                        brand_context=brand_context,
                        brand_values=brand_values
                    )
                    
                    # Ensure required fields are present
                    analysis["run_id"] = run_id
                    analysis["task_name"] = "Conduct_Semantic_Analysis"
                    analysis["brand_name"] = brand_name
                    
                    # Set defaults for important fields to match Supabase schema
                    analysis.setdefault("denotative_meaning", "")
                    analysis.setdefault("brand_personality", "")
                    analysis.setdefault("memorability_score", 0.0)
                    
                    results["semantic"] = analysis
                    semantic_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in semantic analysis for {brand_name}: {str(e)}")
                    error_analysis = {
                        "run_id": run_id,
                        "task_name": "Conduct_Semantic_Analysis",
                        "brand_name": brand_name,
                        "denotative_meaning": f"Error: {str(e)}"
                    }
                    results["semantic"] = error_analysis
                    semantic_analyses.append(error_analysis)
            
            # Linguistic analysis
            if linguistic_agent:
                try:
                    analysis = await linguistic_agent.analyze_brand_name(
                        run_id=run_id,
                        brand_name=brand_name, 
                        brand_context=brand_context,
                        brand_values=brand_values
                    )
                    
                    # Ensure required fields are present
                    analysis["run_id"] = run_id
                    analysis["brand_name"] = brand_name
                    
                    # Set defaults for important fields to match Supabase schema
                    analysis.setdefault("pronunciation_ease", "")
                    analysis.setdefault("overall_readability_score", "")
                    
                    results["linguistic"] = analysis
                    linguistic_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in linguistic analysis for {brand_name}: {str(e)}")
                    error_analysis = {
                        "run_id": run_id,
                        "brand_name": brand_name,
                        "pronunciation_ease": f"Error: {str(e)}",
                        "notes": f"Analysis failed: {str(e)}"
                    }
                    results["linguistic"] = error_analysis
                    linguistic_analyses.append(error_analysis)
            
            # Cultural sensitivity analysis
            if cultural_agent:
                try:
                    analysis = await cultural_agent.analyze_brand_name(
                        run_id=run_id,
                        brand_name=brand_name,
                        brand_context=brand_context,
                        brand_values=brand_values
                    )
                    
                    # Ensure required fields are present
                    analysis["run_id"] = run_id
                    analysis["brand_name"] = brand_name
                    
                    # Set defaults for important fields to match Supabase schema
                    analysis.setdefault("cultural_connotations", "")
                    analysis.setdefault("overall_risk_rating", "")
                    
                    results["cultural"] = analysis
                    cultural_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in cultural sensitivity analysis for {brand_name}: {str(e)}")
                    error_analysis = {
                        "run_id": run_id,
                        "brand_name": brand_name,
                        "cultural_connotations": f"Error: {str(e)}",
                        "notes": f"Analysis failed: {str(e)}"
                    }
                    results["cultural"] = error_analysis
                    cultural_analyses.append(error_analysis)
            
            # Translation analysis
            if translation_agent:
                try:
                    analysis = await translation_agent.analyze_brand_name(
                        run_id=run_id,
                        brand_name=brand_name
                    )
                    
                    # Ensure required fields are present
                    analysis["run_id"] = run_id
                    analysis["brand_name"] = brand_name
                    analysis["target_language"] = analysis.get("target_language", "multiple")
                    
                    # Set defaults for important fields to match Supabase schema
                    analysis.setdefault("direct_translation", "")
                    analysis.setdefault("semantic_shift", "")
                    
                    results["translation"] = analysis
                    translation_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error in translation analysis for {brand_name}: {str(e)}")
                    error_analysis = {
                        "run_id": run_id,
                        "brand_name": brand_name,
                        "target_language": "multiple",
                        "direct_translation": f"Error: {str(e)}",
                        "notes": f"Analysis failed: {str(e)}"
                    }
                    results["translation"] = error_analysis
                    translation_analyses.append(error_analysis)
            
            return results
        
        # Process all brand names concurrently
        tasks = [process_brand_name(brand_name) for brand_name in brand_names]
        analyses_results = await asyncio.gather(*tasks)
        
        # Create a deep copy of generated names to update
        generated_names = [dict(name) for name in state.generated_names]
        
        # Update with analysis results
        for i, name_data in enumerate(generated_names):
            brand_name = name_data["brand_name"]
            for result in analyses_results:
                if result.get("semantic", {}).get("brand_name") == brand_name:
                    name_data["semantic_analysis"] = result.get("semantic", {})
                if result.get("linguistic", {}).get("brand_name") == brand_name:
                    name_data["linguistic_analysis"] = result.get("linguistic", {})
                if result.get("cultural", {}).get("brand_name") == brand_name:
                    name_data["cultural_analysis"] = result.get("cultural", {})
                if result.get("translation", {}).get("brand_name") == brand_name:
                    name_data["translation_analysis"] = result.get("translation", {})
        
        return {
            "generated_names": generated_names,
            "semantic_analysis_output": semantic_analyses,
            "linguistic_analysis_output": linguistic_analyses,
            "cultural_sensitivity_output": cultural_analyses,
            "translation_analysis_output": translation_analyses
        }
        
    except Exception as e:
        logger.error(f"Error in process_analyses: {str(e)}")
        return {
            "errors": [{
                "step": "analyses",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }],
            "semantic_analysis_output": [],
            "linguistic_analysis_output": [],
            "cultural_sensitivity_output": [],
            "translation_analysis_output": []
        }

async def process_evaluation(state: BrandNameGenerationState, agent: BrandNameEvaluator) -> Dict[str, Any]:
    """
    Processes evaluation of brand names based on all analyses.
    
    Args:
        state: The current workflow state
        agent: The brand name evaluator agent
        
    Returns:
        Dict with evaluated_names containing the evaluation results
    """
    try:
        logger.info(f"Running brand name evaluation...")
        
        # Extract required input from state
        run_id = state.run_id
        brand_names = [name["brand_name"] for name in state.generated_names]
        
        # Collect all analyses outputs
        semantic_analyses = getattr(state, "semantic_analysis_output", [])
        linguistic_analyses = getattr(state, "linguistic_analysis_output", [])
        cultural_analyses = getattr(state, "cultural_sensitivity_output", [])
        translation_analyses = getattr(state, "translation_analysis_output", [])
        
        # Organize analyses by brand name for easy access
        analyses_by_name = {}
        for brand_name in brand_names:
            analyses_by_name[brand_name] = {
                "semantic": next((a for a in semantic_analyses if a.get("brand_name") == brand_name), {}),
                "linguistic": next((a for a in linguistic_analyses if a.get("brand_name") == brand_name), {}),
                "cultural": next((a for a in cultural_analyses if a.get("brand_name") == brand_name), {}),
                "translation": next((a for a in translation_analyses if a.get("brand_name") == brand_name), {})
            }
        
        # Evaluate brand names
        evaluations = []
        for brand_name in brand_names:
            try:
                analyses = analyses_by_name.get(brand_name, {})
                
                evaluation = await agent.evaluate_brand_name(
                    run_id=run_id,
                    brand_name=brand_name,
                    semantic_analysis=analyses.get("semantic", {}),
                    linguistic_analysis=analyses.get("linguistic", {}),
                    cultural_analysis=analyses.get("cultural", {}),
                    translation_analysis=analyses.get("translation", {})
                )
                
                # Ensure all required fields are present and have appropriate data types
                # Required fields (is_nullable=NO)
                evaluation["run_id"] = run_id
                evaluation["brand_name"] = brand_name
                evaluation["shortlist_status"] = evaluation.get("shortlist_status", False)
                
                # Optional fields with appropriate data types
                evaluation.setdefault("strategic_alignment_score", 0)  # integer
                evaluation.setdefault("distinctiveness_score", 0)  # integer
                evaluation.setdefault("memorability_score", 0)  # integer
                evaluation.setdefault("pronounceability_score", 0)  # integer
                evaluation.setdefault("meaningfulness_score", 0)  # integer
                evaluation.setdefault("brand_fit_score", 0)  # integer
                evaluation.setdefault("domain_viability_score", 0)  # integer
                evaluation.setdefault("overall_score", 0)  # integer
                evaluation.setdefault("evaluation_comments", "")
                evaluation.setdefault("created_at", datetime.now().isoformat())
                evaluation.setdefault("rank", 0.0)  # numeric
                
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error in evaluation for {brand_name}: {str(e)}")
                # Create a minimal evaluation entry with required fields for failed evaluations
                evaluations.append({
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "shortlist_status": False,
                    "evaluation_comments": f"Evaluation failed: {str(e)}"
                })
        
        # Sort evaluations by overall score (descending)
        evaluations.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        # Assign ranks based on sorted order
        for i, evaluation in enumerate(evaluations):
            evaluation["rank"] = i + 1
        
        return {
            "evaluated_names": evaluations
        }
        
    except Exception as e:
        logger.error(f"Error in process_evaluation: {str(e)}")
        return {
            "errors": [{
                "step": "evaluation",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }],
            "evaluated_names": []
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
    if settings.langchain_tracing_v2:
        return LangChainTracer(
            project_name=settings.langchain_project,
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