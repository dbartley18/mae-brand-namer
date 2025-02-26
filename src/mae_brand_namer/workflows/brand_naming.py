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

def create_workflow(langsmith_client: Optional[Any] = None) -> StateGraph:
    """
    Create and configure the brand naming workflow graph.
    
    This function sets up the workflow state graph with all nodes, edges, and process monitoring.
    It uses callback handlers for task logging and error management.
    
    Args:
        langsmith_client: Optional LangSmith client for tracing and evaluation
        
    Returns:
        Configured StateGraph for the brand naming workflow
    """
    # Create a single Supabase manager instance
    supabase_manager = SupabaseManager()
    
    # Initialize supervisor for process monitoring
    supervisor = ProcessSupervisor(supabase=supabase_manager)
    
    # Create workflow state graph
    workflow = StateGraph(BrandNameGenerationState)
    
    # Helper function to wrap async functions so they are properly handled by LangGraph
    def wrap_async_node(func):
        """Wraps an async function to ensure it's properly awaited before returning"""
        async def wrapper(state):
            result = await func(state)
            return result
        return wrapper
    
    # Define agent nodes with proper async handling
    workflow.add_node("generate_uid", process_uid)
    
    # Wrap process_brand_context and other async functions
    workflow.add_node("understand_brand_context", 
        wrap_async_node(lambda state: process_brand_context(state, BrandContextExpert(
            supabase=supabase_manager
        )))
    )
    
    workflow.add_node("generate_brand_names", 
        wrap_async_node(lambda state: process_brand_names(state, BrandNameCreationExpert(
            supabase=supabase_manager
        )))
    )
    
    workflow.add_node("process_linguistics", 
        wrap_async_node(lambda state: process_linguistics(state, LinguisticsExpert(
            supabase=supabase_manager
        )))
    )
    
    workflow.add_node("process_cultural_sensitivity", 
        wrap_async_node(lambda state: process_cultural_sensitivity(state, CulturalSensitivityExpert(
            supabase=supabase_manager
        )))
    )
    
    workflow.add_node("process_analyses", 
        wrap_async_node(lambda state: process_analyses(state, [
            SemanticAnalysisExpert(supabase=supabase_manager),
            LinguisticsExpert(supabase=supabase_manager),
            CulturalSensitivityExpert(supabase=supabase_manager),
            TranslationAnalysisExpert(supabase=supabase_manager),
            DomainAnalysisExpert(supabase=supabase_manager),
            SEOOnlineDiscoveryExpert(supabase=supabase_manager),
            CompetitorAnalysisExpert(supabase=supabase_manager),
            SurveySimulationExpert(supabase=supabase_manager)
        ]))
    )
    
    workflow.add_node("process_evaluation", 
        wrap_async_node(lambda state: process_evaluation(state, BrandNameEvaluator(
            supabase=supabase_manager
        )))
    )
    
    workflow.add_node("process_market_research", 
        wrap_async_node(lambda state: process_market_research(state, MarketResearchExpert(
            supabase=supabase_manager
        )))
    )
    
    workflow.add_node("compile_report", 
        wrap_async_node(lambda state: process_report(state, ReportCompiler(
            supabase=supabase_manager
        )))
    )
    
    workflow.add_node("store_report", 
        wrap_async_node(lambda state: process_report_storage(state, ReportStorer(
            supabase=supabase_manager
        )))
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
            brand_context = await agent.extract_brand_context(
                user_prompt=state.user_prompt,
                run_id=state.run_id
            )
            
            # Return dictionary of state updates instead of the state object
            return {
                "brand_identity_brief": brand_context["brand_identity_brief"],
                "brand_promise": brand_context["brand_promise"],
                "brand_values": brand_context["brand_values"],
                "brand_personality": brand_context["brand_personality"],
                "brand_tone_of_voice": brand_context["brand_tone_of_voice"],
                "brand_purpose": brand_context["brand_purpose"],
                "brand_mission": brand_context["brand_mission"],
                "target_audience": brand_context["target_audience"],
                "customer_needs": brand_context["customer_needs"],
                "market_positioning": brand_context["market_positioning"],
                "competitive_landscape": brand_context["competitive_landscape"],
                "industry_focus": brand_context["industry_focus"],
                "industry_trends": brand_context["industry_trends"]
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
            brand_names = await agent.generate_brand_names(
                brand_identity=state.brand_identity_brief,
                brand_promise=state.brand_promise,
                brand_values=state.brand_values,
                brand_personality=state.brand_personality,
                target_audience=state.target_audience,
                run_id=state.run_id
            )
            
            # Return dictionary of state updates
            return {
                "generated_names": brand_names
            }
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
                    name_data["semantic_analysis"] = linguistics_analysis[i]
            
            # Return dictionary of state updates
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
            
            # Return dictionary of state updates
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
            
            # Return dictionary of state updates
            return {
                "evaluation_results": evaluation_results,
                "shortlisted_names": shortlisted_names
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
            
            # Return dictionary of state updates
            return {
                "market_research_results": market_research
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
            
            # Return dictionary of state updates
            return {
                "compiled_report": report
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
            
            # Return dictionary of state updates
            return {
                "report_storage_metadata": storage_result
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
    
    # Create the workflow with LangSmith client
    workflow = create_workflow(langsmith_client=langsmith_client)
    
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