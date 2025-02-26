"""Brand naming workflow using LangGraph."""

from typing import Dict, Any, List, Optional, TypedDict, Tuple
from datetime import datetime
import asyncio
from functools import partial
import json
import os
import uuid

from langgraph.graph import StateGraph
from langsmith import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import SystemMessage, HumanMessage
from postgrest import APIError as PostgrestError
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
        
        # Method 2: From a nested state object
        if not run_id and inputs and isinstance(inputs, dict):
            state = inputs.get("state")
            if isinstance(state, dict):
                run_id = state.get("run_id")
        
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
    
    # Build dependencies object including the LangSmith client
    deps = {
        "supabase": supabase_manager,
        "langsmith": langsmith_client
    }
    
    # Define agent nodes
    workflow.add_node("generate_uid", lambda state: asyncio.ensure_future(process_uid(state, UIDGeneratorAgent())))
    workflow.add_node("understand_brand_context", lambda state: process_brand_context(state, BrandContextExpert(
        supabase=supabase_manager, 
        langsmith=langsmith_client
    )))
    workflow.add_node("generate_brand_names", lambda state: process_brand_names(state, BrandNameCreationExpert(
        supabase=supabase_manager,
        langsmith=langsmith_client
    )))
    workflow.add_node("process_linguistics", lambda state: process_linguistics(state, LinguisticsExpert(
        supabase=supabase_manager,
        langsmith=langsmith_client
    )))
    workflow.add_node("process_cultural_sensitivity", lambda state: process_cultural_sensitivity(state, CulturalSensitivityExpert(
        supabase=supabase_manager,
        langsmith=langsmith_client
    )))
    workflow.add_node("process_analyses", lambda state: process_analyses(state, [
        SemanticAnalysisExpert(supabase=supabase_manager, langsmith=langsmith_client),
        LinguisticsExpert(supabase=supabase_manager, langsmith=langsmith_client),
        CulturalSensitivityExpert(supabase=supabase_manager, langsmith=langsmith_client),
        TranslationAnalysisExpert(supabase=supabase_manager, langsmith=langsmith_client),
        DomainAnalysisExpert(supabase=supabase_manager, langsmith=langsmith_client),
        SEOOnlineDiscoveryExpert(supabase=supabase_manager, langsmith=langsmith_client),
        CompetitorAnalysisExpert(supabase=supabase_manager, langsmith=langsmith_client),
        SurveySimulationExpert(supabase=supabase_manager, langsmith=langsmith_client)
    ]))
    workflow.add_node("process_evaluation", lambda state: process_evaluation(state, BrandNameEvaluator(
        supabase=supabase_manager,
        langsmith=langsmith_client
    )))
    workflow.add_node("process_market_research", lambda state: process_market_research(state, MarketResearchExpert(
        supabase=supabase_manager,
        langsmith=langsmith_client
    )))
    workflow.add_node("compile_report", lambda state: process_report(state, ReportCompiler(
        supabase=supabase_manager,
        langsmith=langsmith_client
    )))
    workflow.add_node("store_report", lambda state: process_report_storage(state, ReportStorer(
        supabase=supabase_manager,
        langsmith=langsmith_client
    )))
    
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

async def process_uid(state: BrandNameGenerationState, agent: UIDGeneratorAgent) -> BrandNameGenerationState:
    """Process UID generation with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "generate_uid"}):
            run_id = await agent.generate_uid()
            state["run_id"] = run_id
            state["start_time"] = datetime.now().isoformat()
            state["status"] = "uid_generated"
            return state
            
    except Exception as e:
        logger.error("Error generating UID", error=str(e))
        state["errors"].append({
            "task": "generate_uid",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_brand_context(state: BrandNameGenerationState, agent: BrandContextExpert) -> BrandNameGenerationState:
    """Process brand context extraction with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "understand_brand_context", "run_id": state["run_id"]}):
            brand_context = await agent.extract_brand_context(
                user_prompt=state["user_prompt"],
                run_id=state["run_id"]
            )
            state["brand_context"] = brand_context
            state["status"] = "context_extracted"
            return state
            
    except Exception as e:
        logger.error(
            "Error extracting brand context",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "understand_brand_context",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_brand_names(state: BrandNameGenerationState, agent: BrandNameCreationExpert) -> BrandNameGenerationState:
    """Process brand name generation with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "generate_brand_names", "run_id": state["run_id"]}):
            brand_names = await agent.generate_brand_names(
                brand_context=state["brand_context"], 
                brand_values=state["brand_context"]["brand_values"],
                purpose=state["brand_context"]["brand_purpose"],
                key_attributes=state["brand_context"]["brand_personality"],
                run_id=state["run_id"]
            )
            state["generated_names"] = brand_names
            state["status"] = "names_generated"
            return state
            
    except Exception as e:
        logger.error(
            "Error generating brand names",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "generate_brand_names",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_linguistics(state: BrandNameGenerationState, agent: LinguisticsExpert) -> BrandNameGenerationState:
    """Process linguistic analysis with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "analyze_linguistics", "run_id": state["run_id"]}):
            linguistic_results = {}
            for brand_name in state["generated_names"]:
                result = await agent.analyze_brand_name(
                    run_id=state["run_id"],
                    brand_name=brand_name["brand_name"],
                    brand_context=state["brand_context"]
                )
                linguistic_results[brand_name["brand_name"]] = result
            state["linguistic_analysis_results"] = linguistic_results
            state["status"] = "linguistics_analyzed"
            return state
            
    except Exception as e:
        logger.error(
            "Error analyzing linguistics",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "analyze_linguistics",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_cultural_sensitivity(state: BrandNameGenerationState, agent: CulturalSensitivityExpert) -> BrandNameGenerationState:
    """Process cultural sensitivity analysis with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "analyze_cultural_sensitivity", "run_id": state["run_id"]}):
            cultural_results = {}
            for brand_name in state["generated_names"]:
                result = await agent.analyze_brand_name(
                    run_id=state["run_id"],
                    brand_name=brand_name["brand_name"],
                    brand_context=state["brand_context"]
                )
                cultural_results[brand_name["brand_name"]] = result
            state["cultural_analysis_results"] = cultural_results
            state["status"] = "cultural_sensitivity_analyzed"
            return state
            
    except Exception as e:
        logger.error(
            "Error analyzing cultural sensitivity",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "analyze_cultural_sensitivity",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_analyses(state: BrandNameGenerationState, analyzers: List[Any]) -> BrandNameGenerationState:
    """Run multiple analysis agents concurrently using LangGraph map."""
    try:
        with tracing_enabled(tags={"task": "process_analyses", "run_id": state["run_id"]}):
            # Prepare input data for map
            input_data = [
                {
                    "run_id": state["run_id"],
                    "brand_name": name["brand_name"],
                    "brand_context": state["brand_context"]
                }
                for name in state["generated_names"]
            ]
            
            # Run analyses concurrently using client.map
            if state.get("client"):
                # Use the client from state for map operation with callbacks
                results = await state["client"].map(
                    analyzers,
                    input_data,
                    lambda agent, data: agent.analyze_brand_name(**data),
                    config={"callbacks": [state["client"]]}
                )
            else:
                # If no client is provided, create one for this operation
                from langsmith import Client
                logger.info("Creating new LangGraph client for map operation")
                client = Client()
                
                # Use the new client for map operation
                results = await client.map(
                    analyzers,
                    input_data,
                    lambda agent, data: agent.analyze_brand_name(**data),
                    config={"callbacks": [client]}
                )
            
            state["analysis_results"] = {
                "results": results
            }
            state["status"] = "analyses_completed"
            return state
            
    except Exception as e:
        logger.error(
            "Error running analyses",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "process_analyses",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_market_research(state: BrandNameGenerationState, agent: MarketResearchExpert) -> BrandNameGenerationState:
    """Process market research analysis with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "conduct_market_research", "run_id": state["run_id"]}):
            market_research = await agent.analyze_market_potential(
                run_id=state["run_id"],
                brand_names=[name["brand_name"] for name in state["generated_names"]],
                brand_context=state["brand_context"]
            )
            state["market_research_results"] = market_research
            state["status"] = "market_research_conducted"
            return state
            
    except Exception as e:
        logger.error(
            "Error conducting market research",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "conduct_market_research",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_report(state: BrandNameGenerationState, agent: ReportCompiler) -> BrandNameGenerationState:
    """Process report compilation with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "compile_report", "run_id": state["run_id"]}):
            report = await agent.compile_report(
                run_id=state["run_id"],
                state=state
            )
            state["compiled_report"] = report
            state["status"] = "report_compiled"
            return state
            
    except Exception as e:
        logger.error(
            "Error compiling report",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "compile_report",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_report_storage(state: BrandNameGenerationState, agent: ReportStorer) -> BrandNameGenerationState:
    """Process report storage with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "store_report", "run_id": state["run_id"]}):
            storage_result = await agent.store_report(
                run_id=state["run_id"],
                report_data=state["compiled_report"]
            )
            state["report_storage_metadata"] = storage_result
            state["status"] = "report_stored"
            return state
            
    except Exception as e:
        logger.error(
            "Error storing report",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "store_report",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

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