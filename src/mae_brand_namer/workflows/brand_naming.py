"""Brand naming workflow using LangGraph."""

from typing import Dict, Any, List, Optional, TypedDict, Tuple
from datetime import datetime
import asyncio
from functools import partial
import json
import os

from langchain.graphs import StateGraph
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import SystemMessage, HumanMessage
from supabase.lib.exceptions import APIError, PostgrestError

from ..agents import (
    UIDGeneratorAgent,
    BrandContextExpert,
    BrandNameGenerator,
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
    ProcessSupervisor
)
from ..utils.logging import get_logger
from ..config.settings import settings
from ..config.dependencies import Dependencies, create_dependencies
from ..models.state import BrandNameGenerationState

logger = get_logger(__name__)

async def run_parallel_analysis(
    state: BrandNameGenerationState,
    brand_names: List[str],
    brand_context: Dict[str, Any],
    dependencies: Dependencies
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run analysis agents in parallel for each brand name."""
    
    # Initialize analysis agents with dependencies
    semantic_expert = SemanticAnalysisExpert(dependencies)
    linguistic_expert = LinguisticsExpert(dependencies)
    cultural_expert = CulturalSensitivityExpert(dependencies)
    translation_expert = TranslationAnalysisExpert(dependencies)
    domain_expert = DomainAnalysisExpert(dependencies)
    seo_expert = SEOOnlineDiscoveryExpert(dependencies)
    competitor_expert = CompetitorAnalysisExpert(dependencies)
    survey_expert = SurveySimulationExpert(dependencies)
    
    async def analyze_name(name: str) -> Dict[str, Any]:
        """Run all analyses for a single name in parallel."""
        tasks = [
            semantic_expert.analyze_brand_name(state.run_id, name, brand_context),
            linguistic_expert.analyze_brand_name(state.run_id, name, brand_context),
            cultural_expert.analyze_brand_name(state.run_id, name, brand_context),
            translation_expert.analyze_brand_name(state.run_id, name, brand_context),
            domain_expert.analyze_brand_name(state.run_id, name, brand_context),
            seo_expert.analyze_brand_name(state.run_id, name, brand_context),
            competitor_expert.analyze_brand_name(state.run_id, name, brand_context),
            survey_expert.analyze_brand_name(state.run_id, name, brand_context)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        errors = []
        analyses = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "task": tasks[i].__name__,
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                analyses.update(result)
        
        return {
            "name": name,
            "analyses": analyses,
            "errors": errors
        }
    
    # Run analyses for all names in parallel
    analysis_tasks = [analyze_name(name) for name in brand_names]
    analysis_results = await asyncio.gather(*analysis_tasks)
    
    # Separate successful analyses and errors
    all_analyses = {}
    all_errors = []
    for result in analysis_results:
        all_analyses[result["name"]] = result["analyses"]
        all_errors.extend(result["errors"])
    
    return all_analyses, all_errors

def create_workflow(dependencies: Optional[Dependencies] = None) -> StateGraph:
    """Create the brand naming workflow graph."""
    if dependencies is None:
        dependencies = create_dependencies()
    
    # Initialize workflow graph
    workflow = StateGraph(StateGraph.from_dict({
        "run_id": None,
        "start_time": None,
        "user_prompt": None,
        "brand_context": None,
        "brand_names": None,
        "analyses": None,
        "report": None,
        "errors": [],
        "status": "initialized"
    }))
    
    # Initialize agents with dependencies
    uid_generator = UIDGeneratorAgent()
    brand_context_expert = BrandContextExpert(dependencies)
    brand_name_generator = BrandNameGenerator(dependencies)
    market_research_expert = MarketResearchExpert(dependencies)
    report_compiler = ReportCompiler(dependencies)
    report_storer = ReportStorer(dependencies)
    process_supervisor = ProcessSupervisor(dependencies)
    
    # Add nodes with pre/post processors and error handlers
    workflow.add_node("generate_uid", lambda x: process_uid(x, uid_generator))
    workflow.add_node("understand_brand_context", lambda x: process_brand_context(x, brand_context_expert))
    workflow.add_node("generate_brand_names", lambda x: process_brand_names(x, brand_name_generator))
    workflow.add_node("analyze_names", lambda x: process_parallel_analysis(x, dependencies))
    workflow.add_node("conduct_market_research", lambda x: process_market_research(x, market_research_expert))
    workflow.add_node("compile_report", lambda x: process_report(x, report_compiler))
    workflow.add_node("store_report", lambda x: process_report_storage(x, report_storer))
    
    # Define edges with conditional logic
    workflow.add_edge("generate_uid", "understand_brand_context")
    workflow.add_edge("understand_brand_context", "generate_brand_names")
    workflow.add_edge("generate_brand_names", "analyze_names")
    workflow.add_edge("analyze_names", "conduct_market_research")
    workflow.add_edge("conduct_market_research", "compile_report")
    workflow.add_edge("compile_report", "store_report")
    
    # Set entry point
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

async def process_brand_names(state: BrandNameGenerationState, agent: BrandNameGenerator) -> BrandNameGenerationState:
    """Process brand name generation with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "generate_brand_names", "run_id": state["run_id"]}):
            brand_names = await agent.generate_names(
                brand_context=state["brand_context"],
                run_id=state["run_id"]
            )
            state["brand_names"] = brand_names
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

async def process_parallel_analysis(state: BrandNameGenerationState, dependencies: Dependencies) -> BrandNameGenerationState:
    """Process parallel analysis of brand names."""
    try:
        with tracing_enabled(tags={"task": "analyze_names", "run_id": state["run_id"]}):
            analyses, errors = await run_parallel_analysis(
                state=state,
                brand_names=state["brand_names"],
                brand_context=state["brand_context"],
                dependencies=dependencies
            )
            state["analyses"] = analyses
            state["errors"].extend(errors)
            state["status"] = "names_analyzed"
            return state
            
    except Exception as e:
        logger.error(
            "Error analyzing brand names",
            run_id=state["run_id"],
            error=str(e)
        )
        state["errors"].append({
            "task": "analyze_names",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["status"] = "error"
        raise

async def process_market_research(state: BrandNameGenerationState, agent: MarketResearchExpert) -> BrandNameGenerationState:
    """Process market research with error handling and tracing."""
    try:
        with tracing_enabled(tags={"task": "conduct_market_research", "run_id": state["run_id"]}):
            market_research = await agent.research_names(
                brand_names=state["brand_names"],
                brand_context=state["brand_context"],
                run_id=state["run_id"]
            )
            state["market_research"] = market_research
            state["status"] = "research_completed"
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
                state_data=state
            )
            state["report"] = report
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
            await agent.store_report(
                run_id=state["run_id"],
                report_data=state["report"]
            )
            state["status"] = "completed"
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