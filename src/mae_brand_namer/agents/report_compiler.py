"""Report Compiler for generating comprehensive brand name analysis reports."""

# Standard library imports
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Third-party imports
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from postgrest import APIError

# Local application imports
from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)

class ReportCompiler:
    """Expert in compiling and formatting comprehensive brand naming reports."""
    
    def __init__(self, dependencies: Dependencies):
        """Initialize the ReportCompiler with dependencies."""
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        # Agent identity
        self.role = "Brand Name Analysis Report Compiler"
        self.goal = """Compile comprehensive, actionable reports that synthesize all brand name analyses into clear, 
        strategic recommendations for decision makers."""
        self.backstory = """You are an expert in data synthesis and report compilation, specializing in brand naming analysis. 
        Your expertise helps transform complex analyses into clear, actionable insights that drive informed decisions."""
        
        try:
            # Load prompts
            prompt_dir = Path(__file__).parent / "prompts" / "report_compiler"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.compilation_prompt = load_prompt(str(prompt_dir / "compilation.yaml"))
            
            # Define output schemas for structured parsing
            self.output_schemas = [
                ResponseSchema(name="executive_summary", description="High-level summary of findings"),
                ResponseSchema(name="brand_context_summary", description="Summary of brand context"),
                ResponseSchema(name="methodology", description="Methods used for name generation and evaluation"),
                ResponseSchema(name="name_generation_overview", description="Overview of name generation process"),
                ResponseSchema(name="evaluation_criteria", description="Criteria used for evaluation"),
                ResponseSchema(name="analysis_results", description="Summary of analysis results"),
                ResponseSchema(name="shortlisted_names", description="List of shortlisted brand names"),
                ResponseSchema(name="top_recommendations", description="Top brand name recommendations"),
                ResponseSchema(name="strategic_insights", description="Strategic insights from the process"),
                ResponseSchema(name="next_steps", description="Recommended next steps"),
                ResponseSchema(name="appendix", description="Additional detailed information")
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
            
            # Set up the prompt template
            system_message = SystemMessage(content=self.system_prompt.format())
            human_template = self.compilation_prompt.template
            self.prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessage(content=human_template)
            ])
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=0.2,
                google_api_key=settings.google_api_key,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.error(
                "Error initializing ReportCompiler",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise

    async def compile_report(
        self,
        run_id: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compile a comprehensive report from all analyses.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            state (Dict[str, Any]): Current workflow state with all analyses
            
        Returns:
            Dict[str, Any]: Compiled report
            
        Raises:
            ValueError: If report compilation fails
        """
        try:
            with tracing_enabled(
                tags={
                    "agent": "ReportCompiler",
                    "run_id": run_id
                }
            ):
                # Prepare state data for the prompt
                state_data = {
                    "run_id": run_id,
                    "brand_context": state.get("brand_context", {}),
                    "generated_names": state.get("generated_names", []),
                    "linguistic_analysis": state.get("linguistic_analysis_results", {}),
                    "cultural_analysis": state.get("cultural_analysis_results", {}),
                    "evaluation_results": state.get("evaluation_results", {}),
                    "shortlisted_names": state.get("shortlisted_names", []),
                    "market_research": state.get("market_research_results", {}),
                    "domain_analysis": state.get("domain_analysis_results", []),
                    "seo_analysis": state.get("seo_analysis_results", []),
                    "competitor_analysis": state.get("competitor_analysis_results", [])
                }
                
                # Format prompt with parser instructions
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    state_data=state_data
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse the response
                report_data = self.output_parser.parse(response.content)
                
                # Add metadata
                report = {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    **report_data
                }
                
                # Store the report
                await self._store_report(run_id, report)
                
                # Generate report URL
                report_url = f"{settings.report_base_url}/{run_id}"
                
                return {
                    "report_data": report,
                    "report_url": report_url
                }
                
        except APIError as e:
            logger.error(
                "Supabase API error in report compilation",
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
                "Error in report compilation",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    async def _store_report(
        self,
        run_id: str,
        report: Dict[str, Any]
    ) -> None:
        """Store compiled report in Supabase."""
        try:
            await self.supabase.table("reports").insert(report).execute()
            
        except APIError as e:
            logger.error(
                "Supabase API error storing report",
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
                "Unexpected error storing report",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise 