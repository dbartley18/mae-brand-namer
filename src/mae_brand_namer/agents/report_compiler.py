"""Report Compiler for generating comprehensive brand name analysis reports."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="executive_summary", description="Concise summary of findings and recommendations", type="string"),
            ResponseSchema(name="brand_name_rankings", description="Structured table of scores across evaluation criteria", type="object"),
            ResponseSchema(name="brand_context_summary", description="Summary of brand context and competitive landscape", type="string"),
            ResponseSchema(name="semantic_analysis_summary", description="Key findings from semantic analysis", type="string"),
            ResponseSchema(name="linguistic_analysis_summary", description="Key findings from linguistic analysis", type="string"),
            ResponseSchema(name="cultural_sensitivity_summary", description="Key findings from cultural sensitivity analysis", type="string"),
            ResponseSchema(name="translation_feasibility_summary", description="Key findings from translation analysis", type="string"),
            ResponseSchema(name="domain_assessment_summary", description="Key findings from domain analysis", type="string"),
            ResponseSchema(name="competitor_benchmarking_summary", description="Key findings from competitor analysis", type="string"),
            ResponseSchema(name="seo_discoverability_summary", description="Key findings from SEO analysis", type="string"),
            ResponseSchema(name="market_perception_summary", description="Key findings from survey simulation", type="string"),
            ResponseSchema(name="strategic_recommendations", description="Final recommendations and next steps", type="array"),
            ResponseSchema(name="methodology_summary", description="Overview of the AI-driven evaluation methodology", type="string")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Create the prompt template with metadata for LangGraph Studio
        system_message = SystemMessage(
            content=f"""You are a Report Compilation Expert with the following profile:
            Role: {self.role}
            Goal: {self.goal}
            Backstory: {self.backstory}
            
            Compile a comprehensive report based on the provided analyses.
            Consider:
            1. Executive Summary & Key Findings
            2. Brand Context & Market Landscape
            3. Analysis Summaries by Category
            4. Strategic Recommendations
            5. Methodology & Process
            
            Format your response according to the following schema:
            {{format_instructions}}
            """,
            additional_kwargs={
                "metadata": {
                    "agent_type": "report_compiler",
                    "methodology": "Alina Wheeler's Designing Brand Identity"
                }
            }
        )
        human_template = """Compile a comprehensive report for the following brand name analyses:
        Brand Names: {brand_names}
        Brand Context: {brand_context}
        Analyses: {analyses}
        """
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

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
            with tracing_enabled(tags={"agent": "ReportCompiler", "run_id": run_id}):
                # Extract relevant data from state
                brand_names = state.get("brand_names", [])
                brand_context = state.get("brand_context", {})
                analyses = state.get("analyses", {})
                
                # Format prompt with parser instructions
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_names=json.dumps(brand_names, indent=2),
                    brand_context=json.dumps(brand_context, indent=2),
                    analyses=json.dumps(analyses, indent=2)
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse structured response
                report = self.output_parser.parse(response.content)
                
                # Add metadata
                report.update({
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "brand_names": brand_names,
                    "brand_context": brand_context,
                    "analyses": analyses
                })
                
                # Store report in Supabase
                await self._store_report(run_id, report)
                
                return report
                
        except Exception as e:
            logger.error(
                "Error compiling report",
                extra={
                    "run_id": run_id,
                    "error": str(e)
                }
            )
            raise ValueError(f"Report compilation failed: {str(e)}")
    
    async def _store_report(
        self,
        run_id: str,
        report: Dict[str, Any]
    ) -> None:
        """Store compiled report in Supabase."""
        try:
            await self.supabase.table("reports").insert(report).execute()
            
        except Exception as e:
            logger.error(
                "Error storing report",
                extra={
                    "run_id": run_id,
                    "error": str(e)
                }
            )
            raise 