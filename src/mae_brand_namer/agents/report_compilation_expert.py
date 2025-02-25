"""Report Compilation Expert for generating comprehensive brand naming reports."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import markdown
from pathlib import Path

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ReportCompilationExpert:
    """Expert in compiling and formatting comprehensive brand naming reports."""
    
    def __init__(self):
        """Initialize the ReportCompilationExpert with necessary configurations."""
        # Agent identity
        self.role = "Enterprise Report Compilation & Formatting Specialist"
        self.goal = """Aggregate and synthesize all brand naming analyses into a structured, enterprise-grade report, 
        ensuring clarity, strategic insights, and executive-ready presentation."""
        self.backstory = """You are an expert in report compilation, structured business communication, and data storytelling. 
        Drawing from methodologies outlined in Alina Wheeler's 'Designing Brand Identity', Barbara Minto's 'The Pyramid Principle', 
        and Cole Nussbaumer Knaflic's 'Storytelling with Data', you transform raw analytical insights into a highly professional, 
        logically structured report."""
        
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(settings.supabase_url, settings.supabase_key)
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if settings.tracing_enabled:
            self.tracer = LangChainTracer(project_name=settings.langsmith_project)
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.tracer] if self.tracer else None
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="executive_summary", description="Concise summary of findings and recommendations"),
            ResponseSchema(name="brand_name_rankings", description="Structured table of scores across evaluation criteria"),
            ResponseSchema(name="brand_context_summary", description="Summary of brand context and competitive landscape"),
            ResponseSchema(name="semantic_analysis_summary", description="Key findings from semantic analysis"),
            ResponseSchema(name="linguistic_analysis_summary", description="Key findings from linguistic analysis"),
            ResponseSchema(name="cultural_sensitivity_summary", description="Key findings from cultural sensitivity analysis"),
            ResponseSchema(name="translation_feasibility_summary", description="Key findings from translation analysis"),
            ResponseSchema(name="domain_assessment_summary", description="Key findings from domain analysis"),
            ResponseSchema(name="competitor_benchmarking_summary", description="Key findings from competitor analysis"),
            ResponseSchema(name="seo_discoverability_summary", description="Key findings from SEO analysis"),
            ResponseSchema(name="market_perception_summary", description="Key findings from survey simulation"),
            ResponseSchema(name="strategic_recommendations", description="Final recommendations and next steps"),
            ResponseSchema(name="methodology_summary", description="Overview of the AI-driven evaluation methodology")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Create the prompt template
        system_message = SystemMessage(
            content=f"""You are a Report Compilation Expert with the following profile:
            Role: {self.role}
            Goal: {self.goal}
            Backstory: {self.backstory}
            
            Compile a comprehensive brand naming report following these principles:
            1. Use Minto's Pyramid Principle for executive-friendly structuring
            2. Apply data visualization techniques from 'Storytelling with Data'
            3. Follow enterprise reporting best practices
            4. Ensure insights are actionable and presentation-ready
            
            Format your response according to the following schema:
            {{format_instructions}}
            """,
            additional_kwargs={
                "metadata": {
                    "agent_type": "report_compiler",
                    "methodology": [
                        "Alina Wheeler's Designing Brand Identity",
                        "Barbara Minto's Pyramid Principle",
                        "Cole Nussbaumer Knaflic's Storytelling with Data"
                    ]
                }
            }
        )
        human_template = """Compile a comprehensive brand naming report using the following data:
        Run ID: {run_id}
        Brand Context: {brand_context}
        Brand Names: {brand_names}
        Analysis Results: {analysis_results}
        """
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

    def compile_report(self, run_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile a comprehensive report from all workflow data.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            state (Dict[str, Any]): Current workflow state
            
        Returns:
            Dict[str, Any]: Report data and metadata
        """
        try:
            # Create report data structure
            report_data = {
                "executive_summary": {
                    "user_prompt": state.get("user_prompt", ""),
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_names_generated": len(state.get("generated_names", [])),
                    "shortlisted_names": len(state.get("shortlisted_names", []))
                },
                "brand_context": {
                    "brand_identity_brief": state.get("brand_identity_brief", ""),
                    "brand_promise": state.get("brand_promise", ""),
                    "brand_values": state.get("brand_values", []),
                    "brand_personality": state.get("brand_personality", []),
                    "brand_tone_of_voice": state.get("brand_tone_of_voice", ""),
                    "brand_purpose": state.get("brand_purpose", ""),
                    "brand_mission": state.get("brand_mission", ""),
                    "target_audience": state.get("target_audience", ""),
                    "customer_needs": state.get("customer_needs", []),
                    "market_positioning": state.get("market_positioning", ""),
                    "competitive_landscape": state.get("competitive_landscape", ""),
                    "industry_focus": state.get("industry_focus", ""),
                    "industry_trends": state.get("industry_trends", [])
                },
                "brand_name_generation": {
                    "methodology": state.get("name_generation_methodology", ""),
                    "generated_names": [
                        {
                            "name": name_data["brand_name"],
                            "categories": name_data.get("naming_categories", []),
                            "personality_alignments": name_data.get("brand_personality_alignments", []),
                            "promise_alignments": name_data.get("brand_promise_alignments", []),
                            "target_audience_relevance": name_data.get("target_audience_relevance", 0.0),
                            "market_differentiation": name_data.get("market_differentiation", 0.0),
                            "memorability_scores": name_data.get("memorability_scores", 0.0),
                            "pronounceability_scores": name_data.get("pronounceability_scores", 0.0),
                            "visual_branding_potential": name_data.get("visual_branding_potential", 0.0),
                            "rank": name_data.get("name_rankings", 0.0)
                        }
                        for name_data in state.get("generated_names", [])
                    ]
                },
                "analysis": {
                    "semantic_analysis": state.get("semantic_analysis_results", {}),
                    "linguistic_analysis": state.get("linguistic_analysis_results", {}),
                    "cultural_analysis": state.get("cultural_analysis_results", {}),
                    "translation_analysis": state.get("translation_analysis_results", {}),
                    "competitor_analysis": state.get("competitor_analysis_results", {}),
                    "survey_simulation": state.get("survey_simulation_results", {})
                },
                "evaluation": {
                    "evaluation_results": state.get("evaluation_results", {}),
                    "shortlisted_names": state.get("shortlisted_names", []),
                    "domain_analysis": state.get("domain_analysis_results", {}),
                    "seo_analysis": state.get("seo_analysis_results", {})
                },
                "recommendations": {
                    "final_brand_recommendation": "",  # To be filled by LLM
                    "brand_name_evaluation": "",  # To be filled by LLM
                    "next_steps": "",  # To be filled by LLM
                    "additional_considerations": ""  # To be filled by LLM
                }
            }
            
            # Generate recommendations using LLM
            recommendations = self._generate_recommendations(report_data)
            report_data["recommendations"].update(recommendations)
            
            # Generate markdown content
            markdown_content = self._generate_markdown(report_data, run_id)
            
            # Store in Supabase
            metadata = self._store_in_supabase(run_id, report_data, markdown_content)
            
            return {
                "report_data": report_data,
                "metadata": metadata,
                "markdown_content": markdown_content
            }
            
        except Exception as e:
            error_msg = f"Error compiling report: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _generate_markdown(self, report_data: Dict[str, Any], run_id: str) -> str:
        """
        Generate a Markdown formatted report from the structured data.
        
        Args:
            report_data (Dict[str, Any]): Structured report data
            run_id (str): Unique identifier for this workflow run
            
        Returns:
            str: Markdown formatted report content
        """
        try:
            # Build the report structure
            sections = [
                "# Brand Naming Analysis Report\n",
                f"Run ID: {run_id}\n",
                "## Executive Summary\n",
                f"Brand Purpose: {report_data['executive_summary']['brand_purpose']}\n",
                f"Market Positioning: {report_data['executive_summary']['market_positioning']}\n",
                f"Final Brand Recommendation: {report_data['executive_summary']['final_brand_recommendation']}\n",
                f"Brand Name Evaluation: {report_data['executive_summary']['brand_name_evaluation']}\n",
                
                "## Brand Context\n",
                self._format_dict_section(report_data['brand_context']),
                
                "## Brand Name Generation\n",
                self._format_dict_section(report_data['brand_name_generation']),
                
                "## Brand Name Evaluation\n",
                self._format_dict_section(report_data['brand_name_evaluation']),
                
                "## Analysis Results\n",
                "### Semantic Analysis\n",
                self._format_dict_section(report_data['analysis']['semantic_analysis']),
                
                "### Linguistic Analysis\n",
                self._format_dict_section(report_data['analysis']['linguistic_analysis']),
                
                "### Cultural Sensitivity Analysis\n",
                self._format_dict_section(report_data['analysis']['cultural_analysis']),
                
                "### Translation Analysis\n",
                self._format_dict_section(report_data['analysis']['translation_analysis']),
                
                "### Domain Analysis\n",
                self._format_dict_section(report_data['analysis']['domain_analysis']),
                
                "### SEO Analysis\n",
                self._format_dict_section(report_data['analysis']['seo_analysis']),
                
                "### Competitor Analysis\n",
                "#### Overall Landscape\n",
                self._format_dict_section(report_data['analysis']['competitor_analysis']['overall_landscape']),
                "#### Per-Name Analysis\n",
                self._format_dict_section(report_data['analysis']['competitor_analysis']['per_name_analysis']),
                
                "### Survey Simulation Results\n",
                self._format_dict_section(report_data['analysis']['survey_simulation']),
                
                "## Recommendations\n",
                self._format_dict_section(report_data['recommendations'])
            ]
            
            return "\n".join(sections)
            
        except Exception as e:
            error_msg = f"Error generating Markdown report: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _format_dict_section(self, data: Dict[str, Any]) -> str:
        """Format a dictionary section into Markdown."""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"### {key.replace('_', ' ').title()}\n")
                lines.append(self._format_dict_section(value))
            elif isinstance(value, list):
                lines.append(f"### {key.replace('_', ' ').title()}\n")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._format_dict_section(item))
                    else:
                        lines.append(f"- {item}\n")
            else:
                lines.append(f"### {key.replace('_', ' ').title()}\n{value}\n")
        return "\n".join(lines)

    def _store_in_supabase(self, run_id: str, report_data: Dict[str, Any], markdown_content: str) -> Dict[str, Any]:
        """
        Store the report data in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            report_data (Dict[str, Any]): Structured report data
            markdown_content (str): Markdown formatted report content
            
        Returns:
            Dict[str, Any]: Report metadata including URL and version
        """
        try:
            now = datetime.now().isoformat()
            file_size = len(markdown_content.encode('utf-8')) // 1024  # Size in KB
            
            # Prepare report metadata
            report_metadata = {
                "run_id": run_id,
                "report_url": f"reports/{run_id}/report.md",  # Placeholder URL
                "version": "1.0",
                "created_at": now,
                "last_updated": now,
                "format": "markdown",
                "file_size_kb": file_size,
                "notes": "Initial report generation",
                "report_data": report_data,
                "markdown_content": markdown_content
            }
            
            # Insert into report_compilation table
            self.supabase.table("report_compilation").insert(report_metadata).execute()
            
            return report_metadata
            
        except Exception as e:
            error_msg = f"Error storing report in Supabase: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) 