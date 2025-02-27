"""Report Compilation Expert for generating comprehensive brand naming reports."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import markdown
from pathlib import Path
import asyncio

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class ReportCompilationExpert:
    """Expert in compiling and formatting comprehensive brand naming reports."""
    
    def __init__(self, supabase: SupabaseManager = None):
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
        self.supabase = supabase or SupabaseManager()
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if settings.langchain_tracing_v2:
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

    async def compile_report(self, run_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile a comprehensive report from all workflow data.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            state (Dict[str, Any]): Current workflow state
            
        Returns:
            Dict[str, Any]: Report data and metadata
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
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
            
            # Generate recommendations using LLM - this should be made async too
            recommendations = await self._generate_recommendations(report_data)
            report_data["recommendations"].update(recommendations)
            
            # Generate markdown content
            markdown_content = self._generate_markdown(report_data, run_id)
            
            # Store in Supabase
            metadata = await self._store_in_supabase(run_id, report_data, markdown_content)
            
            return {
                "report_data": report_data,
                "metadata": metadata,
                "markdown_content": markdown_content
            }
            
        except Exception as e:
            error_msg = f"Error compiling report: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    async def _generate_recommendations(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations using LLM, implemented as an async method."""
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Create a prompt for the LLM
            prompt = self.prompt.format_messages(
                format_instructions=self.output_parser.get_format_instructions(),
                run_id=report_data.get("executive_summary", {}).get("run_id", ""),
                brand_context=json.dumps(report_data.get("brand_context", {}), indent=2),
                brand_names=json.dumps(report_data.get("brand_name_generation", {}).get("generated_names", []), indent=2),
                analysis_results=json.dumps(report_data.get("analysis", {}), indent=2)
            )
            
            # Call the LLM with the formatted prompt
            with tracing_enabled(tags={"task": "generate_recommendations"}):
                response = await self.llm.ainvoke(prompt)
                
            # Parse the structured response
            try:
                recommendations = self.output_parser.parse(response.content)
            except Exception as e:
                logger.warning(f"Error parsing LLM response: {str(e)}")
                # Fallback to basic extraction if parsing fails
                recommendations = {
                    "final_brand_recommendation": response.content[:500],
                    "brand_name_evaluation": "See final recommendation",
                    "next_steps": "Review the generated brand names and analysis",
                    "additional_considerations": "Consider trademark and domain availability"
                }
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            # Return fallback recommendations if the LLM call fails
            return {
                "final_brand_recommendation": "Unable to generate specific recommendations due to an error.",
                "brand_name_evaluation": "Please review the analysis results to evaluate brand names.",
                "next_steps": "1. Review brand name analysis data\n2. Consider additional branding elements\n3. Conduct trademark searches",
                "additional_considerations": "Consider domain availability, SEO potential, and international implications."
            }

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

    async def _store_in_supabase(self, run_id: str, report_data: Dict[str, Any], markdown_content: str) -> Dict[str, Any]:
        """
        Store the report data in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            report_data (Dict[str, Any]): Structured report data
            markdown_content (str): Markdown formatted report content
            
        Returns:
            Dict[str, Any]: Report metadata including URL and version
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            now = datetime.now().isoformat()
            file_size = len(markdown_content.encode('utf-8')) // 1024  # Size in KB
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a proper URL with timestamp for uniqueness
            report_path = f"reports/{run_id}/{timestamp}_report.md"
            report_url = f"{settings.supabase_url}/storage/v1/object/public/{report_path}"
            
            # Upload the markdown file to Supabase Storage
            try:
                # Create a bucket for reports if it doesn't exist
                try:
                    await self.supabase.storage().create_bucket("reports")
                except Exception as bucket_error:
                    # Bucket likely already exists
                    logger.debug(f"Bucket creation note: {str(bucket_error)}")
                    
                # Upload the file to Supabase Storage
                encoded_content = markdown_content.encode('utf-8')
                await self.supabase.storage().from_("reports").upload(
                    f"{run_id}/{timestamp}_report.md",
                    encoded_content,
                    {"content-type": "text/markdown"}
                )
                
                logger.info(f"Report uploaded to Supabase Storage: {report_path}")
                
            except Exception as storage_error:
                logger.warning(f"Error uploading to Storage: {str(storage_error)}")
                # Fallback to database-only storage if Storage upload fails
                report_url = f"database-only://{run_id}"
            
            # Prepare report metadata
            report_metadata = {
                "run_id": run_id,
                "report_url": report_url,
                "version": 1,
                "created_at": now,
                "last_updated": now,
                "format": "markdown",
                "file_size_kb": file_size,
                "notes": "Report generated with complete LLM recommendations",
                "report_data": report_data,
                "markdown_content": markdown_content
            }
            
            # Insert into report_compilation table
            await self.supabase.table("report_compilation").insert(report_metadata).execute()
            
            return report_metadata
            
        except Exception as e:
            error_msg = f"Error storing report in Supabase: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) 