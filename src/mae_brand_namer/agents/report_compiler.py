"""Report Compiler for generating comprehensive brand name analysis reports."""

# Standard library imports
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import re
import json
import asyncio
import uuid

# Import S3 libraries
try:
    import boto3
    from botocore.client import Config
except ImportError:
    boto3 = None
    Config = None

# Third-party imports
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from postgrest import APIError

# Local application imports
from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)


class ReportCompiler:
    """Expert in compiling and formatting comprehensive brand naming reports."""

    def __init__(
        self,
        dependencies: Optional[Dependencies] = None,
        supabase: Optional[SupabaseManager] = None,
    ):
        """Initialize the ReportCompiler with dependencies."""
        # Initialize Supabase client
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None

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
                ResponseSchema(
                    name="executive_summary",
                    description="High-level overview of project and key findings",
                ),
                ResponseSchema(
                    name="brand_context", description="Detailed brand context and requirements"
                ),
                ResponseSchema(
                    name="name_generation",
                    description="Brand name generation process and methodology",
                ),
                ResponseSchema(
                    name="linguistic_analysis", description="Linguistic analysis findings"
                ),
                ResponseSchema(name="semantic_analysis", description="Semantic analysis findings"),
                ResponseSchema(
                    name="cultural_sensitivity",
                    description="Cultural sensitivity analysis findings",
                ),
                ResponseSchema(
                    name="name_evaluation", description="Evaluation and shortlisting of names"
                ),
                ResponseSchema(
                    name="domain_analysis", description="Domain availability and strategy findings"
                ),
                ResponseSchema(
                    name="seo_analysis", description="SEO and online discovery findings"
                ),
                ResponseSchema(
                    name="competitor_analysis", description="Competitive landscape analysis"
                ),
                ResponseSchema(
                    name="survey_simulation",
                    description="Market research survey simulation results",
                ),
                ResponseSchema(
                    name="recommendations", description="Strategic recommendations and next steps"
                ),
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)

            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=0.3,
                google_api_key=settings.gemini_api_key,
                convert_system_message_to_human=True,
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 40
                },
            )
            logger.info("Successfully initialized ChatGoogleGenerativeAI for Report Compiler")

            # Set up the prompt template
            system_message = SystemMessage(content=self.system_prompt.format())
            human_template = self.compilation_prompt.template
            self.prompt = ChatPromptTemplate.from_messages(
                [system_message, HumanMessage(content=human_template)]
            )
            logger.info("Successfully set up prompt template")

        except Exception as e:
            logger.error(
                "Error initializing ReportCompiler",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
            raise

    async def _log_batch_process(
        self, 
        run_id: str, 
        batch_id: str, 
        batch_sections: List[str], 
        status: str, 
        start_time: datetime, 
        end_time: Optional[datetime] = None,
        error_message: Optional[str] = None,
        section_token_counts: Optional[Dict[str, int]] = None,
        retry_count: int = 0
    ) -> None:
        """
        Log batch processing status to the process_logs table.
        
        Args:
            run_id: Unique identifier for the workflow run
            batch_id: Identifier for this specific batch (e.g., "batch_1_executive_summary")
            batch_sections: List of section names in this batch
            status: Current status (started, completed, failed)
            start_time: When this batch processing started
            end_time: When this batch processing ended (None if not completed)
            error_message: Error details if status is 'failed'
            section_token_counts: Token usage per section if available
            retry_count: Number of retries for this batch
        """
        if end_time is None and status in ("completed", "failed"):
            end_time = datetime.now(timezone.utc)
            
        duration_sec = None
        if start_time and end_time:
            duration_sec = (end_time - start_time).total_seconds()
        
        # Calculate completion percentage based on sections processed
        all_sections = set()
        for sections in self.report_sections:
            all_sections.update(sections)
        completion_percentage = (len(batch_sections) / len(all_sections)) * 100
        
        # Prepare section status JSON
        section_status = {}
        for section in batch_sections:
            section_status[section] = "completed" if status == "completed" else "pending"
            
        try:
            # Insert or update log entry
            query = """
                INSERT INTO process_logs (
                    run_id, agent_type, task_name, status, start_time, end_time,
                    duration_sec, error_message, retry_count, last_retry_at,
                    retry_status, batch_id, batch_sections, section_token_counts,
                    completion_percentage, section_status
                )
                VALUES (
                    :run_id, 'ReportCompiler', :task_name, :status, :start_time, :end_time,
                    :duration_sec, :error_message, :retry_count, :last_retry_at,
                    :retry_status, :batch_id, :batch_sections, :section_token_counts,
                    :completion_percentage, :section_status
                )
                ON CONFLICT (run_id, agent_type, task_name, batch_id) 
                DO UPDATE SET
                    status = EXCLUDED.status,
                    end_time = EXCLUDED.end_time,
                    duration_sec = EXCLUDED.duration_sec,
                    error_message = EXCLUDED.error_message,
                    last_updated = now(),
                    retry_count = CASE 
                        WHEN EXCLUDED.retry_count > process_logs.retry_count 
                        THEN EXCLUDED.retry_count 
                        ELSE process_logs.retry_count 
                    END,
                    last_retry_at = CASE 
                        WHEN EXCLUDED.retry_count > process_logs.retry_count 
                        THEN EXCLUDED.last_retry_at 
                        ELSE process_logs.last_retry_at 
                    END,
                    retry_status = EXCLUDED.retry_status,
                    section_token_counts = EXCLUDED.section_token_counts,
                    completion_percentage = EXCLUDED.completion_percentage,
                    section_status = EXCLUDED.section_status
            """
            
            # Convert section arrays to proper format for database
            batch_sections_str = "{" + ",".join(batch_sections) + "}"
            
            params = {
                "run_id": run_id,
                "task_name": f"report_compilation_{batch_id}",
                "status": status,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration_sec,
                "error_message": error_message,
                "retry_count": retry_count,
                "last_retry_at": datetime.now(timezone.utc) if retry_count > 0 else None,
                "retry_status": "retrying" if retry_count > 0 and status != "completed" else "completed",
                "batch_id": batch_id,
                "batch_sections": batch_sections_str,
                "section_token_counts": json.dumps(section_token_counts) if section_token_counts else None,
                "completion_percentage": completion_percentage,
                "section_status": json.dumps(section_status)
            }
            
            if self.supabase:
                await self.supabase.execute_query(query, params)
                
        except Exception as e:
            logger.error(
                "Error logging batch process",
                extra={
                    "error": str(e),
                    "run_id": run_id,
                    "batch_id": batch_id
                }
            )

    async def _log_overall_process(
        self, 
        run_id: str, 
        status: str, 
        start_time: datetime,
        end_time: Optional[datetime] = None,
        error_message: Optional[str] = None,
        output_size_kb: Optional[float] = None,
        token_count: Optional[int] = None
    ) -> None:
        """
        Log overall report compilation process status.
        
        Args:
            run_id: Unique identifier for the workflow run
            status: Current status (started, completed, failed)
            start_time: When the overall process started
            end_time: When the overall process ended
            error_message: Error details if status is 'failed'
            output_size_kb: Size of the generated report in KB
            token_count: Total token count for the report
        """
        if end_time is None and status in ("completed", "failed"):
            end_time = datetime.now(timezone.utc)
            
        duration_sec = None
        if start_time and end_time:
            duration_sec = (end_time - start_time).total_seconds()
            
        try:
            # Insert or update log entry for overall process
            query = """
                INSERT INTO process_logs (
                    run_id, agent_type, task_name, status, start_time, end_time,
                    duration_sec, error_message, output_size_kb, token_count
                )
                VALUES (
                    :run_id, 'ReportCompiler', 'report_compilation', :status, :start_time, :end_time,
                    :duration_sec, :error_message, :output_size_kb, :token_count
                )
                ON CONFLICT (run_id, agent_type, task_name) 
                WHERE batch_id IS NULL
                DO UPDATE SET
                    status = EXCLUDED.status,
                    end_time = EXCLUDED.end_time,
                    duration_sec = EXCLUDED.duration_sec,
                    error_message = EXCLUDED.error_message,
                    last_updated = now(),
                    output_size_kb = EXCLUDED.output_size_kb,
                    token_count = EXCLUDED.token_count
            """
            
            params = {
                "run_id": run_id,
                "status": status,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration_sec,
                "error_message": error_message,
                "output_size_kb": output_size_kb,
                "token_count": token_count
            }
            
            if self.supabase:
                await self.supabase.execute_query(query, params)
                
        except Exception as e:
            logger.error(
                "Error logging overall process",
                extra={
                    "error": str(e),
                    "run_id": run_id
                }
            )

    async def compile_report(
        self, run_id: str, state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compile a comprehensive brand name analysis report using sequential section generation.
        
        Args:
            run_id (str): The unique identifier for the workflow run
            state (Optional[Dict[str, Any]]): Only needed for the run_id
            
        Returns:
            Dict[str, Any]: The updated workflow state with report URL and metadata
            
        Raises:
            ValueError: If required data is missing
        """
        # Start timing and logging
        process_start_time = datetime.now(timezone.utc)
        await self._log_overall_process(run_id, "started", process_start_time)
        
        try:
            # Only initialize with necessary data - don't load everything into state
            # We'll query Supabase directly for each section as needed
            
            # Create a minimal report structure to build on
            # This will be filled section by section
            report = {
                "metadata": {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "version": 1
                },
                "sections": []
            }
            
            # Define sections to generate sequentially
            report_sections = [
                "executive_summary",
                "brand_context",
                "name_generation",
                "linguistic_analysis",
                "semantic_analysis",
                "cultural_sensitivity",
                "name_evaluation",
                "domain_analysis", 
                "seo_analysis", 
                "competitor_analysis",
                "survey_simulation",
                "translation_analysis",
                "recommendations"
            ]
            
            # Process each section sequentially - one at a time
            total_token_count = 0
            
            # Setup for prompt templates, only load them once
            system_message = SystemMessage(content=self.system_prompt.format())
            
            for section_name in report_sections:
                section_start_time = datetime.now(timezone.utc)
                
                # Log section processing start
                await self._log_overall_process(
                    run_id=run_id,
                    status=f"processing_section_{section_name}",
                    start_time=section_start_time
                )
                
                logger.info(f"Generating section: {section_name}", extra={"run_id": run_id, "section": section_name})
                
                try:
                    # 1. Query relevant data for this section directly from Supabase
                    section_data = await self._fetch_section_data(run_id, section_name)
                    
                    # 2. Generate this specific section using LLM
                    section_content = await self._generate_section(run_id, section_name, section_data, system_message)
                    
                    # 3. Add to report incrementally
                    if section_content:
                        report["sections"].append({
                            "title": section_name.replace("_", " ").title(),
                            "content": section_content
                        })
                        
                        # Track token count for reporting
                        section_token_count = len(str(section_content).split()) // 4 * 3  # Rough estimate
                        total_token_count += section_token_count
                        
                        logger.info(
                            f"Completed section: {section_name}", 
                            extra={
                                "run_id": run_id, 
                                "section": section_name,
                                "token_count": section_token_count
                            }
                        )
                    else:
                        logger.warning(f"Empty content for section: {section_name}", extra={"run_id": run_id})
                        report["sections"].append({
                            "title": section_name.replace("_", " ").title(),
                            "content": {"summary": f"Error generating {section_name} content."}
                        })
                
                except Exception as e:
                    logger.error(
                        f"Error generating section: {section_name}", 
                        extra={
                            "run_id": run_id, 
                            "section": section_name,
                            "error": str(e)
                        }
                    )
                    # Add an error placeholder but continue with other sections
                    report["sections"].append({
                        "title": section_name.replace("_", " ").title(),
                        "content": {"summary": f"Error generating {section_name} content: {str(e)}"}
                    })
            
            # Once all sections are generated, format the report
            formatted_report = self._format_report_sections(report)
            
            # Generate document
            doc_path = await self._generate_document(formatted_report, run_id)
            
            # Store report and get URL
            report_url = await self._store_report(run_id, doc_path, formatted_report)
            
            # Calculate report size in KB
            try:
                output_size_kb = os.path.getsize(doc_path) / 1024
            except Exception:
                output_size_kb = 0
            
            # Create the minimal state to return
            result = {
                "report": {
                    "url": report_url,
                    "run_id": run_id,
                    "version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "format": "pdf",
                    "file_size_kb": output_size_kb,
                    "token_count": total_token_count,
                    "notes": f"Generated with sequential approach, {total_token_count} tokens"
                },
                "run_id": run_id,
                "report_url": report_url,
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "format": "docx",
                "file_size_kb": output_size_kb,
                "notes": f"Generated with sequential approach, {total_token_count} tokens"
            }
            
            # Log successful completion of the entire process
            process_end_time = datetime.now(timezone.utc)
            await self._log_overall_process(
                run_id=run_id,
                status="completed",
                start_time=process_start_time,
                end_time=process_end_time,
                output_size_kb=output_size_kb,
                token_count=total_token_count
            )
            
            return result
            
        except Exception as e:
            # Log failure of the entire process
            error_message = f"Report compilation failed: {str(e)}"
            logger.error(
                error_message,
                extra={
                    "error_type": type(e).__name__,
                    "run_id": run_id
                }
            )
            await self._log_overall_process(
                run_id=run_id,
                status="failed",
                start_time=process_start_time,
                error_message=error_message
            )
            raise
    
    async def _fetch_section_data(self, run_id: str, section_name: str) -> Dict[str, Any]:
        """
        Fetch data needed for a specific report section directly from Supabase.
        
        Args:
            run_id: The unique identifier for the workflow run
            section_name: The name of the section to generate
            
        Returns:
            Dict containing the data needed for this section
        """
        section_data = {
            "run_id": run_id,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        try:
            # Fetch data based on section type
            if section_name == "executive_summary":
                # For executive summary, we need basic brand context and overall metrics
                brand_contexts = await self._fetch_analysis("brand_context", run_id)
                if brand_contexts and len(brand_contexts) > 0:
                    section_data["brand_context"] = brand_contexts[0]
                
                # Get name generation count
                name_generations = await self._fetch_analysis("brand_name_generation", run_id)
                section_data["total_names_generated"] = len(name_generations)
                
                # Get shortlisted names count
                evaluations = await self._fetch_analysis("brand_name_evaluation", run_id)
                shortlisted = [e for e in evaluations if e.get("shortlist_status") is True]
                section_data["shortlisted_names_count"] = len(shortlisted)
                section_data["shortlisted_names"] = [e.get("brand_name") for e in shortlisted]
                
            elif section_name == "brand_context":
                # Fetch brand context directly from the brand_context table
                brand_contexts = await self._fetch_analysis("brand_context", run_id)
                if brand_contexts and len(brand_contexts) > 0:
                    # Make sure we extract the fields according to the schema
                    context = brand_contexts[0]
                    section_data["brand_context"] = {
                        "brand_identity_brief": context.get("brand_identity_brief"),
                        "brand_mission": context.get("brand_mission"),
                        "brand_personality": context.get("brand_personality", []),
                        "brand_promise": context.get("brand_promise"),
                        "brand_purpose": context.get("brand_purpose"),
                        "brand_tone_of_voice": context.get("brand_tone_of_voice"),
                        "brand_values": context.get("brand_values", []),
                        "competitive_landscape": context.get("competitive_landscape"),
                        "customer_needs": context.get("customer_needs", []),
                        "industry_focus": context.get("industry_focus"),
                        "industry_trends": context.get("industry_trends", []),
                        "market_positioning": context.get("market_positioning"),
                        "target_audience": context.get("target_audience")
                    }
                
            elif section_name == "name_generation":
                # Fetch name generations from brand_name_generation table
                generations = await self._fetch_analysis("brand_name_generation", run_id)
                
                # Structure the data according to schema
                section_data["name_generations"] = [{
                    "brand_name": gen.get("brand_name"),
                    "brand_personality_alignment": gen.get("brand_personality_alignment"),
                    "brand_promise_alignment": gen.get("brand_promise_alignment"),
                    "market_differentiation": gen.get("market_differentiation"),
                    "memorability_score": gen.get("memorability_score"),
                    "naming_category": gen.get("naming_category"),
                    "pronounceability_score": gen.get("pronounceability_score"),
                    "target_audience_relevance": gen.get("target_audience_relevance"),
                    "visual_branding_potential": gen.get("visual_branding_potential")
                } for gen in generations]
                
            elif section_name == "linguistic_analysis":
                # Fetch name generations for the brand names list
                name_generations = await self._fetch_analysis("brand_name_generation", run_id)
                section_data["brand_names"] = [ng.get("brand_name") for ng in name_generations]
                
                # Fetch linguistic analyses
                linguistic_analyses = await self._fetch_analysis("linguistic_analysis", run_id)
                
                # Structure according to schema
                section_data["linguistic_analyses"] = [{
                    "brand_name": la.get("brand_name"),
                    "pronunciation_ease": la.get("pronunciation_ease"),
                    "euphony_vs_cacophony": la.get("euphony_vs_cacophony"),
                    "rhythm_and_meter": la.get("rhythm_and_meter"),
                    "sound_symbolism": la.get("sound_symbolism"),
                    "word_class": la.get("word_class"),
                    "overall_readability_score": la.get("overall_readability_score"),
                    "ease_of_marketing_integration": la.get("ease_of_marketing_integration")
                } for la in linguistic_analyses]
                
            elif section_name == "semantic_analysis":
                # Fetch semantic analyses
                semantic_analyses = await self._fetch_analysis("semantic_analysis", run_id)
                
                # Structure according to schema
                section_data["semantic_analyses"] = [{
                    "brand_name": sa.get("brand_name"),
                    "denotative_meaning": sa.get("denotative_meaning"),
                    "etymology": sa.get("etymology"),
                    "descriptiveness": sa.get("descriptiveness"),
                    "emotional_valence": sa.get("emotional_valence"),
                    "brand_fit_relevance": sa.get("brand_fit_relevance"),
                    "sensory_associations": sa.get("sensory_associations"),
                    "memorability_score": sa.get("memorability_score"),
                    "clarity_understandability": sa.get("clarity_understandability"),
                    "uniqueness_differentiation": sa.get("uniqueness_differentiation")
                } for sa in semantic_analyses]
                
            elif section_name == "cultural_sensitivity":
                # Fetch cultural sensitivity analyses
                cultural_analyses = await self._fetch_analysis("cultural_sensitivity_analysis", run_id)
                
                # Structure according to schema
                section_data["cultural_analyses"] = [{
                    "brand_name": ca.get("brand_name"),
                    "cultural_connotations": ca.get("cultural_connotations"),
                    "symbolic_meanings": ca.get("symbolic_meanings"),
                    "religious_sensitivities": ca.get("religious_sensitivities"),
                    "social_political_taboos": ca.get("social_political_taboos"),
                    "regional_variations": ca.get("regional_variations"),
                    "historical_meaning": ca.get("historical_meaning"),
                    "overall_risk_rating": ca.get("overall_risk_rating")
                } for ca in cultural_analyses]
                
            elif section_name == "name_evaluation":
                # Fetch name generations for the brand names list
                name_generations = await self._fetch_analysis("brand_name_generation", run_id)
                section_data["brand_names"] = [ng.get("brand_name") for ng in name_generations]
                
                # Fetch evaluations
                evaluations = await self._fetch_analysis("brand_name_evaluation", run_id)
                
                # Structure according to schema
                section_data["evaluations"] = [{
                    "brand_name": ev.get("brand_name"),
                    "strategic_alignment_score": ev.get("strategic_alignment_score"),
                    "distinctiveness_score": ev.get("distinctiveness_score"),
                    "memorability_score": ev.get("memorability_score"),
                    "pronounceability_score": ev.get("pronounceability_score"),
                    "brand_fit_score": ev.get("brand_fit_score"),
                    "meaningfulness_score": ev.get("meaningfulness_score"),
                    "overall_score": ev.get("overall_score"),
                    "shortlist_status": ev.get("shortlist_status"),
                    "rank": ev.get("rank"),
                    "evaluation_comments": ev.get("evaluation_comments")
                } for ev in evaluations]
                
            elif section_name == "domain_analysis":
                # Fetch domain analyses
                domain_analyses = await self._fetch_analysis("domain_analysis", run_id)
                
                # Structure according to schema
                section_data["domain_analyses"] = [{
                    "brand_name": da.get("brand_name"),
                    "domain_exact_match": da.get("domain_exact_match"),
                    "alternative_tlds": da.get("alternative_tlds"),
                    "domain_history_reputation": da.get("domain_history_reputation"),
                    "social_media_availability": da.get("social_media_availability"),
                    "seo_keyword_relevance": da.get("seo_keyword_relevance"),
                    "domain_length_readability": da.get("domain_length_readability"),
                    "acquisition_cost": da.get("acquisition_cost"),
                    "scalability_future_proofing": da.get("scalability_future_proofing")
                } for da in domain_analyses]
                
            elif section_name == "seo_analysis":
                # Fetch SEO analyses
                seo_analyses = await self._fetch_analysis("seo_online_discoverability", run_id)
                
                # Structure according to schema
                section_data["seo_analyses"] = [{
                    "brand_name": sa.get("brand_name"),
                    "keyword_alignment": sa.get("keyword_alignment"),
                    "search_volume": sa.get("search_volume"),
                    "branded_keyword_potential": sa.get("branded_keyword_potential"),
                    "non_branded_keyword_potential": sa.get("non_branded_keyword_potential"),
                    "content_marketing_opportunities": sa.get("content_marketing_opportunities"),
                    "social_media_discoverability": sa.get("social_media_discoverability"),
                    "seo_viability_score": sa.get("seo_viability_score"),
                    "seo_recommendations": sa.get("seo_recommendations")
                } for sa in seo_analyses]
                
            elif section_name == "competitor_analysis":
                # Fetch competitor analyses
                competitor_analyses = await self._fetch_analysis("competitor_analysis", run_id)
                
                # Structure according to schema
                section_data["competitor_analyses"] = [{
                    "brand_name": ca.get("brand_name"),
                    "competitor_name": ca.get("competitor_name"),
                    "competitor_naming_style": ca.get("competitor_naming_style"),
                    "competitor_positioning": ca.get("competitor_positioning"),
                    "competitor_differentiation_opportunity": ca.get("competitor_differentiation_opportunity"),
                    "risk_of_confusion": ca.get("risk_of_confusion"),
                    "target_audience_perception": ca.get("target_audience_perception"),
                    "competitive_advantage_notes": ca.get("competitive_advantage_notes"),
                    "trademark_conflict_risk": ca.get("trademark_conflict_risk"),
                    "differentiation_score": ca.get("differentiation_score")
                } for ca in competitor_analyses]
                
            elif section_name == "survey_simulation":
                # Fetch survey simulations
                survey_simulations = await self._fetch_analysis("survey_simulation", run_id)
                
                # Structure according to schema - focusing on most important fields to avoid overwhelming
                section_data["survey_simulations"] = [{
                    "brand_name": ss.get("brand_name"),
                    "persona_segment": ss.get("persona_segment"),
                    "company_name": ss.get("company_name"),
                    "job_title": ss.get("job_title"),
                    "department": ss.get("department"),
                    "personality_fit_score": ss.get("personality_fit_score"),
                    "emotional_association": ss.get("emotional_association"),
                    "competitive_differentiation_score": ss.get("competitive_differentiation_score"),
                    "simulated_market_adoption_score": ss.get("simulated_market_adoption_score"),
                    "qualitative_feedback_summary": ss.get("qualitative_feedback_summary"),
                    "final_survey_recommendation": ss.get("final_survey_recommendation"),
                    "strategic_ranking": ss.get("strategic_ranking")
                } for ss in survey_simulations]
                
            elif section_name == "translation_analysis":
                # Fetch translation analyses
                translation_analyses = await self._fetch_analysis("translation_analysis", run_id)
                
                # Structure according to schema
                section_data["translation_analyses"] = [{
                    "brand_name": ta.get("brand_name"),
                    "target_language": ta.get("target_language"),
                    "direct_translation": ta.get("direct_translation"),
                    "pronunciation_difficulty": ta.get("pronunciation_difficulty"),
                    "semantic_shift": ta.get("semantic_shift"),
                    "cultural_acceptability": ta.get("cultural_acceptability"),
                    "adaptation_needed": ta.get("adaptation_needed"),
                    "proposed_adaptation": ta.get("proposed_adaptation"),
                    "brand_essence_preserved": ta.get("brand_essence_preserved")
                } for ta in translation_analyses]
                
            elif section_name == "recommendations":
                # For recommendations, we need evaluations and other key analyses
                evaluations = await self._fetch_analysis("brand_name_evaluation", run_id)
                domain_analyses = await self._fetch_analysis("domain_analysis", run_id)
                seo_analyses = await self._fetch_analysis("seo_online_discoverability", run_id)
                
                # Structure evaluations by score
                shortlisted = [e for e in evaluations if e.get("shortlist_status") is True]
                section_data["shortlisted_names"] = [{
                    "brand_name": e.get("brand_name"),
                    "overall_score": e.get("overall_score"),
                    "rank": e.get("rank")
                } for e in shortlisted]
                
                # Get domain info for shortlisted names
                shortlisted_domains = []
                for name in [e.get("brand_name") for e in shortlisted]:
                    domain_info = next((da for da in domain_analyses if da.get("brand_name") == name), None)
                    if domain_info:
                        shortlisted_domains.append({
                            "brand_name": name,
                            "domain_exact_match": domain_info.get("domain_exact_match"),
                            "alternative_tlds": domain_info.get("alternative_tlds"),
                            "acquisition_cost": domain_info.get("acquisition_cost")
                        })
                
                section_data["shortlisted_domains"] = shortlisted_domains
                
                # Get SEO info for shortlisted names
                shortlisted_seo = []
                for name in [e.get("brand_name") for e in shortlisted]:
                    seo_info = next((sa for sa in seo_analyses if sa.get("brand_name") == name), None)
                    if seo_info:
                        shortlisted_seo.append({
                            "brand_name": name,
                            "seo_viability_score": seo_info.get("seo_viability_score"),
                            "seo_recommendations": seo_info.get("seo_recommendations")
                        })
                
                section_data["shortlisted_seo"] = shortlisted_seo
                
        except Exception as e:
            logger.error(
                f"Error fetching data for section: {section_name}",
                extra={
                    "error": str(e),
                    "run_id": run_id
                }
            )
            # Return empty data but don't fail - the section generator will handle missing data
            
        return section_data
    
    async def _generate_section(
        self, 
        run_id: str, 
        section_name: str, 
        section_data: Dict[str, Any],
        system_message: SystemMessage
    ) -> Dict[str, Any]:
        """
        Generate a single section of the report using the LLM.
        
        Args:
            run_id: The unique identifier for the workflow run
            section_name: The name of the section to generate
            section_data: Data needed for this section
            system_message: The system message to use
            
        Returns:
            Dict containing the formatted section content
        """
        # Create section-specific prompt
        section_title = section_name.replace("_", " ").title()
        section_instructions = f"Focus specifically on generating the '{section_title}' section of the report."
        section_instructions += f"\nOnly generate content for the '{section_title}' section, not the entire report."
        
        # Determine which schema we need for this section
        section_schema = None
        for schema in self.output_schemas:
            if schema.name == section_name:
                section_schema = schema
                break
                
        if not section_schema:
            logger.warning(f"No schema found for section: {section_name}")
            # Create a simple schema
            section_schema = ResponseSchema(
                name=section_name,
                description=f"{section_title} content"
            )
        
        # Create a specific output parser for just this section
        section_output_parser = StructuredOutputParser.from_response_schemas([section_schema])
        
        # Create human message with focused instructions
        human_message = HumanMessage(content=f"""
        Generate the {section_title} section for a brand naming report using this data:
        
        {section_data}
        
        {section_instructions}
        
        Format your response according to this schema:
        {section_output_parser.get_format_instructions()}
        """)
        
        # Create LLM with appropriate parameters
        temp_llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.gemini_api_key,
            convert_system_message_to_human=True,
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40
            },
        )
        
        # Generate this section with retries
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Generate content for this section only
                messages = [system_message, human_message]
                response = await temp_llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Remove any non-ASCII characters
                content = re.sub(r'[^\x00-\x7F]+', ' ', content)
                
                # Parse the section content
                section_content = section_output_parser.parse(content)
                
                # Success - return the section content
                return section_content.get(section_name, {})
                
            except Exception as e:
                logger.error(
                    f"Error generating content for section {section_name}, attempt {retry_count+1}/{max_retries}",
                    extra={
                        "error": str(e),
                        "run_id": run_id,
                    }
                )
                retry_count += 1
                
                # If all retries failed
                if retry_count >= max_retries:
                    logger.error(f"Failed to generate section {section_name} after {max_retries} attempts")
                    return {"summary": f"Error generating {section_name} content after multiple attempts."}
                
                # Wait before retrying
                await asyncio.sleep(1)
        
        # Fallback content if all else fails
        return {"summary": f"Error generating {section_name} content."}

    def _format_report_sections(
        self, report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format report sections with proper structure and styling."""
        formatted_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
            },
            "sections": [],
        }
        
        # Order sections according to workflow
        section_order = [
            "executive_summary",
            "brand_context",
            "name_generation",
            "linguistic_analysis",
            "semantic_analysis",
            "cultural_sensitivity",
            "name_evaluation",
            "domain_analysis",
            "seo_analysis", 
            "competitor_analysis",
            "survey_simulation",
            "translation_analysis",
            "recommendations",
        ]
        
        for section in section_order:
            section_data = report_data.get(section, {})
            if not section_data:
                continue
            
            formatted_section = {
                "title": section.replace("_", " ").title(),
                "content": self._format_section_content(section, section_data),
            }
            
            formatted_report["sections"].append(formatted_section)
            
        return formatted_report
        
    def _format_section_content(
        self, section_type: str, section_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format section content with appropriate styling and structure."""
        formatted_content = {"summary": section_data.get("summary", ""), "details": []}
        
        # Format based on section type
        if section_type == "survey_simulation":
            # For survey simulation, we need to structure the table data
            formatted_content["table"] = {
                "headers": ["Persona", "Company", "Role", "Brand Score", "Key Feedback"],
                "rows": []  # Will be filled in _format_survey_results_table
            }
        elif section_type == "competitor_analysis":
            # Format competitor analysis as structured data
            formatted_content["summary"] = section_data.get("summary", "")
            
            # Create a more structured format for competitor analysis
            # First, try to extract any table data if present
            table_data = {}
            for key, value in section_data.items():
                if "table" in key.lower() or "comparison" in key.lower():
                    table_data[key] = value
            
            # Then extract key points as bullet points
            bullet_points = []
            key_competitor_aspects = [
                "Competitor Naming Styles", 
                "Market Positioning",
                "Differentiation Opportunities",
                "Risk of Confusion",
                "Target Audience Perception",
                "Competitive Advantages",
                "Trademark Considerations"
            ]
            
            for aspect in key_competitor_aspects:
                if aspect in section_data:
                    bullet_points.append({
                        "heading": aspect,
                        "points": [section_data.get(aspect, "")]
                    })
            
            formatted_content["bullet_points"] = bullet_points
            formatted_content["tables"] = table_data
            
        elif section_type == "domain_analysis":
            # Format as bullet points with subsections
            formatted_content["bullet_points"] = self._format_bullet_points(section_data)
        else:
            # Standard formatting
            formatted_content["details"] = self._format_details(section_data)
            
        return formatted_content

    async def _generate_document(self, report: Dict[str, Any], run_id: str) -> str:
        """Generate a Word document from the report data."""
        doc = Document()

        # Set up document styles
        styles = doc.styles

        # Title style
        title_style = styles.add_style("CustomTitle", WD_STYLE_TYPE.PARAGRAPH)
        title_style.font.size = Pt(24)
        title_style.font.bold = True

        # Heading 1 style
        h1_style = styles.add_style("CustomH1", WD_STYLE_TYPE.PARAGRAPH)
        h1_style.font.size = Pt(16)
        h1_style.font.bold = True

        # Heading 2 style
        h2_style = styles.add_style("CustomH2", WD_STYLE_TYPE.PARAGRAPH)
        h2_style.font.size = Pt(14)
        h2_style.font.bold = True

        # Normal text style
        normal_style = styles.add_style("CustomNormal", WD_STYLE_TYPE.PARAGRAPH)
        normal_style.font.size = Pt(11)

        # Italic style for notes
        note_style = styles.add_style("CustomNote", WD_STYLE_TYPE.PARAGRAPH)
        note_style.font.size = Pt(11)
        note_style.font.italic = True

        # Add title
        title = doc.add_paragraph("Brand Naming Report", style="CustomTitle")
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Add Mae branding and context
        doc.add_paragraph("Generated by Mae Brand Naming Expert", style="CustomNormal")
        doc.add_paragraph(
            f"Date: {datetime.now().strftime('%B %d, %Y %H:%M:%S %Z')}", style="CustomNormal"
        )
        doc.add_paragraph(f"Run ID: {run_id}", style="CustomNormal")
        doc.add_paragraph()

        # Add simulation context note
        note_heading = doc.add_paragraph("Important Note", style="CustomH2")
        note = doc.add_paragraph(
            "This report was generated through the Mae Brand Naming Expert simulation. "
            "The only input provided was the initial user prompt:",
            style="CustomNote",
        )
        doc.add_paragraph(
            f'"{report.get("metadata", {}).get("user_prompt", "")}"', style="CustomNote"
        )
        doc.add_paragraph(
            "All additional context, analysis, and insights were autonomously generated by the simulation.",
            style="CustomNote",
        )

        # Add horizontal line
        doc.add_paragraph("_" * 50, style="CustomNormal")
        doc.add_paragraph()

        # Add table of contents
        toc_heading = doc.add_paragraph("Table of Contents", style="CustomH1")
        for i, section in enumerate(report["sections"], 1):
            toc_entry = doc.add_paragraph(style="CustomNormal")
            toc_entry.add_run(f"{i}. {section['title']}")
        doc.add_paragraph()  # Add space after TOC

        # Add sections
        for section in report["sections"]:
            # Add section title
            doc.add_paragraph(section["title"], style="CustomH1")

            # Add section content
            content = section["content"]

            # Add summary if present
            if content.get("summary"):
                doc.add_paragraph(content["summary"], style="CustomNormal")

            # Add table if present
            if content.get("table"):
                table_data = content["table"]
                table = doc.add_table(rows=1, cols=len(table_data["headers"]))
                table.style = "Table Grid"

                # Add headers
                header_cells = table.rows[0].cells
                for i, header in enumerate(table_data["headers"]):
                    header_cells[i].text = header

                # Add rows
                for row_data in table_data["rows"]:
                    row_cells = table.add_row().cells
                    for i, value in enumerate(row_data.values()):
                        row_cells[i].text = str(value)

            # Add bullet points if present
            if content.get("bullet_points"):
                for bullet_section in content["bullet_points"]:
                    doc.add_paragraph(bullet_section["heading"], style="CustomH2")
                    for point in bullet_section["points"]:
                        doc.add_paragraph(point, style="CustomNormal").style = "List Bullet"

            # Add details if present
            if content.get("details"):
                for detail in content["details"]:
                    doc.add_paragraph(detail["heading"], style="CustomH2")
                    doc.add_paragraph(detail["content"], style="CustomNormal")

            doc.add_paragraph()  # Add spacing between sections

        # Save document to temporary file
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            doc.save(tmp.name)
            return tmp.name

    async def _store_report(self, run_id: str, doc_path: str, report_data: Dict[str, Any]) -> str:
        """Store the report document using Supabase Storage standard upload method.
        
        Args:
            run_id: Unique identifier for the report run
            doc_path: Path to the local document file
            report_data: Report data dictionary
            
        Returns:
            str: URL to the stored report
        """
        # Define timestamp outside try block so it's available in except block
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Define object_key here so it's available in all blocks
        object_key = f"{run_id}/{timestamp}_report.docx"
        
        try:
            # Get storage settings
            bucket_name = settings.s3_bucket  # Using the same bucket name defined in settings
            
            file_size = os.path.getsize(doc_path) // 1024  # Size in KB
            
            logger.info(
                "Storing report using Supabase Storage standard upload",
                extra={
                    "run_id": run_id,
                    "file_size_kb": file_size,
                    "bucket": bucket_name,
                    "object_key": object_key
                }
            )
            
            # Read the file content
            with open(doc_path, "rb") as file:
                file_content = file.read()
            
            # Set content type for the document
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            # Use Supabase client to upload the file
            if self.supabase:
                # Get Supabase client directly
                supabase_client = self.supabase.client
                
                # Upload file using standard upload method - not async in Supabase client
                try:
                    logger.debug("Starting file upload using standard Supabase upload")
                    result = supabase_client.storage.from_(bucket_name).upload(
                        path=object_key,
                        file=file_content,
                        file_options={"content_type": content_type}
                    )
                    
                    logger.debug(f"Upload result: {result}")
                    
                    if hasattr(result, 'error') and result.error:
                        raise Exception(f"Supabase upload returned error: {result.error}")
                        
                except Exception as upload_error:
                    logger.error(
                        f"Supabase upload error",
                        extra={
                            "error": str(upload_error),
                            "bucket": bucket_name,
                            "object_key": object_key
                        }
                    )
                    raise Exception(f"Supabase upload error: {str(upload_error)}")
                
                # Try to get the public URL directly from Supabase client first
                try:
                    report_url = supabase_client.storage.from_(bucket_name).get_public_url(object_key)
                    logger.info(f"Generated public URL from Supabase client: {report_url}")
                except Exception as url_error:
                    logger.warning(
                        f"Could not get URL from Supabase client, falling back to constructed URL",
                        extra={"error": str(url_error)}
                    )
                    # Fall back to constructing the URL manually
                    base_supabase_url = settings.supabase_url
                    if base_supabase_url.endswith('/storage/v1/s3'):
                        base_supabase_url = base_supabase_url[:-13]  # Remove '/storage/v1/s3'
                    elif base_supabase_url.endswith('/storage/v1'):
                        base_supabase_url = base_supabase_url[:-10]  # Remove '/storage/v1'
                        
                    public_url_base = f"{base_supabase_url}/storage/v1/object/public"
                    report_url = f"{public_url_base}/{bucket_name}/{object_key}"
                    logger.info(f"Using constructed public URL: {report_url}")
                
                logger.info(
                    "File uploaded successfully using standard upload, storing metadata",
                    extra={
                        "run_id": run_id,
                        "report_url": report_url
                    }
                )
                
                # Store metadata in report_compilation table
                await self.supabase.execute_with_retry(
                    operation="insert",
                    table="report_compilation",
                    data={
                        "run_id": run_id,
                        "report_url": report_url,
                        "version": 1,
                        "created_at": datetime.now().isoformat(),
                        "format": "docx",
                        "file_size_kb": file_size,
                        "notes": "Comprehensive brand naming analysis report",
                        "upload_protocol": "standard",
                        "storage_bucket": bucket_name,
                        "storage_key": object_key
                    }
                )
                
                # Clean up temporary file
                os.unlink(doc_path)
                
                logger.info(
                    "Report stored successfully using Supabase Storage standard upload",
                    extra={
                        "run_id": run_id, 
                        "report_url": report_url,
                        "file_size_kb": file_size,
                        "bucket": bucket_name
                    }
                )
                
                return report_url
            else:
                raise ValueError("Supabase client not initialized")
                
        except Exception as e:
            logger.error(
                "Error storing report via Supabase Storage standard upload",
                extra={
                    "run_id": run_id, 
                    "error_type": type(e).__name__, 
                    "error_message": str(e),
                    "bucket": settings.s3_bucket
                }
            )
            
            # If we failed to store, keep the local file as a backup
            local_backup_path = f"./reports/{object_key}"
            try:
                import shutil
                os.makedirs(os.path.dirname(local_backup_path), exist_ok=True)
                shutil.copy(doc_path, local_backup_path)
                logger.info(f"Created local backup at {local_backup_path}")
                # Still attempt to clean up the temp file
                os.unlink(doc_path)
                
                # Return a file:// URL as fallback
                absolute_path = os.path.abspath(local_backup_path)
                return f"file://{absolute_path}"
                
            except Exception as backup_error:
                logger.error(f"Failed to create local backup: {str(backup_error)}")
            
            # Raise the original error
            raise

    async def _fetch_analysis(self, analysis_type: str, run_id: str) -> List[Dict[str, Any]]:
        """
        Fetch analysis data from the database for a specific analysis type and run ID.
        
        Args:
            analysis_type (str): The type of analysis to fetch
            run_id (str): The run ID to filter by
            
        Returns:
            List[Dict[str, Any]]: List of analysis results
            
        Raises:
            ValueError: If required fields are missing from the analysis results
        """
        try:
            # The execute_with_retry method applies the .eq() filter automatically
            # when given a key-value pair in the data dictionary
            response = await self.supabase.execute_with_retry(
                operation="select",
                table=analysis_type,
                data={
                    "select": "*",
                    "run_id": run_id  # Will apply .eq(run_id, value) internally
                }
            )
            
            # response is already the data array from the response.data property
            if not response:
                return []
            
            # Validate required fields based on analysis type
            for row in response:
                if analysis_type == "brand_context":
                    required_fields = [
                        "brand_promise",
                        "brand_mission",
                        "brand_values",
                        "brand_personality",
                        "brand_tone_of_voice",
                        "target_audience",
                        "customer_needs",
                        "market_positioning",
                        "brand_identity_brief",
                        "industry_focus",
                        "industry_trends",
                        "competitive_landscape",
                    ]
                elif analysis_type == "brand_name_generation":
                    required_fields = [
                        "brand_name",
                        "naming_category",
                        "brand_personality_alignment",
                        "brand_promise_alignment",
                        "memorability_score",
                        "pronounceability_score",
                        "brand_fit_score",
                        "meaningfulness_score",
                    ]
                elif analysis_type == "brand_name_evaluation":
                    required_fields = [
                        "brand_name",
                        "strategic_alignment_score",
                        "distinctiveness_score",
                        "memorability_score",
                        "pronounceability_score",
                        "meaningfulness_score",
                        "brand_fit_score",
                        "domain_viability_score",
                        "overall_score",
                        "shortlist_status",
                        "evaluation_comments",
                        "rank",
                    ]
                elif analysis_type == "competitor_analysis":
                    required_fields = [
                        "brand_name",
                        "competitor_name",
                        "competitor_positioning",
                        "competitor_strengths",
                        "competitor_weaknesses",
                        "differentiation_score",
                        "risk_of_confusion",
                        "competitive_advantage_notes",
                    ]
                elif analysis_type == "cultural_sensitivity_analysis":
                    required_fields = [
                        "brand_name",
                        "cultural_connotations",
                        "symbolic_meanings",
                        "overall_risk_rating",
                        "notes",
                    ]
                elif analysis_type == "domain_analysis":
                    required_fields = [
                        "brand_name",
                        "domain_exact_match",
                        "alternative_tlds",
                        "acquisition_cost",
                        "notes",
                    ]
                elif analysis_type == "market_research":
                    required_fields = [
                        "brand_name",
                        "market_opportunity",
                        "target_audience_fit",
                        "market_viability",
                        "potential_risks",
                    ]
                elif analysis_type == "semantic_analysis":
                    required_fields = [
                        "brand_name",
                        "denotative_meaning",
                        "etymology",
                        "descriptiveness",
                        "concreteness",
                        "emotional_valence",
                        "brand_personality",
                        "sensory_associations",
                        "figurative_language",
                        "ambiguity",
                        "irony_or_paradox",
                        "humor_playfulness",
                        "phoneme_combinations",
                        "sound_symbolism",
                        "rhyme_rhythm",
                        "alliteration_assonance",
                        "word_length_syllables",
                        "compounding_derivation",
                        "brand_name_type",
                        "memorability_score",
                        "original_pronunciation_ease",
                        "clarity_understandability",
                        "uniqueness_differentiation",
                        "brand_fit_relevance",
                        "semantic_trademark_risk",
                    ]
                elif analysis_type == "seo_online_discoverability":
                    required_fields = [
                        "brand_name",
                        "keyword_alignment",
                        "search_volume",
                        "seo_viability_score",
                        "seo_recommendations",
                    ]
                elif analysis_type == "survey_simulation":
                    required_fields = [
                        "brand_name",
                        "persona_segment",
                        "brand_promise_perception_score",
                        "personality_fit_score",
                        "emotional_association",
                        "competitive_differentiation_score",
                        "psychometric_sentiment_mapping",
                        "competitor_benchmarking_score",
                        "simulated_market_adoption_score",
                        "qualitative_feedback_summary",
                        "raw_qualitative_feedback",
                        "final_survey_recommendation",
                        "strategic_ranking",
                        "industry",
                        "company_size_employees",
                        "company_revenue",
                        "market_share",
                        "company_structure",
                        "geographic_location",
                        "technology_stack",
                        "company_growth_stage",
                        "job_title",
                        "seniority",
                        "years_of_experience",
                        "department",
                        "education_level",
                        "goals_and_challenges",
                        "values_and_priorities",
                        "decision_making_style",
                        "information_sources",
                        "pain_points",
                        "technological_literacy",
                        "attitude_towards_risk",
                        "purchasing_behavior",
                        "online_behavior",
                        "interaction_with_brand",
                        "professional_associations",
                        "technical_skills",
                        "networking_habits",
                        "professional_aspirations",
                        "influence_within_company",
                        "event_attendance",
                        "content_consumption_habits",
                        "vendor_relationship_preferences",
                        "business_chemistry",
                        "reports_to",
                        "buying_group_structure",
                        "decision_maker",
                        "company_focus",
                        "company_maturity",
                        "budget_authority",
                        "preferred_vendor_size",
                        "innovation_adoption",
                        "key_performance_indicators",
                        "professional_development_interests",
                        "social_media_usage",
                        "work_life_balance_priorities",
                        "frustrations_annoyances",
                        "personal_aspirations_life_goals",
                        "motivations",
                        "current_brand_relationships",
                        "product_adoption_lifecycle_stage",
                        "purchase_triggers_events",
                        "success_metrics_product_service",
                        "channel_preferences_brand_interaction",
                        "barriers_to_adoption",
                        "generation_age_range",
                        "company_culture_values",
                        "industry_sub_vertical",
                        "confidence_score_persona_accuracy",
                        "data_sources_persona_creation",
                        "persona_archetype_type",
                        "company_name",
                    ]
                else:
                    required_fields = []

                # Add run_id to required fields for all analysis types
                required_fields.append("run_id")

                # Check for missing fields but don't raise error if field is None
                missing_fields = [field for field in required_fields if field not in row]
                if missing_fields:
                    logger.warning(
                        f"Missing fields in {analysis_type} analysis",
                        extra={
                            "analysis_type": analysis_type,
                            "missing_fields": missing_fields,
                            "run_id": run_id
                        }
                    )

            return response
        except Exception as e:
            logger.error(
                f"Error fetching {analysis_type} analysis",
                extra={
                    "analysis_type": analysis_type,
                    "run_id": run_id,
                    "error": str(e)
                }
            )
            raise

    def _format_bullet_points(self, section_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format section data as bullet points."""
        bullet_points = []
        for key, value in section_data.items():
            if isinstance(value, list):
                bullet_points.append({"heading": key.replace("_", " ").title(), "points": value})
            elif isinstance(value, dict):
                bullet_points.append(
                    {
                        "heading": key.replace("_", " ").title(),
                        "points": [f"{k}: {v}" for k, v in value.items()],
                    }
                )
        return bullet_points

    def _format_details(self, section_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format section details with proper structure."""
        details = []
        for key, value in section_data.items():
            if key == "summary":
                continue
            details.append({"heading": key.replace("_", " ").title(), "content": value})
        return details

    def _group_by_field(
        self, data: List[Dict[str, Any]], field: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group a list of dictionaries by a specified field."""
        groups = {}
        for item in data:
            if item[field] not in groups:
                groups[item[field]] = []
            groups[item[field]].append(item)
        return groups

    def _organize_brand_analyses(
        self,
        brand_names: List[str],
        brand_name_generations: List[Dict[str, Any]],
        brand_name_evaluations: List[Dict[str, Any]],
        competitor_analyses: List[Dict[str, Any]],
        cultural_sensitivity_analyses: List[Dict[str, Any]],
        domain_analyses: List[Dict[str, Any]],
        market_research_analyses: List[Dict[str, Any]],
        semantic_analyses: List[Dict[str, Any]],
        seo_analyses: List[Dict[str, Any]],
        survey_simulations: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Organize all brand-specific analyses by brand name.

        Args:
            brand_names: List of brand names to organize data for
            brand_name_generations: List of brand name generation results
            brand_name_evaluations: List of brand name evaluation results
            competitor_analyses: List of competitor analysis results
            cultural_sensitivity_analyses: List of cultural sensitivity analysis results
            domain_analyses: List of domain analysis results
            market_research_analyses: List of market research results
            semantic_analyses: List of semantic analysis results
            seo_analyses: List of SEO analysis results
            survey_simulations: List of survey simulation results

        Returns:
            Dict mapping brand names to their organized analysis data
        """
        # Create lookups for analyses by brand name
        generation_lookup = {g["brand_name"]: g for g in brand_name_generations}
        evaluation_lookup = {e["brand_name"]: e for e in brand_name_evaluations}
        cultural_lookup = {c["brand_name"]: c for c in cultural_sensitivity_analyses}
        domain_lookup = {d["brand_name"]: d for d in domain_analyses}
        market_lookup = {m["brand_name"]: m for m in market_research_analyses}
        semantic_lookup = {s["brand_name"]: s for s in semantic_analyses}
        seo_lookup = {s["brand_name"]: s for s in seo_analyses}

        # Group competitor analyses by brand name
        competitor_lookup = {}
        for comp in competitor_analyses:
            brand_name = comp.get("brand_name")
            if brand_name:
                if brand_name not in competitor_lookup:
                    competitor_lookup[brand_name] = []
                competitor_lookup[brand_name].append(
                    {
                        "competitor_name": comp.get("competitor_name"),
                        "positioning": comp.get("competitor_positioning"),
                        "strengths": comp.get("competitor_strengths"),
                        "weaknesses": comp.get("competitor_weaknesses"),
                        "differentiation_score": comp.get("differentiation_score"),
                        "risk_of_confusion": comp.get("risk_of_confusion"),
                        "competitive_advantage_notes": comp.get("competitive_advantage_notes"),
                    }
                )

        # Group survey results by brand name
        survey_lookup = {}
        for survey in survey_simulations:
            brand_name = survey.get("brand_name")
            if brand_name:
                if brand_name not in survey_lookup:
                    survey_lookup[brand_name] = []
                survey_lookup[brand_name].append(
                    {
                        "persona": {
                            "segment": survey.get("persona_segment"),
                            "industry": survey.get("industry"),
                            "company": {
                                "name": survey.get("company_name"),
                                "size": survey.get("company_size_employees"),
                                "revenue": survey.get("company_revenue"),
                                "market_share": survey.get("market_share"),
                                "structure": survey.get("company_structure"),
                                "location": survey.get("geographic_location"),
                                "tech_stack": survey.get("technology_stack"),
                                "growth_stage": survey.get("company_growth_stage"),
                                "focus": survey.get("company_focus"),
                                "maturity": survey.get("company_maturity"),
                                "culture": survey.get("company_culture_values"),
                            },
                            "role": {
                                "title": survey.get("job_title"),
                                "seniority": survey.get("seniority"),
                                "experience": survey.get("years_of_experience"),
                                "department": survey.get("department"),
                                "education": survey.get("education_level"),
                                "reports_to": survey.get("reports_to"),
                            },
                            "profile": {
                                "goals_challenges": survey.get("goals_and_challenges"),
                                "values_priorities": survey.get("values_and_priorities"),
                                "decision_style": survey.get("decision_making_style"),
                                "info_sources": survey.get("information_sources"),
                                "pain_points": survey.get("pain_points"),
                                "tech_literacy": survey.get("technological_literacy"),
                                "risk_attitude": survey.get("attitude_towards_risk"),
                                "purchasing_behavior": survey.get("purchasing_behavior"),
                                "online_behavior": survey.get("online_behavior"),
                                "brand_interaction": survey.get("interaction_with_brand"),
                                "associations": survey.get("professional_associations"),
                                "skills": survey.get("technical_skills"),
                                "networking": survey.get("networking_habits"),
                                "aspirations": survey.get("professional_aspirations"),
                                "influence": survey.get("influence_within_company"),
                                "events": survey.get("event_attendance"),
                                "content_habits": survey.get("content_consumption_habits"),
                                "vendor_preferences": survey.get("vendor_relationship_preferences"),
                                "business_chemistry": survey.get("business_chemistry"),
                                "budget_authority": survey.get("budget_authority"),
                                "vendor_size_preference": survey.get("preferred_vendor_size"),
                                "innovation_adoption": survey.get("innovation_adoption"),
                                "kpis": survey.get("key_performance_indicators"),
                                "development_interests": survey.get("professional_development_interests"),
                                "social_media": survey.get("social_media_usage"),
                                "work_life_balance": survey.get("work_life_balance_priorities"),
                                "frustrations": survey.get("frustrations_annoyances"),
                                "life_goals": survey.get("personal_aspirations_life_goals"),
                                "motivations": survey.get("motivations"),
                            },
                            "buying_process": {
                                "group_structure": survey.get("buying_group_structure"),
                                "decision_maker": survey.get("decision_maker"),
                                "brand_relationships": survey.get("current_brand_relationships"),
                                "adoption_stage": survey.get("product_adoption_lifecycle_stage"),
                                "triggers": survey.get("purchase_triggers_events"),
                                "success_metrics": survey.get("success_metrics_product_service"),
                                "channel_preferences": survey.get("channel_preferences_brand_interaction"),
                                "adoption_barriers": survey.get("barriers_to_adoption"),
                            },
                        },
                        "feedback": {
                            "brand_promise_score": survey.get("brand_promise_perception_score"),
                            "personality_fit": survey.get("personality_fit_score"),
                            "emotional_association": survey.get("emotional_association"),
                            "differentiation_score": survey.get("competitive_differentiation_score"),
                            "sentiment_mapping": survey.get("psychometric_sentiment_mapping"),
                            "competitor_benchmark": survey.get("competitor_benchmarking_score"),
                            "market_adoption": survey.get("simulated_market_adoption_score"),
                            "qualitative_summary": survey.get("qualitative_feedback_summary"),
                            "raw_feedback": survey.get("raw_qualitative_feedback"),
                            "recommendation": survey.get("final_survey_recommendation"),
                            "strategic_ranking": survey.get("strategic_ranking"),
                        },
                        "metadata": {
                            "confidence_score": survey.get("confidence_score_persona_accuracy"),
                            "data_sources": survey.get("data_sources_persona_creation"),
                            "archetype": survey.get("persona_archetype_type"),
                            "generation": survey.get("generation_age_range"),
                            "industry_vertical": survey.get("industry_sub_vertical"),
                        },
                    }
                )

        # Organize data by brand name
        organized_data = {}
        for brand_name in brand_names:
            # Get evaluation data to check shortlist status
            evaluation = evaluation_lookup.get(brand_name, {})

            # Basic brand data from generation and evaluation
            brand_data = {
                "generation": {
                    "brand_name": brand_name,
                    "naming_category": generation_lookup.get(brand_name, {}).get("naming_category"),
                    "brand_personality_alignment": generation_lookup.get(brand_name, {}).get("brand_personality_alignment"),
                    "brand_promise_alignment": generation_lookup.get(brand_name, {}).get("brand_promise_alignment"),
                    "memorability_score": generation_lookup.get(brand_name, {}).get("memorability_score"),
                    "pronounceability_score": generation_lookup.get(brand_name, {}).get("pronounceability_score"),
                    "brand_fit_score": generation_lookup.get(brand_name, {}).get("brand_fit_score"),
                    "meaningfulness_score": generation_lookup.get(brand_name, {}).get("meaningfulness_score"),
                },
                "evaluation": {
                    "strategic_alignment_score": evaluation.get("strategic_alignment_score"),
                    "distinctiveness_score": evaluation.get("distinctiveness_score"),
                    "memorability_score": evaluation.get("memorability_score"),
                    "pronounceability_score": evaluation.get("pronounceability_score"),
                    "meaningfulness_score": evaluation.get("meaningfulness_score"),
                    "brand_fit_score": evaluation.get("brand_fit_score"),
                    "domain_viability_score": evaluation.get("domain_viability_score"),
                    "overall_score": evaluation.get("overall_score"),
                    "shortlist_status": evaluation.get("shortlist_status", False),
                    "evaluation_comments": evaluation.get("evaluation_comments"),
                    "rank": evaluation.get("rank"),
                },
            }

            # Add semantic analysis
            semantic = semantic_lookup.get(brand_name, {})
            brand_data["semantic"] = {
                "meaning": {
                    "denotative": semantic.get("denotative_meaning"),
                    "etymology": semantic.get("etymology"),
                    "descriptiveness": semantic.get("descriptiveness"),
                    "concreteness": semantic.get("concreteness"),
                    "emotional_valence": semantic.get("emotional_valence"),
                },
                "linguistic": {
                    "brand_personality": semantic.get("brand_personality"),
                    "sensory_associations": semantic.get("sensory_associations"),
                    "figurative_language": semantic.get("figurative_language"),
                    "ambiguity": semantic.get("ambiguity"),
                    "irony_paradox": semantic.get("irony_or_paradox"),
                    "humor_playfulness": semantic.get("humor_playfulness"),
                },
                "sound": {
                    "phoneme_combinations": semantic.get("phoneme_combinations"),
                    "sound_symbolism": semantic.get("sound_symbolism"),
                    "rhyme_rhythm": semantic.get("rhyme_rhythm"),
                    "alliteration_assonance": semantic.get("alliteration_assonance"),
                    "word_length_syllables": semantic.get("word_length_syllables"),
                },
                "structure": {
                    "compounding_derivation": semantic.get("compounding_derivation"),
                    "brand_name_type": semantic.get("brand_name_type"),
                },
                "scores": {
                    "memorability": semantic.get("memorability_score"),
                    "pronunciation_ease": semantic.get("original_pronunciation_ease"),
                    "clarity": semantic.get("clarity_understandability"),
                    "uniqueness": semantic.get("uniqueness_differentiation"),
                    "brand_fit": semantic.get("brand_fit_relevance"),
                },
                "trademark_risk": semantic.get("semantic_trademark_risk"),
            }

            # Add cultural sensitivity analysis
            cultural = cultural_lookup.get(brand_name, {})
            brand_data["cultural"] = {
                "connotations": cultural.get("cultural_connotations"),
                "symbolic_meanings": cultural.get("symbolic_meanings"),
                "risk_rating": cultural.get("overall_risk_rating"),
                "notes": cultural.get("notes"),
            }

            # Add domain analysis
            domain = domain_lookup.get(brand_name, {})
            brand_data["domain"] = {
                "exact_match": domain.get("domain_exact_match"),
                "alternative_tlds": domain.get("alternative_tlds"),
                "acquisition_cost": domain.get("acquisition_cost"),
                "notes": domain.get("notes"),
            }

            # Add market research
            market = market_lookup.get(brand_name, {})
            brand_data["market"] = {
                "opportunity": market.get("market_opportunity"),
                "audience_fit": market.get("target_audience_fit"),
                "viability": market.get("market_viability"),
                "risks": market.get("potential_risks"),
            }

            # Add SEO analysis
            seo = seo_lookup.get(brand_name, {})
            brand_data["seo"] = {
                "keyword_alignment": seo.get("keyword_alignment"),
                "search_volume": seo.get("search_volume"),
                "viability_score": seo.get("seo_viability_score"),
                "recommendations": seo.get("seo_recommendations"),
            }

            # Add competitor analyses
            brand_data["competitors"] = competitor_lookup.get(brand_name, [])

            # Add survey results
            brand_data["survey"] = survey_lookup.get(brand_name, [])

            # Extract key metrics for easy access
            brand_data["key_metrics"] = {
                "memorability_score": brand_data["evaluation"]["memorability_score"],
                "distinctiveness_score": brand_data["evaluation"]["distinctiveness_score"],
                "strategic_alignment_score": brand_data["evaluation"]["strategic_alignment_score"],
                "overall_score": brand_data["evaluation"]["overall_score"],
                "cultural_risk_rating": brand_data["cultural"]["risk_rating"],
                "seo_viability_score": brand_data["seo"]["viability_score"],
                "market_viability": brand_data["market"]["viability"],
                "brand_fit_score": brand_data["evaluation"]["brand_fit_score"],
                "pronunciation_ease": brand_data["semantic"]["scores"]["pronunciation_ease"],
                "clarity_score": brand_data["semantic"]["scores"]["clarity"],
            }

            organized_data[brand_name] = brand_data

        return organized_data
