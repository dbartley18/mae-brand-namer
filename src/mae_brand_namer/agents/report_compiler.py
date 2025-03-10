"""Report Compiler for generating comprehensive brand name analysis reports."""

# Standard library imports
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import tempfile

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

    async def compile_report(
        self, run_id: str, state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compile a comprehensive brand name analysis report.

        Args:
            run_id (str): The unique identifier for the workflow run
            state (Optional[Dict[str, Any]]): Optional state data (not used internally)

        Returns:
            Dict[str, Any]: The compiled report data

        Raises:
            ValueError: If required data is missing
        """
        # Fetch brand context
        brand_contexts = await self._fetch_analysis("brand_context", run_id)
        if not brand_contexts:
            raise ValueError("No brand context found for run_id")
        brand_context = brand_contexts[0]  # Use first context

        # Fetch name generations
        name_generations = await self._fetch_analysis("brand_name_generation", run_id)
        if not name_generations:
            raise ValueError("No brand names generated for run_id")

        # Get list of brand names
        brand_names = [ng["brand_name"] for ng in name_generations]

        # Fetch all analyses
        evaluations = await self._fetch_analysis("brand_name_evaluation", run_id)
        competitor_analyses = await self._fetch_analysis("competitor_analysis", run_id)
        cultural_analyses = await self._fetch_analysis("cultural_sensitivity_analysis", run_id)
        domain_analyses = await self._fetch_analysis("domain_analysis", run_id)
        market_research = await self._fetch_analysis("market_research", run_id)
        semantic_analyses = await self._fetch_analysis("semantic_analysis", run_id)
        seo_analyses = await self._fetch_analysis("seo_online_discoverability", run_id)
        survey_simulations = await self._fetch_analysis("survey_simulation", run_id)

        # Organize brand analyses
        brand_analyses = self._organize_brand_analyses(
            brand_names=brand_names,
            brand_name_generations=name_generations,
            brand_name_evaluations=evaluations,
            competitor_analyses=competitor_analyses,
            cultural_sensitivity_analyses=cultural_analyses,
            domain_analyses=domain_analyses,
            market_research_analyses=market_research,
            semantic_analyses=semantic_analyses,
            seo_analyses=seo_analyses,
            survey_simulations=survey_simulations,
        )

        # Prepare state data for report generation
        state_data = {
            # Brand context
            "brand_identity_brief": brand_context.get("brand_identity_brief"),
            "brand_mission": brand_context.get("brand_mission"),
            "brand_personality": brand_context.get("brand_personality"),
            "brand_promise": brand_context.get("brand_promise"),
            "brand_purpose": brand_context.get("brand_purpose"),
            "brand_tone_of_voice": brand_context.get("brand_tone_of_voice"),
            "brand_values": brand_context.get("brand_values"),
            "competitive_landscape": brand_context.get("competitive_landscape"),
            "customer_needs": brand_context.get("customer_needs"),
            "industry_focus": brand_context.get("industry_focus"),
            "industry_trends": brand_context.get("industry_trends"),
            "market_positioning": brand_context.get("market_positioning"),
            "target_audience": brand_context.get("target_audience"),
            # Brand analyses organized by name
            "brand_analyses": brand_analyses,
            # Metadata
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_names_analyzed": len(brand_names),
            "shortlisted_names": [
                name
                for name in brand_names
                if brand_analyses[name]["evaluation"].get("shortlist_status")
            ],
        }

        # Format messages for LLM
        # Create the messages using the compilation prompt
        human_message = HumanMessage(content=self.compilation_prompt.format(**state_data))
        system_message = SystemMessage(content=self.system_prompt.format())
        messages = [system_message, human_message]
        
        # Generate report using LLM
        response = await self.llm.ainvoke(messages)
        
        # Format report sections
        formatted_report = await self._format_report_sections(response, state_data)

        # Generate document
        doc_path = await self._generate_document(formatted_report, run_id)

        # Store report and get URL
        report_url = await self._store_report(run_id, doc_path, formatted_report)

        # Return final report data
        return {
            "state_data": state_data,
            "report_content": formatted_report,
            "report_url": report_url,
            "metadata": {
                "run_id": run_id,
                "timestamp": state_data["timestamp"],
                "total_names": state_data["total_names_analyzed"],
                "shortlisted_names": len(state_data["shortlisted_names"]),
            },
        }

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
        """Store the report document in Supabase Storage and metadata in the database."""
        try:
            # Upload document to Supabase Storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_path = f"reports/{run_id}/{timestamp}_report.docx"

            with open(doc_path, "rb") as f:
                await self.supabase.storage().from_("reports").upload(
                    storage_path,
                    f.read(),
                    {
                        "content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    },
                )

            # Generate public URL
            report_url = f"{settings.supabase_url}/storage/v1/object/public/reports/{storage_path}"

            # Calculate file size
            file_size = os.path.getsize(doc_path) // 1024  # Size in KB

            # Store metadata in report_compilation table using execute_with_retry
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
                }
            )

            # Clean up temporary file
            os.unlink(doc_path)

            return report_url

        except Exception as e:
            logger.error(
                "Error storing report",
                extra={"run_id": run_id, "error_type": type(e).__name__, "error_message": str(e)},
            )
            if os.path.exists(doc_path):
                os.unlink(doc_path)
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
            # Use execute_with_retry instead of building and executing the query directly
            response = await self.supabase.execute_with_retry(
                operation="select",
                table=analysis_type,
                data={
                    "select": "*",
                    "run_id": f"eq.{run_id}"
                }
            )
            
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

    async def _fetch_workflow_data(self, run_id: str) -> Dict[str, Any]:
        """Fetch workflow data from Supabase."""
        try:
            # Use execute_with_retry instead of building and executing the query directly
            result = await self.supabase.execute_with_retry(
                operation="select",
                table="workflow_runs",
                data={
                    "select": "*",
                    "run_id": f"eq.{run_id}"
                }
            )
            
            return result[0] if result else {}
        except APIError as e:
            logger.error(
                "Error fetching workflow data",
                extra={"run_id": run_id, "error_type": type(e).__name__, "error_message": str(e)},
            )
            return {}

    async def _format_report_sections(
        self, response: AIMessage, state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format report sections with proper structure and styling."""
        formatted_report = {
            "metadata": {
                "run_id": state_data["run_id"],
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
            },
            "sections": [],
        }

        # Parse the response content from AIMessage
        try:
            # Parse the response from the LLM
            report_data = self.output_parser.parse(response.content)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Provide a minimal report on parsing failure
            report_data = {
                "executive_summary": {"summary": "Error generating report. Please try again."},
                "recommendations": {"summary": "No recommendations available due to report generation error."}
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
            "recommendations",
        ]

        for section in section_order:
            section_data = report_data.get(section, {})
            if not section_data:
                continue

            formatted_section = {
                "title": section.replace("_", " ").title(),
                "content": self._format_section_content(
                    section, section_data, state_data.get(section, {})
                ),
            }

            formatted_report["sections"].append(formatted_section)

        return formatted_report

    def _format_section_content(
        self, section_type: str, section_data: Dict[str, Any], raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format section content with appropriate styling and structure."""
        formatted_content = {"summary": section_data.get("summary", ""), "details": []}

        # Format based on section type
        if section_type == "survey_simulation":
            # Format survey results as a table
            formatted_content["table"] = self._format_survey_results_table(raw_data)
        elif section_type in ["competitor_analysis", "domain_analysis"]:
            # Format as bullet points with subsections
            formatted_content["bullet_points"] = self._format_bullet_points(section_data)
        else:
            # Standard formatting
            formatted_content["details"] = self._format_details(section_data)

        return formatted_content

    def _format_survey_results_table(self, survey_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format survey simulation results as a table."""
        return {
            "headers": ["Persona", "Company", "Role", "Brand Score", "Key Feedback"],
            "rows": [
                {
                    "persona": response.get("persona_name", ""),
                    "company": response.get("company_name", ""),
                    "role": response.get("role", ""),
                    "score": response.get("brand_score", 0),
                    "feedback": response.get("key_feedback", ""),
                }
                for response in survey_data.get("responses", [])
            ],
        }

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
                                "development_interests": survey.get(
                                    "professional_development_interests"
                                ),
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
                                "channel_preferences": survey.get(
                                    "channel_preferences_brand_interaction"
                                ),
                                "adoption_barriers": survey.get("barriers_to_adoption"),
                            },
                        },
                        "feedback": {
                            "brand_promise_score": survey.get("brand_promise_perception_score"),
                            "personality_fit": survey.get("personality_fit_score"),
                            "emotional_association": survey.get("emotional_association"),
                            "differentiation_score": survey.get(
                                "competitive_differentiation_score"
                            ),
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
                    "brand_personality_alignment": generation_lookup.get(brand_name, {}).get(
                        "brand_personality_alignment"
                    ),
                    "brand_promise_alignment": generation_lookup.get(brand_name, {}).get(
                        "brand_promise_alignment"
                    ),
                    "memorability_score": generation_lookup.get(brand_name, {}).get(
                        "memorability_score"
                    ),
                    "pronounceability_score": generation_lookup.get(brand_name, {}).get(
                        "pronounceability_score"
                    ),
                    "brand_fit_score": generation_lookup.get(brand_name, {}).get("brand_fit_score"),
                    "meaningfulness_score": generation_lookup.get(brand_name, {}).get(
                        "meaningfulness_score"
                    ),
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
