"""Report Compiler for generating comprehensive brand name analysis reports."""

# Standard library imports
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
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
    
    def __init__(self, dependencies: Optional[Dependencies] = None, supabase: Optional[SupabaseManager] = None):
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
                ResponseSchema(name="executive_summary", description="High-level overview of project and key findings"),
                ResponseSchema(name="brand_context", description="Detailed brand context and requirements"),
                ResponseSchema(name="name_generation", description="Brand name generation process and methodology"),
                ResponseSchema(name="linguistic_analysis", description="Linguistic analysis findings"),
                ResponseSchema(name="semantic_analysis", description="Semantic analysis findings"),
                ResponseSchema(name="cultural_sensitivity", description="Cultural sensitivity analysis findings"),
                ResponseSchema(name="name_evaluation", description="Evaluation and shortlisting of names"),
                ResponseSchema(name="domain_analysis", description="Domain availability and strategy findings"),
                ResponseSchema(name="seo_analysis", description="SEO and online discovery findings"),
                ResponseSchema(name="competitor_analysis", description="Competitive landscape analysis"),
                ResponseSchema(name="survey_simulation", description="Market research survey simulation results"),
                ResponseSchema(name="recommendations", description="Strategic recommendations and next steps")
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=0.3,
                google_api_key=settings.gemini_api_key,
                convert_system_message_to_human=True
            )
            logger.info("Successfully initialized ChatGoogleGenerativeAI for Report Compiler")
            
            # Set up the prompt template
            system_message = SystemMessage(content=self.system_prompt.format())
            human_template = self.compilation_prompt.template
            self.prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessage(content=human_template)
            ])
            logger.info("Successfully set up prompt template")
            
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
            with tracing_v2_enabled():
                # Fetch all analysis results from Supabase
                brand_contexts = await self._fetch_analysis("brand_context", run_id)
                name_generations = await self._fetch_analysis("brand_name_generation", run_id)
                linguistic_analyses = await self._fetch_analysis("linguistic_analysis", run_id)
                semantic_analyses = await self._fetch_analysis("semantic_analysis", run_id)
                cultural_analyses = await self._fetch_analysis("cultural_sensitivity_analysis", run_id)
                name_evaluations = await self._fetch_analysis("brand_name_evaluation", run_id)
                domain_analyses = await self._fetch_analysis("domain_analysis", run_id)
                seo_analyses = await self._fetch_analysis("seo_online_discoverability", run_id)
                competitor_analyses = await self._fetch_analysis("competitor_analysis", run_id)
                survey_simulations = await self._fetch_analysis("survey_simulation", run_id)
                
                # Get the original user prompt from the workflow_runs table
                workflow_data = await self._fetch_workflow_data(run_id)
                user_prompt = workflow_data.get("user_prompt", "")
                
                # Prepare state data for the prompt, organizing lists of analyses
                state_data = {
                    "run_id": run_id,
                    "user_prompt": user_prompt,
                    "brand_context": brand_contexts[0] if brand_contexts else {},  # Should only be one
                    "name_generation": {
                        "all_names": name_generations,
                        "by_category": self._group_by_field(name_generations, "naming_category")
                    },
                    "brand_analyses": self._organize_brand_analyses(
                        brand_names=[ng["brand_name"] for ng in name_generations],
                        linguistic=linguistic_analyses,
                        semantic=semantic_analyses,
                        cultural=cultural_analyses,
                        evaluation=name_evaluations,
                        domain=domain_analyses,
                        seo=seo_analyses
                    ),
                    "competitor_analysis": {
                        "all_analyses": competitor_analyses,
                        "by_brand": self._group_by_field(competitor_analyses, "brand_name")
                    },
                    "survey_simulation": {
                        "all_responses": survey_simulations,
                        "by_brand": self._group_by_field(survey_simulations, "brand_name"),
                        "by_persona": self._group_by_field(survey_simulations, "persona_segment")
                    }
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
                
                # Add metadata including user prompt
                report_data["metadata"] = {
                    "run_id": run_id,
                    "user_prompt": user_prompt,
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    "analysis_counts": {
                        "brand_contexts": len(brand_contexts),
                        "name_generations": len(name_generations),
                        "linguistic_analyses": len(linguistic_analyses),
                        "semantic_analyses": len(semantic_analyses),
                        "cultural_analyses": len(cultural_analyses),
                        "name_evaluations": len(name_evaluations),
                        "domain_analyses": len(domain_analyses),
                        "seo_analyses": len(seo_analyses),
                        "competitor_analyses": len(competitor_analyses),
                        "survey_simulations": len(survey_simulations)
                    }
                }
                
                # Format the report sections
                report = await self._format_report_sections(report_data, state_data)
                
                # Generate document
                doc_path = await self._generate_document(report, run_id)
                
                # Store the report in Supabase
                report_url = await self._store_report(run_id, doc_path, report)
                
                return {
                    "report_data": report,
                    "report_url": report_url
                }
                
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

    async def _generate_document(self, report: Dict[str, Any], run_id: str) -> str:
        """Generate a Word document from the report data."""
        doc = Document()
        
        # Set up document styles
        styles = doc.styles
        
        # Title style
        title_style = styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
        title_style.font.size = Pt(24)
        title_style.font.bold = True
        
        # Heading 1 style
        h1_style = styles.add_style('CustomH1', WD_STYLE_TYPE.PARAGRAPH)
        h1_style.font.size = Pt(16)
        h1_style.font.bold = True
        
        # Heading 2 style
        h2_style = styles.add_style('CustomH2', WD_STYLE_TYPE.PARAGRAPH)
        h2_style.font.size = Pt(14)
        h2_style.font.bold = True
        
        # Normal text style
        normal_style = styles.add_style('CustomNormal', WD_STYLE_TYPE.PARAGRAPH)
        normal_style.font.size = Pt(11)
        
        # Italic style for notes
        note_style = styles.add_style('CustomNote', WD_STYLE_TYPE.PARAGRAPH)
        note_style.font.size = Pt(11)
        note_style.font.italic = True
        
        # Add title
        title = doc.add_paragraph("Brand Naming Report", style='CustomTitle')
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add Mae branding and context
        doc.add_paragraph("Generated by Mae Brand Naming Expert", style='CustomNormal')
        doc.add_paragraph(f"Date: {datetime.now().strftime('%B %d, %Y %H:%M:%S %Z')}", style='CustomNormal')
        doc.add_paragraph(f"Run ID: {run_id}", style='CustomNormal')
        doc.add_paragraph()
        
        # Add simulation context note
        note_heading = doc.add_paragraph("Important Note", style='CustomH2')
        note = doc.add_paragraph(
            "This report was generated through the Mae Brand Naming Expert simulation. "
            "The only input provided was the initial user prompt:", style='CustomNote'
        )
        doc.add_paragraph(f'"{report.get("metadata", {}).get("user_prompt", "")}"', style='CustomNote')
        doc.add_paragraph(
            "All additional context, analysis, and insights were autonomously generated by the simulation.",
            style='CustomNote'
        )
        
        # Add horizontal line
        doc.add_paragraph("_" * 50, style='CustomNormal')
        doc.add_paragraph()
        
        # Add table of contents
        toc_heading = doc.add_paragraph("Table of Contents", style='CustomH1')
        for i, section in enumerate(report["sections"], 1):
            toc_entry = doc.add_paragraph(style='CustomNormal')
            toc_entry.add_run(f"{i}. {section['title']}")
        doc.add_paragraph()  # Add space after TOC
        
        # Add sections
        for section in report["sections"]:
            # Add section title
            doc.add_paragraph(section["title"], style='CustomH1')
            
            # Add section content
            content = section["content"]
            
            # Add summary if present
            if content.get("summary"):
                doc.add_paragraph(content["summary"], style='CustomNormal')
            
            # Add table if present
            if content.get("table"):
                table_data = content["table"]
                table = doc.add_table(rows=1, cols=len(table_data["headers"]))
                table.style = 'Table Grid'
                
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
                    doc.add_paragraph(bullet_section["heading"], style='CustomH2')
                    for point in bullet_section["points"]:
                        doc.add_paragraph(point, style='CustomNormal').style = 'List Bullet'
            
            # Add details if present
            if content.get("details"):
                for detail in content["details"]:
                    doc.add_paragraph(detail["heading"], style='CustomH2')
                    doc.add_paragraph(detail["content"], style='CustomNormal')
            
            doc.add_paragraph()  # Add spacing between sections
        
        # Save document to temporary file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            doc.save(tmp.name)
            return tmp.name

    async def _store_report(
        self,
        run_id: str,
        doc_path: str,
        report_data: Dict[str, Any]
    ) -> str:
        """Store the report document in Supabase Storage and metadata in the database."""
        try:
            # Upload document to Supabase Storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_path = f"reports/{run_id}/{timestamp}_report.docx"
            
            with open(doc_path, 'rb') as f:
                await self.supabase.storage().from_("reports").upload(
                    storage_path,
                    f.read(),
                    {"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
                )
            
            # Generate public URL
            report_url = f"{settings.supabase_url}/storage/v1/object/public/reports/{storage_path}"
            
            # Calculate file size
            file_size = os.path.getsize(doc_path) // 1024  # Size in KB
            
            # Store metadata in report_compilation table
            await self.supabase.table("report_compilation").insert({
                "run_id": run_id,
                "report_url": report_url,
                "version": 1,
                "created_at": datetime.now().isoformat(),
                "format": "docx",
                "file_size_kb": file_size,
                "notes": "Comprehensive brand naming analysis report"
            }).execute()
            
            # Clean up temporary file
            os.unlink(doc_path)
            
            return report_url
            
        except Exception as e:
            logger.error(
                "Error storing report",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            if os.path.exists(doc_path):
                os.unlink(doc_path)
            raise

    async def _fetch_analysis(self, analysis_type: str, run_id: str) -> List[Dict[str, Any]]:
        """
        Fetch ALL analysis results from Supabase for a given run_id.
        
        Args:
            analysis_type (str): The type of analysis (table name) to fetch from
            run_id (str): The unique identifier for this workflow run
            
        Returns:
            List[Dict[str, Any]]: List of all analysis results for this run_id, ordered appropriately
        """
        try:
            logger.info(f"Fetching all {analysis_type} analyses for run_id: {run_id}")
            
            # Define the order based on table type
            order = None
            if analysis_type == "brand_name_generation":
                # Order by naming_category and rank
                order = "naming_category.asc,rank.asc"
            elif analysis_type == "brand_name_evaluation":
                # Order by overall_score (descending) and shortlist_status
                order = "overall_score.desc,shortlist_status.desc"
            elif analysis_type == "survey_simulation":
                # Order by brand_name and strategic_ranking
                order = "brand_name.asc,strategic_ranking.asc"
            elif analysis_type == "competitor_analysis":
                # Order by competitor_name
                order = "competitor_name.asc"
            elif "rank" in analysis_type:
                # For tables with rank field (linguistic, semantic, cultural, domain)
                order = "brand_name.asc,rank.asc"
            else:
                # Default ordering by brand_name if it exists
                order = "brand_name.asc"
            
            # Build and execute query using proper Supabase query builder
            query = self.supabase.table(analysis_type).select("*").eq("run_id", run_id)
            
            # Add ordering if specified
            if order:
                query = query.order(order)
            
            # Execute the query
            response = await query.execute()
            
            if response and response.data:
                logger.info(f"Successfully fetched {len(response.data)} {analysis_type} analyses for run_id: {run_id}")
                return response.data
            else:
                logger.warning(f"No {analysis_type} analyses found for run_id: {run_id}")
                return []
                
        except Exception as e:
            logger.warning(
                f"Error fetching {analysis_type} analyses",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "run_id": run_id,
                    "table": analysis_type
                }
            )
            return []

    async def _fetch_workflow_data(self, run_id: str) -> Dict[str, Any]:
        """
        Fetch workflow data including user prompt from workflow_runs table.
        This should return exactly one record since there's one workflow per run_id.
        """
        try:
            # Build and execute query using proper Supabase query builder
            response = await self.supabase.table("workflow_runs").select("*").eq("run_id", run_id).execute()
            
            if response and response.data and len(response.data) > 0:
                logger.info(f"Successfully fetched workflow data for run_id: {run_id}")
                return response.data[0]  # Return the first (and should be only) record
            else:
                logger.warning(f"No workflow data found for run_id: {run_id}")
                return {}
        except Exception as e:
            logger.warning(
                f"Error fetching workflow data",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "run_id": run_id
                }
            )
            return {}

    async def _format_report_sections(
        self,
        report_data: Dict[str, Any],
        state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format report sections with proper structure and styling."""
        formatted_report = {
            "metadata": {
                "run_id": state_data["run_id"],
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            },
            "sections": []
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
            "recommendations"
        ]
        
        for section in section_order:
            section_data = report_data.get(section, {})
            if not section_data:
                continue
                
            formatted_section = {
                "title": section.replace("_", " ").title(),
                "content": self._format_section_content(section, section_data, state_data.get(section, {}))
            }
            
            formatted_report["sections"].append(formatted_section)
        
        return formatted_report

    def _format_section_content(
        self,
        section_type: str,
        section_data: Dict[str, Any],
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format section content with appropriate styling and structure."""
        formatted_content = {
            "summary": section_data.get("summary", ""),
            "details": []
        }
        
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
            "headers": [
                "Persona",
                "Company",
                "Role",
                "Brand Score",
                "Key Feedback"
            ],
            "rows": [
                {
                    "persona": response.get("persona_name", ""),
                    "company": response.get("company_name", ""),
                    "role": response.get("role", ""),
                    "score": response.get("brand_score", 0),
                    "feedback": response.get("key_feedback", "")
                }
                for response in survey_data.get("responses", [])
            ]
        }

    def _format_bullet_points(self, section_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format section data as bullet points."""
        bullet_points = []
        for key, value in section_data.items():
            if isinstance(value, list):
                bullet_points.append({
                    "heading": key.replace("_", " ").title(),
                    "points": value
                })
            elif isinstance(value, dict):
                bullet_points.append({
                    "heading": key.replace("_", " ").title(),
                    "points": [f"{k}: {v}" for k, v in value.items()]
                })
        return bullet_points

    def _format_details(self, section_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format section details with proper structure."""
        details = []
        for key, value in section_data.items():
            if key == "summary":
                continue
            details.append({
                "heading": key.replace("_", " ").title(),
                "content": value
            })
        return details

    def _group_by_field(self, data: List[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group a list of dictionaries by a specified field."""
        groups = {}
        for item in data:
            if item[field] not in groups:
                groups[item[field]] = []
            groups[item[field]].append(item)
        return groups

    def _organize_brand_analyses(self, brand_names: List[str], linguistic: List[Dict[str, Any]], semantic: List[Dict[str, Any]], cultural: List[Dict[str, Any]], evaluation: List[Dict[str, Any]], domain: List[Dict[str, Any]], seo: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Organize brand analyses by brand name."""
        organized_analyses = {}
        for brand_name in brand_names:
            organized_analyses[brand_name] = {
                "linguistic_analysis": [item for item in linguistic if item["brand_name"] == brand_name],
                "semantic_analysis": [item for item in semantic if item["brand_name"] == brand_name],
                "cultural_sensitivity_analysis": [item for item in cultural if item["brand_name"] == brand_name],
                "brand_name_evaluation": [item for item in evaluation if item["brand_name"] == brand_name],
                "domain_analysis": [item for item in domain if item["brand_name"] == brand_name],
                "seo_online_discoverability": [item for item in seo if item["brand_name"] == brand_name]
            }
        return organized_analyses 