"""
Report Formatter

This module handles the second step of the two-step report generation process.
It pulls raw data from the report_raw_data table and formats it into a polished report.
"""

import os
import json
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Set

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.parser import parse_xml

from src.mae_brand_namer.utils.supabase_utils import SupabaseManager
from src.mae_brand_namer.agents.prompts.report_formatter import (
    get_executive_summary_prompt,
    get_recommendations_prompt,
    get_title_page_prompt,
    get_toc_prompt
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportFormatter:
    """
    Handles formatting and generation of reports using raw data from the report_raw_data table.
    This is the second step in the two-step report generation process.
    """

    # Mapping between DB section names and formatter section names
    SECTION_MAPPING = {
        # DB section name -> Formatter section name
        "brand_context": "brand_context",
        "name_gen": "name_generation",
        "name_semantics": "semantic_analysis",
        "name_linguistics": "linguistic_analysis",
        "cultural_check": "cultural_sensitivity",
        "name_eval": "brand_name_evaluation",
        "translation_check": "translation_analysis",
        "market_analysis": "market_research",
        "competitor_check": "competitor_analysis",
        "domain_check": "domain_analysis",
        "survey_results": "survey_simulation",
        "exec_summary": "executive_summary",
        "final_recommendations": "recommendations"
    }

    # Reverse mapping for convenience
    REVERSE_SECTION_MAPPING = {v: k for k, v in SECTION_MAPPING.items()}

    # Default storage bucket for reports
    STORAGE_BUCKET = "agent_reports"
    
    # Report formats
    FORMAT_DOCX = "docx"

    def __init__(self, supabase_client=None, llm_client=None):
        """Initialize the ReportFormatter with a Supabase connection and optional LLM client."""
        if supabase_client:
            self.supabase = supabase_client
        else:
            # Create a new connection if none was provided
            logger.info("Initializing ReportFormatter with a new Supabase connection")
            self.supabase = SupabaseManager()
            
        self.llm_client = llm_client
        
        # Register data transformers
        self.data_transformers = {
            "brand_context": self._transform_brand_context,
            "name_generation": self._transform_name_generation,
            "semantic_analysis": self._transform_semantic_analysis,
            "linguistic_analysis": self._transform_linguistic_analysis,
            "cultural_sensitivity": self._transform_cultural_sensitivity,
            "brand_name_evaluation": self._transform_name_evaluation,
            "translation_analysis": self._transform_translation_analysis,
            "market_research": self._transform_market_research,
            "competitor_analysis": self._transform_competitor_analysis,
            "domain_analysis": self._transform_domain_analysis,
            "survey_simulation": self._transform_survey_simulation,
        }
        
        # Track formatting errors for report summary
        self.formatting_errors = {}

    def _transform_brand_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform brand context data to the expected format."""
        if not data:
            return {}
            
        if "brand_context" not in data:
            # Wrap existing data in brand_context field if needed
            return {"brand_context": data}
            
        return data
        
    def _transform_name_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform name generation data to the expected format."""
        if not data:
            return {}
            
        if "names" in data and "name_generations" not in data:
            # Convert names field to name_generations
            data["name_generations"] = data.pop("names")
            
        # Ensure each name has required fields
        if "name_generations" in data:
            for name in data["name_generations"]:
                if "naming_category" not in name:
                    name["naming_category"] = "General"
                if "brand_name" not in name and "name" in name:
                    name["brand_name"] = name.pop("name")
                    
        return data
        
    def _transform_semantic_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform semantic analysis data to the expected format."""
        if not data:
            return {}
            
        if "analyses" in data and "semantic_analyses" not in data:
            # Convert analyses field to semantic_analyses
            data["semantic_analyses"] = data.pop("analyses")
            
        return data
        
    def _transform_linguistic_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform linguistic analysis data to the expected format."""
        if not data:
            return {}
            
        if "analyses" in data and "linguistic_analyses" not in data:
            # Convert analyses field to linguistic_analyses
            data["linguistic_analyses"] = data.pop("analyses")
            
        return data
        
    def _transform_cultural_sensitivity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform cultural sensitivity data to the expected format."""
        if not data:
            return {}
            
        if "analyses" in data and "cultural_analyses" not in data:
            # Convert analyses field to cultural_analyses
            data["cultural_analyses"] = data.pop("analyses")
            
        return data
        
    def _transform_name_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform name evaluation data to the expected format."""
        if not data:
            return {}
            
        if "evaluations" not in data and "name_evaluations" in data:
            # Convert name_evaluations field to evaluations
            data["evaluations"] = data.pop("name_evaluations")
            
        return data
        
    def _transform_translation_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform translation analysis data to the expected format."""
        if not data:
            return {}
            
        if "analyses" in data and "translation_analyses" not in data:
            # Convert analyses field to translation_analyses
            data["translation_analyses"] = data.pop("analyses")
            
        return data
        
    def _transform_market_research(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform market research data to the expected format."""
        if not data:
            return {}
            
        if "research" in data and "market_researches" not in data:
            # Convert research field to market_researches
            data["market_researches"] = data.pop("research")
            
        return data
        
    def _transform_competitor_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform competitor analysis data to the expected format."""
        if not data:
            return {}
            
        if "analyses" in data and "competitor_analyses" not in data:
            # Convert analyses field to competitor_analyses
            data["competitor_analyses"] = data.pop("analyses")
            
        return data
        
    def _transform_domain_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform domain analysis data to the expected format."""
        if not data:
            return {}
            
        if "analyses" in data and "domain_analyses" not in data:
            # Convert analyses field to domain_analyses
            data["domain_analyses"] = data.pop("analyses")
            
        return data
        
    def _transform_survey_simulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform survey simulation data to the expected format."""
        if not data:
            return {}
            
        if "surveys" in data and "survey_simulations" not in data:
            # Convert surveys field to survey_simulations
            data["survey_simulations"] = data.pop("surveys")
            
        return data

    def _validate_section_data(self, section_name: str, data: Any) -> Tuple[bool, List[str]]:
        """
        Validate section data for common issues and provide diagnostics.
        
        Args:
            section_name: Name of the section being validated
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if data exists
        if data is None:
            issues.append(f"Section '{section_name}' data is None")
            return False, issues
            
        # Check if data is empty
        if isinstance(data, dict) and not data:
            issues.append(f"Section '{section_name}' contains an empty dictionary")
        elif isinstance(data, list) and not data:
            issues.append(f"Section '{section_name}' contains an empty list")
        elif isinstance(data, str) and not data.strip():
            issues.append(f"Section '{section_name}' contains an empty string")
            
        # Check if data is in expected format
        if section_name in ["brand_context"] and not isinstance(data, dict):
            issues.append(f"Section '{section_name}' should be a dictionary, got {type(data).__name__}")
            return False, issues
            
        if section_name in ["name_generation", "linguistic_analysis", "semantic_analysis", 
                          "cultural_sensitivity", "translation_analysis", "competitor_analysis", 
                          "domain_analysis", "market_research"]:
            # These sections should typically have arrays of items
            list_fields = [
                "name_generations", "linguistic_analyses", "semantic_analyses", 
                "cultural_analyses", "translation_analyses", "competitor_analyses", 
                "domain_analyses", "market_researches"
            ]
            
            # Check if any expected list fields are present and valid
            found_valid_list = False
            for field in list_fields:
                if isinstance(data, dict) and field in data:
                    if not isinstance(data[field], list):
                        issues.append(f"Field '{field}' in section '{section_name}' should be a list, got {type(data[field]).__name__}")
                    elif not data[field]:
                        issues.append(f"Field '{field}' in section '{section_name}' is an empty list")
                    else:
                        found_valid_list = True
                        
                        # Check if items in the list have expected format
                        invalid_items = [i for i, item in enumerate(data[field]) 
                                        if not isinstance(item, dict) or "brand_name" not in item]
                        if invalid_items:
                            issues.append(f"{len(invalid_items)} items in '{field}' missing 'brand_name' or not dictionaries")
            
            if not found_valid_list and isinstance(data, dict):
                issues.append(f"No valid list fields found in section '{section_name}' data")
        
        # For LLM-generated sections, just check they're dictionaries with some content
        if section_name in ["executive_summary", "recommendations"]:
            if not isinstance(data, dict):
                issues.append(f"LLM-generated section '{section_name}' should be a dictionary, got {type(data).__name__}")
                return False, issues
        
        # Return overall validity and list of issues
        return len(issues) == 0, issues

    async def fetch_raw_data(self, run_id: str) -> Dict[str, Any]:
        """Fetch all raw data for a specific run_id from the report_raw_data table."""
        logger.info(f"Fetching all raw data for run_id: {run_id}")
        
        query = f"""
        SELECT section_name, raw_data
        FROM report_raw_data
        WHERE run_id = '{run_id}'
        ORDER BY section_name
        """
        
        result = await self.supabase.execute_with_retry(query, {})
        
        if not result:
            logger.warning(f"No data found for run_id: {run_id}")
            return {}
        
        # Track data quality issues
        data_quality_issues = {}
        
        # Transform results into a dictionary with section_name as keys
        sections_data = {}
        for row in result:
            db_section_name = row['section_name']
            raw_data = row['raw_data']
            
            # Map DB section name to formatter section name
            formatter_section_name = self.SECTION_MAPPING.get(db_section_name, db_section_name)
            
            # Apply transformation if available
            if formatter_section_name in self.data_transformers:
                transformer = self.data_transformers[formatter_section_name]
                try:
                    transformed_data = transformer(raw_data)
                    
                    # Validate transformed data
                    is_valid, issues = self._validate_section_data(formatter_section_name, transformed_data)
                    if not is_valid or issues:
                        for issue in issues:
                            logger.warning(f"Data quality issue in {formatter_section_name}: {issue}")
                        if issues:
                            data_quality_issues[formatter_section_name] = issues
                    
                    sections_data[formatter_section_name] = transformed_data
                except Exception as e:
                    logger.error(f"Error transforming data for section {formatter_section_name}: {str(e)}")
                    logger.debug(f"Raw data for failed transformation: {str(raw_data)[:500]}...")
                    # Store error but continue with other sections
                    data_quality_issues[formatter_section_name] = [f"Transformation error: {str(e)}"]
                    # Still store raw data for potential fallback handling
                    sections_data[formatter_section_name] = raw_data
            else:
                # Use raw data if no transformer is available
                # Also validate raw data
                is_valid, issues = self._validate_section_data(formatter_section_name, raw_data)
                if not is_valid or issues:
                    for issue in issues:
                        logger.warning(f"Data quality issue in {formatter_section_name}: {issue}")
                    if issues:
                        data_quality_issues[formatter_section_name] = issues
                        
                sections_data[formatter_section_name] = raw_data
        
        # Store data quality issues for later reporting
        if data_quality_issues:
            logger.warning(f"Found {len(data_quality_issues)} sections with data quality issues")
            sections_data["_data_quality_issues"] = data_quality_issues
            
        logger.info(f"Found {len(sections_data)} sections for run_id: {run_id}")
        return sections_data

    def _setup_document_styles(self, doc: Document) -> None:
        """Set up consistent document styles."""
        # Set default font and document properties
        style = doc.styles['Normal']
        style.font.name = 'Aptos'
        style.font.size = Pt(11)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.line_spacing = 1.15
        
        # Set heading styles
        for i in range(1, 4):
            heading_style = doc.styles[f'Heading {i}']
            heading_style.font.name = 'Aptos Header'
            
            # Different sizes for different heading levels
            if i == 1:
                heading_style.font.size = Pt(16)
                heading_style.font.color.rgb = RGBColor(0, 70, 127)  # Dark blue for main headers
            elif i == 2:
                heading_style.font.size = Pt(14)
                heading_style.font.color.rgb = RGBColor(0, 112, 192)  # Medium blue for subheaders
            else:
                heading_style.font.size = Pt(12)
                heading_style.font.color.rgb = RGBColor(68, 114, 196)  # Light blue for minor headers
                
            heading_style.font.bold = True
            heading_style.paragraph_format.space_before = Pt(12 + (3-i)*4)  # More space before larger headings
            heading_style.paragraph_format.space_after = Pt(6)
            heading_style.paragraph_format.keep_with_next = True  # Keep headings with following paragraph
        
        # Set List Bullet style
        if 'List Bullet' in doc.styles:
            bullet_style = doc.styles['List Bullet']
            bullet_style.font.name = 'Aptos'
            bullet_style.font.size = Pt(11)
            bullet_style.paragraph_format.left_indent = Inches(0.25)
            bullet_style.paragraph_format.first_line_indent = Inches(-0.25)
        
        # Set Caption style for tables and figures
        if 'Caption' in doc.styles:
            caption_style = doc.styles['Caption']
            caption_style.font.name = 'Aptos'
            caption_style.font.size = Pt(10)
            caption_style.font.italic = True
            caption_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_style.paragraph_format.space_before = Pt(6)
            caption_style.paragraph_format.space_after = Pt(12)
        
        # Set Quote style 
        if 'Quote' in doc.styles:
            quote_style = doc.styles['Quote']
            quote_style.font.name = 'Aptos'
            quote_style.font.size = Pt(11)
            quote_style.font.italic = True
            quote_style.paragraph_format.left_indent = Inches(0.5)
            quote_style.paragraph_format.right_indent = Inches(0.5)
            
        # Set Intense Quote style
        if 'Intense Quote' in doc.styles:
            intense_quote = doc.styles['Intense Quote']
            intense_quote.font.name = 'Aptos'
            intense_quote.font.size = Pt(11)
            intense_quote.font.bold = True
            intense_quote.font.italic = True
            intense_quote.paragraph_format.left_indent = Inches(0.5)
            intense_quote.paragraph_format.right_indent = Inches(0.5)
    
    def _add_section_divider(self, doc: Document, add_page_break: bool = False) -> None:
        """Add a consistent divider between sections."""
        # Add horizontal line
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run("_" * 50)
        
        # Add page break if requested
        if add_page_break:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        else:
            # Just add some space
            doc.add_paragraph()

    def _add_title_page(self, doc: Document, run_id: str, brand_context: Dict[str, Any]) -> None:
        """Add a professional title page to the document."""
        # Get title page content from LLM
        title_content = self.llm_client.generate(
            get_title_page_prompt(run_id=run_id, brand_context=brand_context)
        )
        
        # Add title
        title = doc.add_heading(title_content["title"], level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add subtitle
        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle.add_run(title_content["subtitle"]).italic = True
        
        # Add metadata
        metadata = doc.add_paragraph()
        metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER
        metadata.add_run(f"Run ID: {run_id}\n")
        metadata.add_run(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").italic = True
        
        # Add page break
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def _add_table_of_contents(self, doc: Document) -> None:
        """Add a table of contents to the document."""
        # Get TOC content from LLM
        toc_content = self.llm_client.generate(get_toc_prompt())
        
        # Add TOC heading
        doc.add_heading("Table of Contents", level=1)
        doc.add_paragraph()
        
        # Add TOC entries
        for entry in toc_content["sections"]:
            p = doc.add_paragraph()
            p.add_run(entry["title"]).bold = True
            p.add_run(" " * 2)
            p.add_run(entry["description"])
        
        # Add page break
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    async def upload_report_to_storage(self, file_path: str, run_id: str) -> str:
        """
        Upload the generated report to Supabase Storage.
        
        Args:
            file_path: Path to the local report file
            run_id: The run ID associated with the report
            
        Returns:
            The URL of the uploaded report
        """
        logger.info(f"Uploading report to Supabase Storage: {file_path}")
        
        # Extract filename from path
        filename = os.path.basename(file_path)
        
        # Define the storage path - organize by run_id and timestamp
        storage_path = f"{run_id}/{filename}"
        
        # Get file size in KB
        file_size_kb = os.path.getsize(file_path) // 1024
        
        try:
            # Read the file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Upload to Supabase Storage
            result = await self.supabase.storage_upload_with_retry(
                bucket=self.STORAGE_BUCKET,
                path=storage_path,
                file=file_content
            )
            
            # Get the public URL
            public_url = await self.supabase.storage_get_public_url(
                bucket=self.STORAGE_BUCKET,
                path=storage_path
            )
            
            logger.info(f"Report uploaded successfully to: {public_url}")
            
            # Store metadata in report_metadata table
            await self.store_report_metadata(
                run_id=run_id,
                report_url=public_url,
                file_size_kb=file_size_kb,
                format=self.FORMAT_DOCX
            )
            
            return public_url
            
        except Exception as e:
            logger.error(f"Error uploading report to storage: {str(e)}")
            raise
    
    async def store_report_metadata(self, run_id: str, report_url: str, file_size_kb: int, format: str = FORMAT_DOCX, notes: str = None) -> None:
        """
        Store metadata about the report in the report_metadata table.
        
        Args:
            run_id: The run ID associated with the report
            report_url: URL where the report is accessible
            file_size_kb: Size of the report file in KB
            format: Format of the report (default: docx)
            notes: Optional notes about the report
        """
        logger.info(f"Storing report metadata for run_id: {run_id}")
        
        # Check if a report already exists for this run_id to determine version
        query = f"""
        SELECT MAX(version) as current_version 
        FROM report_metadata 
        WHERE run_id = '{run_id}'
        """
        
        result = await self.supabase.execute_with_retry(query, {})
        current_version = 1  # Default to version 1
        
        if result and result[0]['current_version']:
            current_version = result[0]['current_version'] + 1
        
        # Insert metadata into the report_metadata table
        insert_query = f"""
        INSERT INTO report_metadata 
        (run_id, report_url, version, format, file_size_kb, notes, created_at)
        VALUES
        ('{run_id}', '{report_url}', {current_version}, '{format}', {file_size_kb}, 
        {'NULL' if notes is None else f"'{notes}'"}, NOW())
        """
        
        await self.supabase.execute_with_retry(insert_query, {})
        logger.info(f"Report metadata stored successfully for run_id: {run_id}, version: {current_version}")

    def _handle_section_error(self, doc: Document, section_name: str, error: Exception) -> None:
        """
        Handle errors that occur during section formatting.
        
        Args:
            doc: The document being generated
            section_name: The name of the section where the error occurred
            error: The exception that was raised
        """
        # Log detailed error with traceback for troubleshooting
        logger.error(f"Error formatting section '{section_name}': {str(error)}")
        logger.debug(f"Error traceback: {traceback.format_exc()}")
        
        # Store error for summary report
        self.formatting_errors[section_name] = str(error)
        
        # Add error message to the document
        error_para = doc.add_paragraph(style='Intense Quote')
        error_para.add_run("⚠️ ERROR: ").bold = True
        error_para.add_run(f"This section could not be properly formatted due to the following error:\n{str(error)}")
        
        # Add suggestions for troubleshooting
        doc.add_paragraph(
            "Possible solutions:\n"
            "• Check if the data is properly structured for this section\n"
            "• Verify that all required fields are present\n"
            "• Review the report_raw_data table for this section"
        )

    async def log_report_generation_issues(self, run_id: str) -> None:
        """
        Log report generation issues to the process_logs table.
        
        Args:
            run_id: The run ID associated with the report
        """
        try:
            # Skip if there are no issues to log
            if not (self.formatting_errors or hasattr(self, 'missing_sections') and self.missing_sections):
                return
                
            # Get data quality issues if they exist
            data_quality_issues = {}
            if hasattr(self, '_data_quality_issues'):
                data_quality_issues = self._data_quality_issues
                
            logger.info(f"Logging report generation issues for run_id: {run_id}")
            
            # Use the ProcessSupervisor to log task completion with issues
            from ..agents.process_supervisor import ProcessSupervisor
            supervisor = ProcessSupervisor(supabase=self.supabase)
            
            await supervisor.log_task_completion(
                run_id=run_id,
                agent_type="report_formatter",
                task_name="generate_report",
                data_quality_issues=data_quality_issues,
                formatting_errors=self.formatting_errors,
                missing_sections=list(self.missing_sections) if hasattr(self, 'missing_sections') and self.missing_sections else None
            )
            
            logger.info(f"Successfully logged report generation issues for run_id: {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log report generation issues: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Continue execution - logging issues should not prevent report generation
    
    async def generate_report(self, run_id: str, output_dir: Optional[str] = None, upload_to_storage: bool = True) -> str:
        """
        Generate a complete report document for the given run_id.
        
        Args:
            run_id: The run ID to generate a report for
            output_dir: Optional directory to save the report in
            upload_to_storage: Whether to upload the report to Supabase Storage
            
        Returns:
            Path to the generated report document
        """
        # Reset error tracking for this run
        self.formatting_errors = {}
        self.missing_sections = set()  # Track missing sections
        
        # Create output directory if needed
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"formatted_report_{timestamp}"
            
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Report generation for {run_id} - Output in {output_dir}")
        
        # Fetch all raw data for this run
        try:
            sections_data = await self.fetch_raw_data(run_id)
        except Exception as e:
            logger.error(f"Critical error fetching data for run_id {run_id}: {str(e)}")
            error_msg = f"Failed to fetch data for run_id: {run_id}. Error: {str(e)}"
            # Create a minimal document with error information
            doc = Document()
            self._setup_document_styles(doc)
            doc.add_heading(f"Error Report for Run ID: {run_id}", level=0)
            doc.add_paragraph(error_msg, style='Intense Quote')
            doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            doc_path = os.path.join(output_dir, f"Error_Report_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
            doc.save(doc_path)
            logger.info(f"Error report saved to {doc_path}")
            
            # Log error to process_logs
            await self.log_report_generation_issues(run_id)
            
            return doc_path
            
        if not sections_data:
            error_msg = f"No data found for run_id: {run_id}. Cannot generate report."
            logger.error(error_msg)
            
            # Create a minimal document with error information
            doc = Document()
            self._setup_document_styles(doc)
            doc.add_heading(f"Empty Report for Run ID: {run_id}", level=0)
            doc.add_paragraph(error_msg, style='Intense Quote')
            doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            doc_path = os.path.join(output_dir, f"Empty_Report_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
            doc.save(doc_path)
            logger.info(f"Empty report saved to {doc_path}")
            
            # Log error to process_logs
            await self.log_report_generation_issues(run_id)
            
            return doc_path
            
        # Extract and remove data quality issues if present
        data_quality_issues = {}
        if "_data_quality_issues" in sections_data:
            data_quality_issues = sections_data.pop("_data_quality_issues")
            logger.info(f"Found {len(data_quality_issues)} sections with data quality issues")
            # Store for logging
            self._data_quality_issues = data_quality_issues
        
        # Create a new document
        doc = Document()
        
        # Set up document styles
        self._setup_document_styles(doc)
        
        # Add title page
        try:
            brand_context = sections_data.get("brand_context", {})
            self._add_title_page(doc, run_id, brand_context)
        except Exception as e:
            logger.error(f"Error creating title page: {str(e)}")
            self._handle_section_error(doc, "title_page", e)
        
        # Add table of contents
        try:
            self._add_table_of_contents(doc)
        except Exception as e:
            logger.error(f"Error creating table of contents: {str(e)}")
            self._handle_section_error(doc, "table_of_contents", e)
        
        # Define the order of sections - updated to match requirements
        section_order = [
            "executive_summary",  # Moved to front
            "brand_context",
            "name_generation",
            "semantic_analysis",
            "linguistic_analysis",
            "cultural_sensitivity",
            "brand_name_evaluation",
            "translation_analysis",
            "market_research",
            "competitor_analysis",
            "domain_analysis", 
            "survey_simulation",
            "recommendations"  # Moved to end
        ]
        
        # Define which sections should have page breaks
        major_sections = [
            "executive_summary",
            "brand_context",
            "name_generation",
            "recommendations"
        ]
        
        # Process each section in order
        for section_name in section_order:
            if section_name in sections_data:
                logger.info(f"Formatting section: {section_name}")
                
                # Add section header
                section_title = section_name.replace("_", " ").title()
                heading = doc.add_heading(section_title, level=1)
                
                # If there are known data quality issues, add warning
                if section_name in data_quality_issues:
                    warning_para = doc.add_paragraph(style='Intense Quote')
                    warning_para.add_run("⚠️ Warning: ").bold = True
                    warning_para.add_run("This section has known data quality issues that may affect formatting.")
                
                # Format the section based on its type
                try:
                    if section_name == "executive_summary":
                        # Generate executive summary using LLM
                        summary_content = self.llm_client.generate(
                            get_executive_summary_prompt(
                                run_id=run_id,
                                sections_data=sections_data,
                                brand_context=brand_context,
                                total_names=len(sections_data.get("name_generation", {}).get("name_generations", [])),
                                shortlisted_names=sections_data.get("shortlisted_names", []),
                                user_prompt=sections_data.get("user_prompt", "")
                            )
                        )
                        self._format_executive_summary(doc, summary_content)
                    elif section_name == "recommendations":
                        # Generate recommendations using LLM
                        recommendations_content = self.llm_client.generate(
                            get_recommendations_prompt(
                                run_id=run_id,
                                sections_data=sections_data,
                                brand_context=brand_context
                            )
                        )
                        self._format_recommendations(doc, recommendations_content)
                    else:
                        self._format_section(doc, section_name, sections_data[section_name])
                except Exception as e:
                    self._handle_section_error(doc, section_name, e)
                
                # Add divider with page break for major sections
                add_page_break = section_name in major_sections
                self._add_section_divider(doc, add_page_break)
            else:
                logger.warning(f"Missing section: {section_name}")
                self.missing_sections.add(section_name)
                
                # Add a placeholder message for missing sections
                section_title = section_name.replace("_", " ").title()
                heading = doc.add_heading(section_title, level=1)
                
                missing_para = doc.add_paragraph(style='Quote')
                missing_para.add_run("❓ MISSING SECTION: ").bold = True
                missing_para.add_run(f"No data was found for this section.")
                
                # Add divider with page break for major sections
                add_page_break = section_name in major_sections
                self._add_section_divider(doc, add_page_break)
        
        # Add a report generation summary with error information
        if self.formatting_errors or self.missing_sections or data_quality_issues:
            doc.add_heading("Report Generation Summary", level=1)
            
            if self.missing_sections:
                doc.add_heading("Missing Sections", level=2)
                doc.add_paragraph("The following sections were missing from the data and could not be included:")
                for section in sorted(self.missing_sections):
                    doc.add_paragraph(f"• {section.replace('_', ' ').title()}", style='List Bullet')
            
            if data_quality_issues:
                doc.add_heading("Data Quality Issues", level=2)
                doc.add_paragraph("The following sections had data quality issues identified during processing:")
                
                # Create a table to show data quality issues
                table = doc.add_table(rows=1, cols=2)
                table.style = 'Table Grid'
                
                # Add headers
                header_cells = table.rows[0].cells
                header_cells[0].text = "Section"
                header_cells[1].text = "Issues"
                
                for cell in header_cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.bold = True
                
                # Add data rows
                for section, issues in sorted(data_quality_issues.items()):
                    row = table.add_row().cells
                    row[0].text = section.replace('_', ' ').title()
                    row[1].text = "\n• ".join([""] + issues)
            
            if self.formatting_errors:
                doc.add_heading("Formatting Errors", level=2)
                doc.add_paragraph("The following sections encountered errors during formatting:")
                for section, error in sorted(self.formatting_errors.items()):
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(f"{section.replace('_', ' ').title()}: ").bold = True
                    p.add_run(error)
            
            # Add troubleshooting information
            doc.add_heading("Troubleshooting Information", level=2)
            doc.add_paragraph(
                "If you encounter issues with this report, please check the following:\n"
                "• Verify that all required data is present in the report_raw_data table\n"
                "• Ensure that the data format matches the expected structure for each section\n"
                "• Check the application logs for more detailed error information\n"
                "• Consider rerunning the report generation with corrected data"
            )
            
            # Log issues to process_logs
            await self.log_report_generation_issues(run_id)
        
        # Save the document
        doc_path = os.path.join(output_dir, f"Report_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx")
        doc.save(doc_path)
        logger.info(f"Report saved to {doc_path}")
        
        # Upload to Supabase Storage if requested
        if upload_to_storage:
            try:
                public_url = await self.upload_report_to_storage(doc_path, run_id)
                logger.info(f"Report accessible at: {public_url}")
            except Exception as e:
                logger.error(f"Failed to upload report to storage: {str(e)}")
                # Continue even if upload fails
        
        return doc_path

    def _format_executive_summary(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the executive summary section using LLM-generated content."""
        # Check for required fields
        required_fields = ["introduction", "project_overview", "methodology", 
                          "key_findings", "top_recommendations", "strategic_implications"]
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' in executive summary data")
        
        # Introduction and Brand Overview
        doc.add_heading("Introduction and Brand Overview", level=2)
        doc.add_paragraph(data["introduction"])
        
        # Project Overview and Objectives
        doc.add_heading("Project Overview and Objectives", level=2)
        doc.add_paragraph(data["project_overview"])
        
        # Methodology Summary
        doc.add_heading("Methodology Summary", level=2)
        doc.add_paragraph(data["methodology"])
        
        # Key Findings and Insights
        doc.add_heading("Key Findings and Insights", level=2)
        doc.add_paragraph(data["key_findings"])
        
        # Top Recommendations
        doc.add_heading("Top Recommendations", level=2)
        doc.add_paragraph(data["top_recommendations"])
        
        # Strategic Implications
        doc.add_heading("Strategic Implications", level=2)
        doc.add_paragraph(data["strategic_implications"])

    def _format_brand_context(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the brand context section according to requirements."""
        # Extract brand context data
        context = data.get("brand_context", {})
        if not context:
            doc.add_paragraph("No brand context data available.", style='Quote')
            return
        
        # Check if context is properly formatted
        if not isinstance(context, dict):
            raise TypeError(f"Brand context data must be a dictionary, got {type(context).__name__}")
            
        # Add key elements of brand context in the required order
        for field in [
            "brand_promise", 
            "brand_personality", 
            "brand_tone_of_voice", 
            "brand_values", 
            "brand_purpose", 
            "brand_mission", 
            "target_audience",
            "customer_needs", 
            "market_positioning", 
            "competitive_landscape",
            "industry_focus", 
            "industry_trends", 
            "brand_identity_brief"
        ]:
            if field in context:
                heading = field.replace("_", " ").title()
                doc.add_heading(heading, level=2)
                
                # Handle arrays vs text fields
                if isinstance(context[field], list):
                    if not context[field]:  # Empty list
                        doc.add_paragraph("No data available for this field.", style='Quote')
                    else:
                        for item in context[field]:
                            doc.add_paragraph(f"• {item}", style='List Bullet')
                elif isinstance(context[field], dict):
                    # Handle JSON fields
                    if not context[field]:  # Empty dict
                        doc.add_paragraph("No data available for this field.", style='Quote')
                    else:
                        for key, value in context[field].items():
                            p = doc.add_paragraph(style='List Bullet')
                            p.add_run(f"{key}: ").bold = True
                            p.add_run(str(value))
                else:
                    # Handle None or empty string
                    if context[field] is None or (isinstance(context[field], str) and not context[field].strip()):
                        doc.add_paragraph("No data available for this field.", style='Quote')
                    else:
                        doc.add_paragraph(context[field])

    def _format_name_generation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name generation section according to requirements."""
        # Extract name generation data
        names = data.get("name_generations", [])
        if not names:
            doc.add_paragraph("No name generation data available.", style='Quote')
            return
            
        # Verify names is a list
        if not isinstance(names, list):
            raise TypeError(f"Name generations data must be a list, got {type(names).__name__}")
            
        # Add summary
        doc.add_paragraph(
            f"A total of {len(names)} brand names were generated across {len(set(name.get('naming_category', 'Uncategorized') for name in names if isinstance(name, dict)))} different categories.",
            style='Quote'
        )
        
        # Group names by category as required - handle invalid items
        categories = {}
        invalid_items = 0
        
        for name in names:
            if not isinstance(name, dict):
                invalid_items += 1
                continue
                
            category = name.get("naming_category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
            
        if invalid_items > 0:
            doc.add_paragraph(f"⚠️ Warning: {invalid_items} name entries were skipped due to invalid format.", style='Intense Quote')
            
        # Add each category with all required fields
        for category, names_list in categories.items():
            doc.add_heading(f"{category}", level=2)
            
            # Create a table for names in this category with all required fields
            table = doc.add_table(rows=1, cols=5)
            table.style = 'Table Grid'
            
            # Add headers and style them
            header_cells = table.rows[0].cells
            for i, header in enumerate(["Brand Name", "Personality Alignment", "Promise Alignment", "Memorability", "Market Differentiation"]):
                header_cells[i].text = header
                # Make headers bold with light blue background
                for paragraph in header_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)  # White text
                header_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                # Set cell background color (if possible)
                try:
                    header_cells[i]._element.tcPr.append(parse_xml(f'<w:shd w:fill="4472C4" w:val="clear"/>'))
                except Exception as e:
                    logger.warning(f"Could not set cell background color: {str(e)}")
            
            # Add rows with all the required data
            for name in names_list:
                row = table.add_row().cells
                row[0].text = name.get("brand_name", "N/A")
                row[1].text = name.get("brand_personality_alignment", "N/A")
                row[2].text = name.get("brand_promise_alignment", "N/A")
                row[3].text = name.get("memorability_score_details", "N/A")
                row[4].text = name.get("market_differentiation_details", "N/A")
                
                # Make brand name column bold
                for paragraph in row[0].paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                
            # Add table caption
            caption = doc.add_paragraph(f"Table: {category} Brand Name Options", style='Caption')
            
            # Add some space after the table
            doc.add_paragraph()

    def _format_linguistic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the linguistic analysis section according to requirements."""
        # Extract linguistic analysis data
        analyses = data.get("linguistic_analyses", [])
        if not analyses:
            doc.add_paragraph("No linguistic analysis data available.", style='Quote')
            return
        
        # Verify analyses is a list
        if not isinstance(analyses, list):
            raise TypeError(f"Linguistic analyses data must be a list, got {type(analyses).__name__}")
            
        # Count valid analyses
        valid_analyses = [a for a in analyses if isinstance(a, dict) and "brand_name" in a]
        invalid_count = len(analyses) - len(valid_analyses)
        
        # Add introduction
        doc.add_paragraph(
            f"Linguistic analysis was performed on {len(valid_analyses)} brand names to evaluate "
            "pronunciation, readability, and other linguistic features."
        )
        
        if invalid_count > 0:
            doc.add_paragraph(f"⚠️ Warning: {invalid_count} analyses were skipped due to invalid format.", style='Intense Quote')
        
        # Group by brand name as required
        brands = {}
        for analysis in valid_analyses:
            brand_name = analysis.get("brand_name", "Unknown")
            brands[brand_name] = analysis
        
        if not brands:
            doc.add_paragraph("No valid linguistic analyses found with brand names.", style='Quote')
            return
            
        # Process each brand
        for brand_name, analysis in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a table with key linguistic features
            table = doc.add_table(rows=6, cols=2)
            table.style = 'Table Grid'
            
            # List of fields to display
            fields = [
                ("Pronunciation Ease", "pronunciation_ease"),
                ("Euphony vs Cacophony", "euphony_vs_cacophony"),
                ("Word Class", "word_class"),
                ("Semantic Distance from Competitors", "semantic_distance_from_competitors"),
                ("Ease of Marketing Integration", "ease_of_marketing_integration"),
                ("Overall Readability Score", "overall_readability_score")
            ]
            
            # Add rows with field-value pairs
            for i, (display_name, field_name) in enumerate(fields):
                if i < len(table.rows):
                    row = table.rows[i].cells
                    row[0].text = display_name
                    row[0].paragraphs[0].runs[0].bold = True
                    row[1].text = analysis.get(field_name, "N/A")
            
            # Add notes if available
            if analysis.get("notes"):
                doc.add_heading("Notes", level=3)
                doc.add_paragraph(analysis.get("notes"))
            
            # Add some space between brands
            doc.add_paragraph()

    def _format_semantic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the semantic analysis section."""
        # Extract semantic analysis data
        analyses = data.get("semantic_analyses", [])
        if not analyses:
            doc.add_paragraph("No semantic analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Semantic analysis was conducted for {len(analyses)} brand names to understand "
            "their meaning, emotional valence, and associations."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Denotative Meaning"
        header_cells[2].text = "Emotional Valence"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("denotative_meaning", "")
            row[2].text = analysis.get("emotional_valence", "")

    def _format_cultural_sensitivity(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the cultural sensitivity section."""
        # Extract cultural sensitivity data
        analyses = data.get("cultural_analyses", [])
        if not analyses:
            doc.add_paragraph("No cultural sensitivity analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Cultural sensitivity analysis was performed on {len(analyses)} brand names to identify "
            "any potential cultural issues or sensitivities."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Overall Risk Rating"
        header_cells[2].text = "Cultural Connotations"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("overall_risk_rating", "")
            row[2].text = analysis.get("cultural_connotations", "")

    def _format_translation_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the translation analysis section according to requirements."""
        # Extract translation analysis data
        analyses = data.get("translation_analyses", [])
        if not analyses:
            doc.add_paragraph("No translation analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Translation analysis was conducted for {len(analyses)} brand name-language combinations "
            "to evaluate international viability."
        )
        
        # Group by brand name as required
        brands = {}
        for analysis in analyses:
            brand_name = analysis.get("brand_name", "Unknown")
            if brand_name not in brands:
                brands[brand_name] = []
            brands[brand_name].append(analysis)
        
        # Process each brand
        for brand_name, brand_analyses in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a table for this brand with all required fields
            table = doc.add_table(rows=1, cols=5)
            table.style = 'Table Grid'
            
            # Add headers
            header_cells = table.rows[0].cells
            header_cells[0].text = "Target Language"
            header_cells[1].text = "Direct Translation"
            header_cells[2].text = "Cultural Acceptability"
            header_cells[3].text = "Adaptation Needed"
            header_cells[4].text = "Brand Essence Preserved"
            
            # Add data rows with all required fields
            for analysis in brand_analyses:
                row = table.add_row().cells
                row[0].text = analysis.get("target_language", "")
                row[1].text = analysis.get("direct_translation", "")
                row[2].text = analysis.get("cultural_acceptability", "")
                row[3].text = "Yes" if analysis.get("adaptation_needed") else "No"
                row[4].text = analysis.get("brand_essence_preserved", "")
                
                # If adaptation needed, show the proposed adaptation
                if analysis.get("adaptation_needed") and analysis.get("proposed_adaptation"):
                    additional_row = table.add_row().cells
                    additional_row[0].text = "Proposed Adaptation:"
                    additional_row[1].text = analysis.get("proposed_adaptation", "")
                    additional_row[1].merge(additional_row[4])
            
            # Add space after table
            doc.add_paragraph()

    def _format_survey_simulation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the survey simulation section according to requirements."""
        # Extract survey simulation data
        surveys = data.get("survey_simulations", [])
        if not surveys:
            doc.add_paragraph("No survey simulation data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Market survey simulations were conducted for {len(surveys)} brand names "
            "to evaluate market reception and adoption potential."
        )
        
        # Group by brand name as required
        brands = {}
        for survey in surveys:
            brand_name = survey.get("brand_name", "Unknown")
            if brand_name not in brands:
                brands[brand_name] = []
            brands[brand_name].append(survey)
        
        # Process each brand
        for brand_name, brand_surveys in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a summary table with key metrics
            summary_table = doc.add_table(rows=1, cols=5)
            summary_table.style = 'Table Grid'
            
            # Add headers
            headers = summary_table.rows[0].cells
            headers[0].text = "Brand Promise Score"
            headers[1].text = "Personality Fit"
            headers[2].text = "Competitive Diff."
            headers[3].text = "Market Adoption"
            headers[4].text = "Strategic Ranking"
            
            # Calculate averages
            avg_promise = sum(float(s.get("brand_promise_perception_score", 0)) for s in brand_surveys) / len(brand_surveys)
            avg_personality = sum(float(s.get("personality_fit_score", 0)) for s in brand_surveys) / len(brand_surveys)
            avg_diff = sum(float(s.get("competitive_differentiation_score", 0)) for s in brand_surveys) / len(brand_surveys)
            avg_adoption = sum(float(s.get("simulated_market_adoption_score", 0)) for s in brand_surveys) / len(brand_surveys)
            
            # Get strategic ranking (use the first one as representative)
            strategic_ranking = brand_surveys[0].get("strategic_ranking", "N/A")
            
            # Add summary row
            summary_row = summary_table.add_row().cells
            summary_row[0].text = f"{avg_promise:.1f}"
            summary_row[1].text = f"{avg_personality:.1f}"
            summary_row[2].text = f"{avg_diff:.1f}"
            summary_row[3].text = f"{avg_adoption:.1f}"
            summary_row[4].text = str(strategic_ranking)
            
            # Qualitative feedback
            doc.add_heading("Qualitative Feedback", level=3)
            for i, survey in enumerate(brand_surveys):
                if survey.get("qualitative_feedback_summary"):
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(f"Feedback {i+1}: ").bold = True
                    p.add_run(survey.get("qualitative_feedback_summary", ""))
            
            # Final recommendation if available
            if brand_surveys[0].get("final_survey_recommendation"):
                doc.add_heading("Final Recommendation", level=3)
                doc.add_paragraph(brand_surveys[0].get("final_survey_recommendation"))
            
            # Add demographic information if available
            doc.add_heading("Survey Demographics", level=3)
            demo_table = doc.add_table(rows=5, cols=2)
            demo_table.style = 'Table Grid'
            
            row = demo_table.rows[0].cells
            row[0].text = "Industry"
            row[1].text = brand_surveys[0].get("industry", "N/A")
            
            row = demo_table.rows[1].cells
            row[0].text = "Company Size"
            row[1].text = brand_surveys[0].get("company_size_employees", "N/A")
            
            row = demo_table.rows[2].cells
            row[0].text = "Job Titles"
            row[1].text = ", ".join(set(s.get("job_title", "N/A") for s in brand_surveys if s.get("job_title")))
            
            row = demo_table.rows[3].cells
            row[0].text = "Departments"
            row[1].text = ", ".join(set(s.get("department", "N/A") for s in brand_surveys if s.get("department")))
            
            row = demo_table.rows[4].cells
            row[0].text = "Seniority Levels"
            row[1].text = ", ".join(set(s.get("seniority", "N/A") for s in brand_surveys if s.get("seniority")))
            
            # Add separation between brands
            doc.add_paragraph()
            doc.add_paragraph("---")
            doc.add_paragraph()

    def _format_name_evaluation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name evaluation section."""
        # Extract name evaluation data
        evaluations = data.get("evaluations", [])
        if not evaluations:
            doc.add_paragraph("No name evaluation data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Comprehensive evaluations were performed for {len(evaluations)} brand names "
            "to assess their overall performance and suitability."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Overall Score"
        header_cells[2].text = "Shortlisted"
        
        # Add data rows
        for eval in evaluations:
            row = table.add_row().cells
            row[0].text = eval.get("brand_name", "")
            row[1].text = str(eval.get("overall_score", ""))
            row[2].text = "Yes" if eval.get("shortlist_status") else "No"
            
        # Highlight top performers
        shortlisted = [e for e in evaluations if e.get("shortlist_status")]
        if shortlisted:
            doc.add_heading("Shortlisted Names", level=2)
            for name in shortlisted:
                p = doc.add_paragraph(style='List Bullet')
                p.add_run(name.get("brand_name", "")).bold = True
                p.add_run(f" - {name.get('evaluation_comments', '')}")

    def _format_domain_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the domain analysis section."""
        # Extract domain analysis data
        analyses = data.get("domain_analyses", [])
        if not analyses:
            doc.add_paragraph("No domain analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"Domain availability analysis was conducted for {len(analyses)} brand names "
            "to evaluate online presence opportunities."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "Domain Exact Match"
        header_cells[2].text = "Acquisition Cost"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("domain_exact_match", "")
            row[2].text = analysis.get("acquisition_cost", "")

    def _format_seo_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the SEO analysis section."""
        # Extract SEO analysis data
        analyses = data.get("seo_analyses", [])
        if not analyses:
            doc.add_paragraph("No SEO analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            f"SEO and online discoverability analysis was performed for {len(analyses)} brand names "
            "to evaluate search engine performance potential."
        )
        
        # Create a table
        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        
        # Add headers
        header_cells = table.rows[0].cells
        header_cells[0].text = "Brand Name"
        header_cells[1].text = "SEO Viability Score"
        header_cells[2].text = "Search Volume"
        
        # Add data rows
        for analysis in analyses:
            row = table.add_row().cells
            row[0].text = analysis.get("brand_name", "")
            row[1].text = analysis.get("seo_viability_score", "")
            row[2].text = analysis.get("search_volume", "")
            
        # Add SEO recommendations
        doc.add_heading("SEO Recommendations", level=2)
        for analysis in analyses:
            if analysis.get("seo_recommendations"):
                p = doc.add_paragraph(style='List Bullet')
                p.add_run(f"{analysis.get('brand_name', '')}: ").bold = True
                p.add_run(analysis.get("seo_recommendations", ""))

    def _format_competitor_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the competitor analysis section."""
        # Extract competitor analysis data
        analyses = data.get("competitor_analyses", [])
        if not analyses:
            doc.add_paragraph("No competitor analysis data available.")
            return
            
        # Add introduction
        doc.add_paragraph(
            "Competitor analysis was performed to evaluate the proposed brand names "
            "against existing competitors in the marketplace."
        )
        
        # Group by brand name
        brands = {}
        for analysis in analyses:
            brand_name = analysis.get("brand_name", "Unknown")
            if brand_name not in brands:
                brands[brand_name] = []
            brands[brand_name].append(analysis)
        
        # Process each brand
        for brand_name, competitors in brands.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a table for this brand
            table = doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'
            
            # Add headers
            header_cells = table.rows[0].cells
            header_cells[0].text = "Competitor"
            header_cells[1].text = "Risk of Confusion"
            header_cells[2].text = "Trademark Conflict Risk"
            
            # Add data rows
            for comp in competitors:
                row = table.add_row().cells
                row[0].text = comp.get("competitor_name", "")
                row[1].text = comp.get("risk_of_confusion", "")
                row[2].text = comp.get("trademark_conflict_risk", "")
            
            # Add space after table
            doc.add_paragraph()

    def _format_market_research(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the market research section."""
        # Extract market research data
        research = data.get("market_researches", [])
        if not research:
            doc.add_paragraph("No market research data available.", style='Quote')
            return
            
        # Verify research is a list
        if not isinstance(research, list):
            raise TypeError(f"Market research data must be a list, got {type(research).__name__}")
            
        # Count valid researches
        valid_researches = [r for r in research if isinstance(r, dict) and "brand_name" in r]
        invalid_count = len(research) - len(valid_researches)
            
        # Add introduction
        doc.add_paragraph(
            f"Market research was conducted for {len(valid_researches)} shortlisted brand names "
            "to evaluate their market viability and potential."
        )
        
        if invalid_count > 0:
            doc.add_paragraph(f"⚠️ Warning: {invalid_count} research entries were skipped due to invalid format.", style='Intense Quote')
            
        if not valid_researches:
            doc.add_paragraph("No valid market research entries found with brand names.", style='Quote')
            return
        
        # Process each brand
        for res in valid_researches:
            brand_name = res.get("brand_name", "Unknown")
            doc.add_heading(brand_name, level=2)
            
            # Fields to check with their display names
            fields = [
                ("market_opportunity", "Market Opportunity"),
                ("target_audience_fit", "Target Audience Fit"),
                ("potential_risks", "Potential Risks")
            ]
            
            # Add each field if available
            has_content = False
            for field_name, display_name in fields:
                content = res.get(field_name)
                if content:
                    has_content = True
                    doc.add_heading(display_name, level=3)
                    doc.add_paragraph(content)
            
            # If no content was found, add a note
            if not has_content:
                doc.add_paragraph("No detailed market research data available for this brand name.", style='Quote')
                
            # Add space between brands
            doc.add_paragraph()

    def _format_recommendations(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the recommendations section using LLM-generated content."""
        # Check for required fields
        if "introduction" not in data:
            raise ValueError("Missing 'introduction' field in recommendations data")
            
        # Add introduction
        doc.add_paragraph(data["introduction"])
        
        # Add shortlisted names summary
        if data.get("shortlisted_names"):
            doc.add_heading("Shortlisted Names", level=2)
            for name_data in data["shortlisted_names"]:
                if not isinstance(name_data, dict) or "name" not in name_data:
                    doc.add_paragraph("⚠️ Error: Invalid name data format", style='Intense Quote')
                    continue
                    
                p = doc.add_paragraph()
                p.add_run(name_data["name"]).bold = True
                p.add_run("\n")
                
                if "description" in name_data:
                    p.add_run(name_data["description"])
                    
                if "strengths" in name_data and isinstance(name_data["strengths"], list):
                    p.add_run("\nKey Strengths:")
                    for strength in name_data["strengths"]:
                        doc.add_paragraph(f"• {strength}", style='List Bullet')
                        
                doc.add_paragraph()
        else:
            doc.add_paragraph("No shortlisted names data available.", style='Quote')
        
        # Add final recommendations
        if "final_recommendations" in data and isinstance(data["final_recommendations"], list):
            doc.add_heading("Final Recommendations", level=2)
            if not data["final_recommendations"]:
                doc.add_paragraph("No final recommendations provided.", style='Quote')
            
            for rec in data["final_recommendations"]:
                if not isinstance(rec, dict) or "name" not in rec:
                    doc.add_paragraph("⚠️ Error: Invalid recommendation format", style='Intense Quote')
                    continue
                    
                p = doc.add_paragraph()
                p.add_run(rec["name"]).bold = True
                p.add_run("\n")
                
                if "rationale" in rec:
                    p.add_run(rec["rationale"])
                    
                if rec.get("implementation_notes"):
                    doc.add_paragraph("Implementation Notes:", style='Intense Quote')
                    doc.add_paragraph(rec["implementation_notes"])
                    
                doc.add_paragraph()
        else:
            doc.add_heading("Final Recommendations", level=2)
            doc.add_paragraph("No final recommendations data available.", style='Quote')
        
        # Add strategic considerations
        if data.get("strategic_considerations") and isinstance(data["strategic_considerations"], list):
            doc.add_heading("Strategic Considerations", level=2)
            for consideration in data["strategic_considerations"]:
                doc.add_paragraph(f"• {consideration}", style='List Bullet')
        
        # Add next steps
        if data.get("next_steps") and isinstance(data["next_steps"], list):
            doc.add_heading("Next Steps", level=2)
            for step in data["next_steps"]:
                doc.add_paragraph(f"• {step}", style='List Bullet')

    def _format_generic_section(self, doc: Document, section_name: str, data: Dict[str, Any]) -> None:
        """Format a section with no specific formatting method."""
        if not data:
            doc.add_paragraph(f"No data available for {section_name}.", style='Quote')
            return
            
        # Make sure data is a dictionary
        if not isinstance(data, dict):
            doc.add_paragraph(f"⚠️ Warning: Data for {section_name} is not in the expected format (dictionary).", style='Intense Quote')
            # Try to display something useful
            doc.add_paragraph(f"Data type: {type(data).__name__}")
            doc.add_paragraph(f"Raw content: {str(data)[:500]}..." if len(str(data)) > 500 else str(data))
            return
            
        # Add a simple dump of the data
        doc.add_paragraph(f"Data for {section_name}:")
        
        # Dump the keys
        keys = ", ".join(data.keys()) if data else "No keys found"
        doc.add_paragraph(f"Available data fields: {keys}")
        
        # Try to detect lists and display them
        for key, value in data.items():
            if isinstance(value, list) and value:
                doc.add_heading(f"{key.replace('_', ' ').title()}", level=2)
                doc.add_paragraph(f"Total items: {len(value)}")
                
                # Track invalid items
                invalid_items = 0
                
                # Display a sample of items
                for i, item in enumerate(value[:5]):
                    if isinstance(item, dict):
                        # For dictionaries, show key-value pairs
                        p = doc.add_paragraph(style='List Bullet')
                        p.add_run(f"Item {i+1}: ")
                        
                        # Find name-like fields to display prominently
                        name_field = None
                        for name_key in ["brand_name", "name", "title"]:
                            if name_key in item and item[name_key]:
                                name_field = item[name_key]
                                break
                                
                        if name_field:
                            p.add_run(f"{name_field} ").bold = True
                            
                            # Display other fields
                            field_count = 0
                            for k, v in item.items():
                                if k not in ["brand_name", "name", "title"] and isinstance(v, (str, int, float, bool)) and str(v):
                                    if field_count < 3:  # Limit to 3 fields for readability
                                        p.add_run(f"{k}: {v}; ")
                                        field_count += 1
                    elif item is not None:
                        # For non-dictionaries, show the item directly
                        doc.add_paragraph(f"• {item}", style='List Bullet')
                    else:
                        invalid_items += 1
                
                if invalid_items > 0:
                    doc.add_paragraph(f"⚠️ Warning: {invalid_items} items were None or invalid format", style='Intense Quote')
                    
                if len(value) > 5:
                    doc.add_paragraph(f"... and {len(value) - 5} more items.")
            elif isinstance(value, dict) and value:
                doc.add_heading(f"{key.replace('_', ' ').title()}", level=2)
                
                # For dictionaries, show a table of key-value pairs
                table = doc.add_table(rows=min(len(value), 10) + 1, cols=2)
                table.style = 'Table Grid'
                
                # Add header
                header_cells = table.rows[0].cells
                header_cells[0].text = "Field"
                header_cells[1].text = "Value"
                
                # Make headers bold
                for i in range(2):
                    for paragraph in header_cells[i].paragraphs:
                        for run in paragraph.runs:
                            run.font.bold = True
                
                # Add data rows
                for i, (k, v) in enumerate(list(value.items())[:10]):
                    if i < len(table.rows) - 1:
                        row = table.rows[i+1].cells
                        row[0].text = k
                        row[1].text = str(v)[:100] + ("..." if len(str(v)) > 100 else "")
                
                if len(value) > 10:
                    doc.add_paragraph(f"... and {len(value) - 10} more fields.")
            elif value is None or (isinstance(value, (list, dict)) and not value):
                doc.add_paragraph(f"{key.replace('_', ' ').title()}: No data available", style='Quote')
            elif isinstance(value, (str, int, float, bool)):
                doc.add_heading(f"{key.replace('_', ' ').title()}", level=2)
                doc.add_paragraph(str(value))
            else:
                doc.add_paragraph(f"{key.replace('_', ' ').title()}: {type(value).__name__} (cannot display)", style='Quote')

    def _format_section(self, doc: Document, section_name: str, section_data: Dict[str, Any]) -> None:
        """Format a specific section based on its type."""
        # Make sure section_data is a dictionary
        if not isinstance(section_data, dict):
            raise TypeError(f"Section data for '{section_name}' must be a dictionary, got {type(section_data).__name__}")
            
        try:
            # Use specific formatting methods based on section_name
            if section_name == "brand_context":
                self._format_brand_context(doc, section_data)
            elif section_name == "name_generation":
                self._format_name_generation(doc, section_data)
            elif section_name == "linguistic_analysis":
                self._format_linguistic_analysis(doc, section_data)
            elif section_name == "semantic_analysis":
                self._format_semantic_analysis(doc, section_data)
            elif section_name == "cultural_sensitivity":
                self._format_cultural_sensitivity(doc, section_data)
            elif section_name == "translation_analysis":
                self._format_translation_analysis(doc, section_data)
            elif section_name == "survey_simulation":
                self._format_survey_simulation(doc, section_data)
            elif section_name == "name_evaluation":
                self._format_name_evaluation(doc, section_data)
            elif section_name == "domain_analysis":
                self._format_domain_analysis(doc, section_data)
            elif section_name == "seo_analysis":
                self._format_seo_analysis(doc, section_data)
            elif section_name == "competitor_analysis":
                self._format_competitor_analysis(doc, section_data)
            elif section_name == "market_research":
                self._format_market_research(doc, section_data)
            else:
                # Generic formatting if no specific method exists
                logger.warning(f"No specific formatter found for section: {section_name}, using generic formatter")
                self._format_generic_section(doc, section_name, section_data)
        except Exception as e:
            # If a specific formatting method fails, try the generic formatter as fallback
            logger.error(f"Error in specific formatter for {section_name}, falling back to generic: {str(e)}")
            doc.add_paragraph(f"⚠️ Error in specialized formatter: {str(e)}", style='Intense Quote')
            
            try:
                self._format_generic_section(doc, section_name, section_data)
            except Exception as e2:
                # If even the generic formatter fails, we have a serious problem with the data
                raise ValueError(f"Failed to format section '{section_name}' with both specialized and generic formatters. Original error: {str(e)}. Generic formatter error: {str(e2)}")


async def main(run_id: str = None):
    """Main function to run the formatter."""
    if not run_id:
        # Use a default run ID for testing
        run_id = "mae_20250312_141302_d45cccde"  # Replace with an actual run ID
        
    formatter = ReportFormatter()
    output_path = await formatter.generate_report(run_id)
    print(f"Report generated at: {output_path}")


if __name__ == "__main__":
    # Command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a formatted report from raw data")
    parser.add_argument("run_id", help="The run ID to generate a report for")
    args = parser.parse_args()
    
    asyncio.run(main(args.run_id)) 