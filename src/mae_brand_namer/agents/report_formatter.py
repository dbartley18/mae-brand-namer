#!/usr/bin/env python3
"""
Report Formatter

It pulls raw data from the report_raw_data table and formats it into a polished report.
"""

import os
import json
import re
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from pathlib import Path
import traceback

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.parser import parse_xml
from docx.enum.section import WD_SECTION

from src.mae_brand_namer.utils.supabase_utils import SupabaseManager
from src.mae_brand_namer.config.dependencies import Dependencies
from src.mae_brand_namer.config.settings import settings
from src.mae_brand_namer.utils.logging import get_logger
from langchain.prompts import load_prompt
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

logger = get_logger(__name__)

def _safe_load_prompt(path: str) -> PromptTemplate:
    """
    Safely load a prompt template with fallback to a basic template if loading fails.
    
    Args:
        path: Path to the prompt template file
        
    Returns:
        A PromptTemplate object
    """
    try:
        return load_prompt(path)
    except Exception as e:
        logger.error(f"Error loading prompt from {path}: {str(e)}")
        # Create a basic template as fallback
        with open(path, 'r') as f:
            content = f.read()
            # Extract input variables and template from content
            if '_type: prompt' in content and 'input_variables:' in content and 'template:' in content:
                # Parse the input variables
                vars_line = content.split('input_variables:')[1].split('\n')[0]
                try:
                    input_vars = eval(vars_line)
                except:
                    # Fallback to regex extraction
                    import re
                    match = re.search(r'input_variables:\s*\[(.*?)\]', content, re.DOTALL)
                    if match:
                        vars_text = match.group(1)
                        input_vars = [v.strip(' "\'') for v in vars_text.split(',')]
                    else:
                        input_vars = []
                
                # Extract the template
                template_content = content.split('template: |')[1].strip()
                return PromptTemplate(template=template_content, input_variables=input_vars)
        
        # If all else fails, return a minimal template
        return PromptTemplate(template="Please process the following data: {data}", 
                             input_variables=["data"])

class ReportFormatter:
    """
    Handles formatting and generation of reports using raw data from the report_raw_data table.
    This is the second step in the two-step report generation process.
    """

    # Mapping between DB section names and formatter section names
    SECTION_MAPPING = {
        # DB section name -> Formatter section name (as per notepad.md)
        "brand_context": "Brand Context",
        "brand_name_generation": "Name Generation",
        "linguistic_analysis": "Linguistic Analysis",
        "semantic_analysis": "Semantic Analysis",
        "cultural_sensitivity_analysis": "Cultural Sensitivity",
        "translation_analysis": "Translation Analysis",
        "survey_simulation": "Survey Simulation",
        "brand_name_evaluation": "Name Evaluation",
        "domain_analysis": "Domain Analysis",
        "seo_online_discoverability": "SEO Analysis",
        "competitor_analysis": "Competitor Analysis",
        "market_research": "Market Research",
        "exec_summary": "Executive Summary",
        "final_recommendations": "Strategic Recommendations"
    }

    # Reverse mapping for convenience
    REVERSE_SECTION_MAPPING = {
        # Formatter section name -> DB section name
        "brand_context": "brand_context",
        "brand_name_generation": "brand_name_generation",
        "linguistic_analysis": "linguistic_analysis",
        "semantic_analysis": "semantic_analysis", 
        "cultural_sensitivity_analysis": "cultural_sensitivity_analysis",
        "translation_analysis": "translation_analysis",
        "survey_simulation": "survey_simulation",
        "brand_name_evaluation": "brand_name_evaluation",
        "domain_analysis": "domain_analysis",
        "seo_analysis": "seo_online_discoverability",
        "competitor_analysis": "competitor_analysis",
        "market_research": "market_research",
        "executive_summary": "exec_summary",
        "recommendations": "final_recommendations"
    }

    # Default storage bucket for reports
    STORAGE_BUCKET = "agent_reports"
    
    # Report formats
    FORMAT_DOCX = "docx"
    
    # Section order for report generation (in exact order from notepad.md)
    SECTION_ORDER = [
        "exec_summary",
        "brand_context",
        "brand_name_generation",
        "linguistic_analysis",
        "semantic_analysis",
        "cultural_sensitivity_analysis",
        "translation_analysis",
        "brand_name_evaluation", 
        "domain_analysis",
        "seo_online_discoverability",
        "competitor_analysis",
        "market_research",
        "survey_simulation",
        "final_recommendations"
    ]
    
    def __init__(self, dependencies: Optional[Dependencies] = None):
        """Initialize the ReportFormatter with dependencies."""
        # Extract clients from dependencies if available
        if dependencies:
            self.supabase = dependencies.supabase
            # Safely try to get LLM from dependencies, with fallback if not available
            try:
                self.llm = dependencies.llm
                logger.info(f"Using LLM from dependencies: {type(self.llm).__name__}")
            except AttributeError:
                # Log the issue and create a new LLM instance
                logger.warning("'llm' attribute not found in Dependencies, creating new instance")
                self._initialize_llm()
        else:
            # Create clients if not provided
            self.supabase = SupabaseManager()
            self._initialize_llm()
                
        logger.info("ReportFormatter initialized successfully with LLM")
        
        # Initialize error tracking
        self.formatting_errors = {}
        self.missing_sections = set()
        
        # Initialize current run ID
        self.current_run_id = None
        
        # Create transformers mapping
        self.transformers = {
            "brand_context": self._transform_brand_context,
            "brand_name_generation": self._transform_name_generation,
            "semantic_analysis": self._transform_semantic_analysis,
            "linguistic_analysis": self._transform_linguistic_analysis,
            "cultural_sensitivity_analysis": self._transform_cultural_sensitivity,
            "brand_name_evaluation": self._transform_name_evaluation,
            "translation_analysis": self._transform_translation_analysis,
            "market_research": self._transform_market_research,
            "competitor_analysis": self._transform_competitor_analysis,
            "domain_analysis": self._transform_domain_analysis,
            "survey_simulation": self._transform_survey_simulation,
            "seo_online_discoverability": self._transform_seo_analysis,
            # Add transformers as needed
        }
        
        # Load prompts from YAML files
        try:
            self.prompts = {
                # Title page and TOC
                "title_page": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "title_page.yaml")),
                "table_of_contents": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "table_of_contents.yaml")),
                
                # Main section prompts
                "executive_summary": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "executive_summary.yaml")),
                "recommendations": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "recommendations.yaml")),
                "seo_analysis": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "seo_analysis.yaml")),
                "brand_context": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_context.yaml")),
                "brand_name_generation": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_name_generation.yaml")),
                "semantic_analysis": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "semantic_analysis.yaml")),
                "linguistic_analysis": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "linguistic_analysis.yaml")),
                "cultural_sensitivity": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "cultural_sensitivity.yaml")),
                "translation_analysis": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "translation_analysis.yaml")),
                "market_research": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "market_research.yaml")),
                "competitor_analysis": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "competitor_analysis.yaml")),
                "name_evaluation": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_name_evaluation.yaml")),
                "domain_analysis": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "domain_analysis.yaml")),
                "survey_simulation": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "survey_simulation.yaml")),
                "system": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "system.yaml")),
                "shortlisted_names_summary": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "shortlisted_names_summary.yaml")),
                "format_section": _safe_load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "format_section.yaml"))
            }
            
            logger.info("Successfully loaded report formatter prompts")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            # Create an empty dictionary as fallback
            self.prompts = {}
    
    def _initialize_llm(self):
        """Initialize the LLM with proper error handling."""
        try:
            # Verify API key is available
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key is not set in settings")
                
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=1.0,
                google_api_key=settings.gemini_api_key,
                convert_system_message_to_human=True
            )
            logger.info(f"Created new LLM instance: {type(self.llm).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise ValueError(f"Could not initialize LLM: {str(e)}")
            
    async def _safe_llm_invoke(self, messages, section_name=None, fallback_response=None):
        """Safe wrapper for LLM invocation with error handling.
        
        Args:
            messages: The messages to send to the LLM
            section_name: Optional name of the section being processed (for logging)
            fallback_response: Optional fallback response to return if invocation fails
            
        Returns:
            The LLM response or fallback response on failure
        """
        context = f" for section {section_name}" if section_name else ""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                return await self.llm.ainvoke(messages)
            except Exception as e:
                logger.error(f"Error during LLM call{context} (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All LLM retry attempts failed{context}. Using fallback.")
                    if fallback_response:
                        return fallback_response
                    
                    # Create a simple response object that mimics the LLM response structure
                    class FallbackResponse:
                        def __init__(self, content):
                            self.content = content
                    
                    error_msg = f"LLM processing failed after {max_retries} attempts. Error: {str(e)}"
                    return FallbackResponse(json.dumps({
                        "title": f"Error Processing {section_name if section_name else 'Content'}",
                        "content": error_msg,
                        "sections": [{"heading": "Error Details", "content": error_msg}]
                    }))

    async def _generate_seo_content(self, brand_name: str, seo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SEO content using the LLM."""
        try:
            # Prepare the prompt
            prompt = self.prompts["seo_analysis"].format(
                run_id=self.current_run_id,
                brand_name=brand_name,
                seo_data=json.dumps(seo_data, indent=2)
            )
            
            # Get response from LLM
            messages = [
                SystemMessage(content="You are an expert SEO analyst providing insights for brand names."),
                HumanMessage(content=prompt)
            ]
            
            response = await self._safe_llm_invoke(messages, "SEO Analysis")
            
            # Extract JSON from response
            content = response.content
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                # Try to find any JSON object in the response
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    logger.warning("Could not extract JSON from LLM response for SEO analysis")
                    return {
                        "seo_viability_assessment": "Error extracting structured content from LLM response.",
                        "search_landscape_analysis": content,
                        "content_strategy_recommendations": "",
                        "technical_seo_considerations": "",
                        "action_plan": []
                    }
                    
        except Exception as e:
            logger.error(f"Error generating SEO content: {str(e)}")
            return {
                "seo_viability_assessment": f"Error generating SEO content: {str(e)}",
                "search_landscape_analysis": "",
                "content_strategy_recommendations": "",
                "technical_seo_considerations": "",
                "action_plan": []
            }

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

    def _transform_seo_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform SEO analysis data to the expected format."""
        if not data:
            return {}
            
        if "analyses" in data and "seo_analyses" not in data:
            # Convert analyses field to seo_analyses
            data["seo_analyses"] = data.pop("analyses")
            
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
        
        # Update to use the new execute_with_retry API
        result = await self.supabase.execute_with_retry(
            "select",
            "report_raw_data",
            {
                "run_id": f"eq.{run_id}",
                "select": "section_name,raw_data",
                "order": "section_name"
            }
        )
        
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
            
            # Use the DB section name as the key directly since that's what we use in SECTION_ORDER
            # This way we maintain the exact mapping between database names and our processing
            formatter_section_name = db_section_name
            
            # Apply transformation if available
            if formatter_section_name in self.transformers:
                transformer = self.transformers[formatter_section_name]
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

    async def _add_title_page(self, doc: Document, run_id: str, brand_context: Dict[str, Any]) -> None:
        """Add a professional title page to the document."""
        # Get title page content from LLM for any customizations
        try:
            title_content = await self._safe_llm_invoke([
                SystemMessage(content="You are an expert report formatter creating a professional title page for a brand naming report."),
                HumanMessage(content=str(self.prompts["title_page"].format(
                    run_id=run_id,
                    brand_context=brand_context
                )))
            ])
            
            # Try to parse the response for any customizations
            try:
                content = json.loads(title_content.content)
            except json.JSONDecodeError:
                # Use default values if parsing fails
                content = {
                    "title": "Brand Naming Report",
                    "subtitle": "Generated by Mae Brand Naming Expert"
                }
        except Exception as e:
            logger.error(f"Error getting title page from LLM: {str(e)}")
            # Use default values if LLM fails
            content = {
                "title": "Brand Naming Report",
                "subtitle": "Generated by Mae Brand Naming Expert"
            }
        
        # Add main title - always use "Brand Naming Report" as specified in notepad
        title = doc.add_heading("Brand Naming Report", level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add subtitle - always use "Generated by Mae Brand Naming Expert" as specified in notepad
        subtitle = doc.add_paragraph()
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle.add_run("Generated by Mae Brand Naming Expert").bold = True
        
        # Add date and run ID
        metadata = doc.add_paragraph()
        metadata.alignment = WD_ALIGN_PARAGRAPH.CENTER
        metadata.add_run(f"Date: {datetime.now().strftime('%B %d, %Y')}\n")
        metadata.add_run(f"Run ID: {run_id}")
        
        # Add space 
        doc.add_paragraph()
        
        # Add note about AI generation with user prompt in the exact format specified
        user_prompt = brand_context.get("user_prompt", "Not available")
        note = doc.add_paragraph()
        note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # First line
        note_text = note.add_run(
            f"This report was generated through the Mae Brand Naming Expert Agent Simulation. "
            f"The only input provided was the initial user prompt:\n\n"
        )
        note_text.italic = True
        
        # User prompt in quotes, emphasized
        prompt_text = note.add_run(f"\"{user_prompt}\"\n\n")
        prompt_text.italic = True
        prompt_text.bold = True
        
        # Final line
        final_text = note.add_run(
            f"All additional context, analysis, and insights were autonomously generated by the AI Agent Simulation"
        )
        final_text.italic = True
        
        # Add page break
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    async def _add_table_of_contents(self, doc: Document) -> None:
        """Add a table of contents to the document."""
        # Add TOC heading
        doc.add_heading("Table of Contents", level=1)
        
        # Define the exact section list according to notepad requirements
        # Use the same section order as in generate_report but with display names
        toc_sections = [self.SECTION_MAPPING.get(section_name, section_name.replace("_", " ").title()) 
                       for section_name in self.SECTION_ORDER]
        
        # Add TOC entries with consistent styling
        for i, section in enumerate(toc_sections, 1):
            p = doc.add_paragraph(style='TOC 1')
            p.add_run(f"{i}. {section}")
            
            # Add page number placeholder (would be replaced in a real TOC)
            tab = p.add_run("\t")
            p.add_run("___")
        
        # Optional: Get any additional TOC information from LLM for section descriptions
        try:
            toc_content = await self._safe_llm_invoke([
                SystemMessage(content="You are an expert report formatter creating a professional table of contents for a brand naming report."),
                HumanMessage(content=str(self.prompts["table_of_contents"]))
            ])
            
            # Try to extract section descriptions
            try:
                content = json.loads(toc_content.content)
                if "sections" in content and isinstance(content["sections"], list):
                    # Add section descriptions if available, but keep the original TOC order
                    doc.add_paragraph()
                    doc.add_heading("Section Descriptions", level=2)
                    
                    for section in content["sections"]:
                        if "title" in section and "description" in section:
                            p = doc.add_paragraph()
                            p.add_run(f"{section['title']}: ").bold = True
                            p.add_run(section["description"])
            except json.JSONDecodeError:
                # Skip descriptions if parsing fails
                pass
        except Exception as e:
            logger.error(f"Error getting TOC descriptions from LLM: {str(e)}")
            # Continue without descriptions
        
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
        
        # Define the storage path - organize by run_id
        storage_path = f"{run_id}/{filename}"
        
        # Get file size in KB
        file_size_kb = os.path.getsize(file_path) // 1024
        
        try:
            # Read the file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Upload to Supabase Storage with proper content type
            result = await self.supabase.storage_upload_with_retry(
                bucket=self.STORAGE_BUCKET,
                path=storage_path,
                file=file_content,
                file_options={"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
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
        result = await self.supabase.execute_with_retry(
            "select",
            "report_metadata",
            {
                "run_id": f"eq.{run_id}",
                "select": "MAX(version) as current_version"
            }
        )
        current_version = 1  # Default to version 1
        
        if result and result[0]['current_version']:
            current_version = result[0]['current_version'] + 1
        
        # Insert metadata into the report_metadata table
        await self.supabase.execute_with_retry(
            "insert",
            "report_metadata",
            {
                "run_id": run_id,
                "report_url": report_url,
                "version": current_version,
                "format": format,
                "file_size_kb": file_size_kb,
                "notes": notes,
                "created_at": "NOW()"
            }
        )
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
        # Set current run ID for use in LLM prompts
        self.current_run_id = run_id
        
        # Reset error tracking for this run
        self.formatting_errors = {}
        self.missing_sections = set()  # Track missing sections
        
        # Create output directory if needed
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"reports/{run_id}"
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Report generation for {run_id} - Output in {output_dir}")
        
        # Check LLM availability - debug verbose to catch potential issues
        logger.debug(f"LLM instance: {type(self.llm).__name__} - {self.llm}")
        
        # Fetch all raw data for this run
        try:
            sections_data = await self.fetch_raw_data(run_id)
        except Exception as e:
            logger.error(f"Critical error fetching data for run_id {run_id}: {str(e)}")
            error_msg = f"Failed to fetch report data: {str(e)}"
            raise ValueError(error_msg)
        
        # Create a new document
        doc = Document()
        
        # Extract brand context for title page
        brand_context = sections_data.get("brand_context", {})
        
        # Add title page
        try:
            await self._add_title_page(doc, run_id, brand_context)
        except Exception as e:
            logger.error(f"Error adding title page: {str(e)}")
            doc.add_heading(f"Brand Name Analysis Report - {run_id}", level=0)
        
        # Add table of contents
        try:
            await self._add_table_of_contents(doc)
        except Exception as e:
            logger.error(f"Error adding table of contents: {str(e)}")
            doc.add_heading("Table of Contents", level=1)
            doc.add_paragraph("Error generating table of contents")
        
        # Track data quality issues
        data_quality_issues = {}
        
        # Process each section in order
        for section_name in self.SECTION_ORDER:
            if section_name in sections_data:
                logger.info(f"Formatting section: {section_name}")
                
                # Add section header using the proper title from SECTION_MAPPING
                section_title = self.SECTION_MAPPING.get(section_name, section_name.replace("_", " ").title())
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
                        await self._add_executive_summary(doc, sections_data)
                    elif section_name == "recommendations":
                        # Generate recommendations using LLM
                        await self._add_recommendations(doc, sections_data)
                    else:
                        # Use the _format_section method which handles different section types
                        success, error = await self._format_section(doc, section_name, sections_data)
                        if not success:
                            logger.warning(f"Section '{section_name}' had formatting issues: {error}")
                            # Don't return early, continue with other sections
                except Exception as e:
                    logger.error(f"Error processing section {section_name}: {str(e)}")
                    logger.error(traceback.format_exc())  # Add traceback for debugging
                    self._handle_section_error(doc, section_name, e)
                    # Don't return early, continue with other sections
                
                # Add divider with page break for major sections
                add_page_break = True  # Add page break after each major section
                self._add_section_divider(doc, add_page_break)
            else:
                logger.warning(f"Missing section: {section_name}")
                self.missing_sections.add(section_name)
                
                # Add a placeholder message for missing sections
                section_title = self.SECTION_MAPPING.get(section_name, section_name.replace("_", " ").title())
                heading = doc.add_heading(section_title, level=1)
                
                missing_para = doc.add_paragraph(style='Quote')
                missing_para.add_run("❓ MISSING SECTION: ").bold = True
                missing_para.add_run(f"No data was found for this section.")
                
                # Add divider with page break for major sections
                add_page_break = True  # Add page break after each major section
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
        
        # Save the document with a Supabase-friendly name format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{run_id}_{timestamp}.docx"
        doc_path = os.path.join(output_dir, filename)
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

    async def _add_executive_summary(self, doc: Document, raw_data: Dict[str, Any]) -> None:
        """
        Add the executive summary section to the document.
        Uses LLM to generate a well-formatted executive summary from the raw data.
        
        Args:
            doc: The document to add the section to
            raw_data: All the raw data for the report
        """
        try:
            logger.info("Generating executive summary")
            
            # Extract executive_summary from raw_data if it exists
            exec_summary_data = raw_data.get("executive_summary", {})
            
            # Format prompt with the current run ID
            formatted_prompt = self.prompts["executive_summary"].format(
                run_id=self.current_run_id,
                raw_data=json.dumps(raw_data, indent=2)
            )
            
            # Call LLM to generate the executive summary
            summary_content = await self._safe_llm_invoke([
                SystemMessage(content=self.prompts["system"].format()),
                HumanMessage(content=formatted_prompt)
            ])
            
            # Parse the response
            try:
                content = json.loads(summary_content.content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse executive summary response as JSON")
                # Fallback to raw response
                doc.add_paragraph(summary_content.content)
                return
                
            # Add the content to the document
            if "summary" in content:
                doc.add_paragraph(content["summary"])
                
            if "key_points" in content and content["key_points"]:
                doc.add_heading("Key Points", level=2)
                for point in content["key_points"]:
                    bullet = doc.add_paragraph(style='List Bullet')
                    bullet.add_run(point)
                    
            if "brand_context_summary" in content:
                doc.add_heading("Brand Context", level=2)
                doc.add_paragraph(content["brand_context_summary"])
                
            if "name_options_summary" in content:
                doc.add_heading("Name Options Overview", level=2)
                doc.add_paragraph(content["name_options_summary"])
                
            if "recommendations_summary" in content:
                doc.add_heading("Recommendations", level=2)
                doc.add_paragraph(content["recommendations_summary"])
                
        except Exception as e:
            logger.error(f"Error adding executive summary: {str(e)}")
            doc.add_paragraph("Error generating executive summary. Using available data instead.")
            
            # Fallback to basic formatting
            if exec_summary_data:
                for key, value in exec_summary_data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)

    async def _add_recommendations(self, doc: Document, raw_data: Dict[str, Any]) -> None:
        """
        Add the recommendations section to the document.
        Uses LLM to generate well-formatted recommendations from the raw data.
        
        Args:
            doc: The document to add the section to
            raw_data: All the raw data for the report
        """
        try:
            logger.info("Generating recommendations section")
            
            # Extract recommendations from raw_data if it exists
            recommendations_data = raw_data.get("recommendations", {})
            
            # Format prompt with the current run ID
            formatted_prompt = self.prompts["recommendations"].format(
                run_id=self.current_run_id,
                raw_data=json.dumps(raw_data, indent=2)
            )
            
            # Call LLM to generate the recommendations
            recommendations_content = await self._safe_llm_invoke([
                SystemMessage(content=self.prompts["system"].format()),
                HumanMessage(content=formatted_prompt)
            ])
            
            # Parse the response
            try:
                content = json.loads(recommendations_content.content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse recommendations response as JSON")
                # Fallback to raw response
                doc.add_paragraph(recommendations_content.content)
                return
                
            # Add the content to the document
            if "top_recommendations" in content and content["top_recommendations"]:
                doc.add_heading("Top Name Recommendations", level=2)
                for i, rec in enumerate(content["top_recommendations"], 1):
                    name = rec.get("name", f"Recommendation {i}")
                    doc.add_heading(name, level=3)
                    
                    if "rationale" in rec:
                        doc.add_paragraph(rec["rationale"])
                    
                    if "strengths" in rec and rec["strengths"]:
                        doc.add_heading("Strengths", level=4)
                        for strength in rec["strengths"]:
                            bullet = doc.add_paragraph(style='List Bullet')
                            bullet.add_run(strength)
                    
                    if "considerations" in rec and rec["considerations"]:
                        doc.add_heading("Considerations", level=4)
                        for consideration in rec["considerations"]:
                            bullet = doc.add_paragraph(style='List Bullet')
                            bullet.add_run(consideration)
            
            if "implementation_strategy" in content:
                doc.add_heading("Implementation Strategy", level=2)
                doc.add_paragraph(content["implementation_strategy"])
                
            if "alternative_options" in content and content["alternative_options"]:
                doc.add_heading("Alternative Options", level=2)
                doc.add_paragraph(content["alternative_options"])
                
            if "final_thoughts" in content:
                doc.add_heading("Final Thoughts", level=2)
                doc.add_paragraph(content["final_thoughts"])
                
        except Exception as e:
            logger.error(f"Error adding recommendations: {str(e)}")
            doc.add_paragraph("Error generating recommendations. Using available data instead.")
            
            # Fallback to basic formatting
            if recommendations_data:
                for key, value in recommendations_data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)

    async def _format_section(self, doc: Document, section_name: str, raw_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Format a specific section of the report document based on section name.
        
        Args:
            doc: The document to add content to
            section_name: Name of the section to format (database field name)
            raw_data: All the raw data for the report
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Get section-specific data
            section_data = raw_data.get(section_name, {})
            
            if not section_data:
                logger.warning(f"No data available for section: {section_name}")
                doc.add_paragraph(f"No data available for this section.", style='Quote')
                return False, "No data available"
            
            # Handle different section types with specific formatting based on database field names
            if section_name == "brand_context":
                await self._format_brand_context(doc, section_data)
            elif section_name == "brand_name_generation":
                await self._format_name_generation(doc, section_data)
            elif section_name == "linguistic_analysis":
                await self._format_linguistic_analysis(doc, section_data)
            elif section_name == "semantic_analysis":
                await self._format_semantic_analysis(doc, section_data)
            elif section_name == "cultural_sensitivity_analysis":
                await self._format_cultural_sensitivity(doc, section_data)
            elif section_name == "translation_analysis":
                await self._format_translation_analysis(doc, section_data)
            elif section_name == "brand_name_evaluation":
                await self._format_name_evaluation(doc, section_data)
            elif section_name == "domain_analysis":
                await self._format_domain_analysis(doc, section_data)
            elif section_name == "seo_online_discoverability":
                await self._format_seo_analysis(doc, section_data)
            elif section_name == "competitor_analysis":
                await self._format_competitor_analysis(doc, section_data)
            elif section_name == "market_research":
                await self._format_market_research(doc, section_data)
            elif section_name == "survey_simulation":
                await self._format_survey_simulation(doc, section_data)
            elif section_name == "exec_summary":
                await self._add_executive_summary(doc, raw_data)
            elif section_name == "final_recommendations":
                await self._add_recommendations(doc, raw_data)
            else:
                # For any other section type, use generic LLM-based formatting
                await self._format_generic_section(doc, section_name, section_data)
            
            return True, None
            
        except Exception as e:
            error_msg = f"Error formatting section {section_name}: {str(e)}"
            logger.error(error_msg)
            doc.add_paragraph(f"Error occurred while formatting this section: {str(e)}", style='Intense Quote')
            return False, error_msg
            
    async def _format_generic_section(self, doc: Document, section_name: str, data: Dict[str, Any]) -> None:
        """Format a section using LLM when no specific formatter is available."""
        try:
            # Generate a section prompt template key based on section name
            prompt_key = section_name.lower().replace(" ", "_")
            
            # Try to find a matching prompt template
            if prompt_key in self.prompts:
                prompt_template = self.prompts[prompt_key]
                logger.info(f"Using prompt template '{prompt_key}' for section {section_name}")
                
                # Format the prompt with section data and run_id
                try:
                    formatted_prompt = prompt_template.format(
                        run_id=self.current_run_id,
                        section_data=json.dumps(data, indent=2)
                    )
                    logger.debug(f"Formatted prompt for {section_name} with variables: run_id, section_data")
                except Exception as e:
                    logger.error(f"Error formatting prompt for {section_name}: {str(e)}")
                    # Try with different variable names that might be in the template
                    try:
                        # Check what variables the template expects
                        expected_vars = prompt_template.input_variables
                        logger.debug(f"Template expects variables: {expected_vars}")
                        
                        # Create a variables dict with all possible variations
                        variables = {
                            "run_id": self.current_run_id,
                            "section_data": json.dumps(data, indent=2),
                            "data": json.dumps(data, indent=2),
                            section_name: json.dumps(data, indent=2)
                        }
                        
                        # Only include variables the template expects
                        filtered_vars = {k: v for k, v in variables.items() if k in expected_vars}
                        logger.debug(f"Using filtered variables: {list(filtered_vars.keys())}")
                        
                        formatted_prompt = prompt_template.format(**filtered_vars)
                    except Exception as e2:
                        logger.error(f"Second attempt at formatting prompt failed: {str(e2)}")
                        doc.add_paragraph(f"Error formatting section: Could not prepare prompt", style='Intense Quote')
                        return
                
                # Log before LLM call
                logger.info(f"Making LLM call for section: {section_name}")
                
                # Call LLM with the formatted prompt
                try:
                    system_content = self.prompts["system"].format() if "system" in self.prompts else "You are an expert report formatter."
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=formatted_prompt)
                    ]
                    
                    # Make the LLM call with error handling
                    response = await self.llm.ainvoke(messages)
                    logger.info(f"Received LLM response for {section_name} section (length: {len(response.content) if hasattr(response, 'content') else 'unknown'})")
                    
                except Exception as e:
                    logger.error(f"Error during LLM call for {section_name}: {str(e)}")
                    logger.error(f"Error details: {traceback.format_exc()}")
                    doc.add_paragraph(f"Error generating content: LLM call failed - {str(e)}", style='Intense Quote')
                    
                    # Add basic formatting as fallback
                    self._format_generic_section_fallback(doc, section_name, data)
                    return
                
                # Try to parse the response as JSON
                try:
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Look for JSON in code blocks
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                    if json_match:
                        # Extract the content from the code block
                        json_str = json_match.group(1)
                        content = json.loads(json_str)
                        logger.debug(f"Successfully parsed JSON from code block in LLM response")
                    else:
                        # Try to parse the whole response as JSON
                        content = json.loads(content)
                        logger.debug(f"Successfully parsed entire LLM response as JSON")
                    
                    # Process the structured content
                    for key, value in content.items():
                        if isinstance(value, str) and value:
                            heading_text = key.replace("_", " ").title()
                            doc.add_heading(heading_text, level=2)
                            doc.add_paragraph(value)
                        elif isinstance(value, list) and value:
                            heading_text = key.replace("_", " ").title()
                            doc.add_heading(heading_text, level=2)
                            for item in value:
                                bullet = doc.add_paragraph(style='List Bullet')
                                if isinstance(item, str):
                                    bullet.add_run(item)
                                elif isinstance(item, dict) and "title" in item and "description" in item:
                                    bullet.add_run(f"{item['title']}: ").bold = True
                                    bullet.add_run(item["description"])
                                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response as JSON for {section_name} - using raw text")
                    # Fallback to using raw text if not valid JSON
                    raw_content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Remove any markdown code block markers
                    cleaned_content = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', raw_content)
                    
                    doc.add_paragraph(cleaned_content)
            else:
                logger.warning(f"No prompt template found for section {section_name}, using fallback formatting")
                # If no specific prompt template is found, use a generic approach
                self._format_generic_section_fallback(doc, section_name, data)
                    
        except Exception as e:
            logger.error(f"Error in generic section formatting for {section_name}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            doc.add_paragraph(f"Error formatting section: {str(e)}", style='Intense Quote')
    
    def _format_generic_section_fallback(self, doc: Document, section_name: str, data: Dict[str, Any]) -> None:
        """Fallback method for formatting a section when LLM or other approaches fail."""
        try:
            doc.add_paragraph(f"Section data:", style='Quote')
            
            # Add section data in a readable format
            for key, value in data.items():
                if isinstance(value, str) and value:
                    doc.add_heading(key.replace("_", " ").title(), level=2)
                    doc.add_paragraph(value)
                elif isinstance(value, list) and value:
                    doc.add_heading(key.replace("_", " ").title(), level=2)
                    for item in value:
                        if isinstance(item, str):
                            bullet = doc.add_paragraph(style='List Bullet')
                            bullet.add_run(str(item))
                        elif isinstance(item, dict):
                            # Create a subheading for each item if it has a name or title
                            item_title = item.get("name", item.get("title", item.get("brand_name", f"Item {value.index(item) + 1}")))
                            doc.add_heading(item_title, level=3)
                            
                            # Add item details
                            for sub_key, sub_value in item.items():
                                if sub_key not in ["name", "title", "brand_name"] and sub_value:
                                    p = doc.add_paragraph()
                                    p.add_run(f"{sub_key.replace('_', ' ').title()}: ").bold = True
                                    p.add_run(str(sub_value))
        except Exception as e:
            logger.error(f"Error in fallback formatting for {section_name}: {str(e)}")
            doc.add_paragraph(f"Error in fallback formatting: {str(e)}", style='Intense Quote')

    async def _format_survey_simulation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the survey simulation section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section presents simulated market research findings based on "
                "survey responses. The analysis includes market receptivity, participant "
                "feedback, and recommendations based on survey data."
            )
            
            # Process survey simulations
            if "survey_simulations" in data and isinstance(data["survey_simulations"], list):
                simulations = data["survey_simulations"]
                
                for simulation in simulations:
                    # Add a heading for each brand name
                    if "brand_name" in simulation:
                        doc.add_heading(simulation["brand_name"], level=2)
                        
                        # Process name analysis if available
                        name_analysis = simulation.get("name_analysis", {})
                        
                        # Add market receptivity
                        if "market_receptivity" in name_analysis:
                            doc.add_heading("Market Adoption Potential", level=3)
                            doc.add_paragraph(name_analysis["market_receptivity"])
                        
                        # Add participant feedback
                        if "participant_feedback" in name_analysis:
                            doc.add_heading("Qualitative Feedback", level=3)
                            doc.add_paragraph(name_analysis["participant_feedback"])
                        
                        # Add final recommendations (if available) or fall back to recommendations
                        if "final_recommendations" in name_analysis:
                            doc.add_heading("Final Recommendations", level=3)
                            doc.add_paragraph(name_analysis["final_recommendations"])
                        elif "recommendations" in name_analysis:
                            doc.add_heading("Recommendations", level=3)
                            doc.add_paragraph(name_analysis["recommendations"])
                        
                        # Add survey methodology
                        if "survey_methodology" in simulation:
                            doc.add_heading("Survey Methodology", level=3)
                            doc.add_paragraph(simulation["survey_methodology"])
                            
            # If no structured data, try to use the raw data
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting survey simulation: {str(e)}")
            doc.add_paragraph(f"Error formatting survey simulation section: {str(e)}", style='Intense Quote')
    
    async def _format_linguistic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the linguistic analysis section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section presents linguistic analysis findings based on "
                "the analysis of brand names and their associated linguistic features."
            )
            
            # Process linguistic analyses
            if "linguistic_analyses" in data and isinstance(data["linguistic_analyses"], list):
                analyses = data["linguistic_analyses"]
                
                for analysis in analyses:
                    # Add a heading for each brand name
                    if "brand_name" in analysis:
                        doc.add_heading(analysis["brand_name"], level=2)
                        
                        # Process linguistic features
                        if "features" in analysis:
                            doc.add_heading("Linguistic Features", level=3)
                            for feature in analysis["features"]:
                                bullet = doc.add_paragraph(style='List Bullet')
                                bullet.add_run(feature)
            else:
                # If no structured data, try to use the raw data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting linguistic analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting linguistic analysis section: {str(e)}", style='Intense Quote')

    async def _format_cultural_sensitivity(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format cultural sensitivity analysis section."""
        try:
            # Get the LLM to format this section
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert report formatter creating a professional section on cultural sensitivity analysis for brand names."),
                HumanMessage(content=str(self.prompts["format_section"].format(
                    section_name="Cultural Sensitivity Analysis",
                    section_data=data
                )))
            ])
            
            # Parse the response
            try:
                content = json.loads(response.content)
                
                # Add the formatted content
                doc.add_heading(content.get("title", "Cultural Sensitivity Analysis"), level=1)
                
                # Add introduction
                if "introduction" in content:
                    doc.add_paragraph(content["introduction"])
                
                # Add main content sections
                for section in content.get("sections", []):
                    if "heading" in section:
                        doc.add_heading(section["heading"], level=2)
                    
                    if "content" in section:
                        if isinstance(section["content"], list):
                            for para in section["content"]:
                                doc.add_paragraph(para)
                        else:
                            doc.add_paragraph(section["content"])
                    
                    # Add any bullet points
                    if "bullet_points" in section:
                        for point in section["bullet_points"]:
                            p = doc.add_paragraph(style='List Bullet')
                            p.add_run(point)
                
                # Add conclusion
                if "conclusion" in content:
                    doc.add_heading("Conclusion", level=2)
                    doc.add_paragraph(content["conclusion"])
                    
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                doc.add_paragraph("Cultural sensitivity analysis data could not be properly formatted.")
                doc.add_paragraph(response.content)
                
        except Exception as e:
            logger.error(f"Error formatting cultural sensitivity section: {str(e)}")
            doc.add_paragraph(f"Error occurred while formatting cultural sensitivity section: {str(e)}", style='Intense Quote')

    async def _format_name_evaluation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format brand name evaluation section."""
        try:
            # Get the LLM to format this section
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert report formatter creating a professional section on brand name evaluation."),
                HumanMessage(content=str(self.prompts["format_section"].format(
                    section_name="Brand Name Evaluation",
                    section_data=data
                )))
            ])
            
            # Parse the response
            try:
                content = json.loads(response.content)
                
                # Add the formatted content
                doc.add_heading(content.get("title", "Brand Name Evaluation"), level=1)
                
                # Add introduction
                if "introduction" in content:
                    doc.add_paragraph(content["introduction"])
                
                # Add main content sections
                for section in content.get("sections", []):
                    if "heading" in section:
                        doc.add_heading(section["heading"], level=2)
                    
                    if "content" in section:
                        if isinstance(section["content"], list):
                            for para in section["content"]:
                                doc.add_paragraph(para)
                        else:
                            doc.add_paragraph(section["content"])
                    
                    # Add any bullet points
                    if "bullet_points" in section:
                        for point in section["bullet_points"]:
                            p = doc.add_paragraph(style='List Bullet')
                            p.add_run(point)
                
                # Add conclusion
                if "conclusion" in content:
                    doc.add_heading("Conclusion", level=2)
                    doc.add_paragraph(content["conclusion"])
                    
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                doc.add_paragraph("Brand name evaluation data could not be properly formatted.")
                doc.add_paragraph(response.content)
                
        except Exception as e:
            logger.error(f"Error formatting brand name evaluation section: {str(e)}")
            doc.add_paragraph(f"Error occurred while formatting brand name evaluation section: {str(e)}", style='Intense Quote')

    async def _format_seo_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format SEO analysis section."""
        try:
            # Get the LLM to format this section
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert report formatter creating a professional section on SEO and online discoverability analysis for brand names."),
                HumanMessage(content=str(self.prompts["format_section"].format(
                    section_name="SEO Analysis",
                    section_data=data
                )))
            ])
            
            # Parse the response
            try:
                content = json.loads(response.content)
                
                # Add the formatted content
                doc.add_heading(content.get("title", "SEO and Online Discoverability Analysis"), level=1)
                
                # Add introduction
                if "introduction" in content:
                    doc.add_paragraph(content["introduction"])
                
                # Add main content sections
                for section in content.get("sections", []):
                    if "heading" in section:
                        doc.add_heading(section["heading"], level=2)
                    
                    if "content" in section:
                        if isinstance(section["content"], list):
                            for para in section["content"]:
                                doc.add_paragraph(para)
                        else:
                            doc.add_paragraph(section["content"])
                    
                    # Add any bullet points
                    if "bullet_points" in section:
                        for point in section["bullet_points"]:
                            p = doc.add_paragraph(style='List Bullet')
                            p.add_run(point)
                
                # Add conclusion
                if "conclusion" in content:
                    doc.add_heading("Conclusion", level=2)
                    doc.add_paragraph(content["conclusion"])
                    
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                doc.add_paragraph("SEO analysis data could not be properly formatted.")
                doc.add_paragraph(response.content)
                
        except Exception as e:
            logger.error(f"Error formatting SEO analysis section: {str(e)}")
            doc.add_paragraph(f"Error occurred while formatting SEO analysis section: {str(e)}", style='Intense Quote')

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