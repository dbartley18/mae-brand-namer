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
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from pathlib import Path
import traceback
import time

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.parser import parse_xml
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

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
    
    def __init__(self, run_id: str, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the ReportFormatter with dependencies."""
        # Initialize Supabase client
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        # Always initialize our own LLM (there is no LLM in Dependencies)
        self._initialize_llm()
        logger.info("ReportFormatter initialized successfully with LLM")
        
        # Initialize error tracking
        self.formatting_errors = {}
        self.missing_sections = set()
        
        # Initialize current run ID
        self.current_run_id = run_id
        
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
        
        logger.info("Initializing ReportFormatter for run_id: %s", run_id)
        
        # Load prompts from YAML files
        try:
            logger.info("Loading report formatter prompts")
            
            prompt_paths = {
                # Title page and TOC
                "title_page": str(Path(__file__).parent / "prompts" / "report_formatter" / "title_page.yaml"),
                "table_of_contents": str(Path(__file__).parent / "prompts" / "report_formatter" / "table_of_contents.yaml"),
                
                # Main section prompts
                "executive_summary": str(Path(__file__).parent / "prompts" / "report_formatter" / "executive_summary.yaml"),
                "recommendations": str(Path(__file__).parent / "prompts" / "report_formatter" / "recommendations.yaml"),
                "seo_analysis": str(Path(__file__).parent / "prompts" / "report_formatter" / "seo_analysis.yaml"),
                "brand_context": str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_context.yaml"),
                "brand_name_generation": str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_name_generation.yaml"),
                "semantic_analysis": str(Path(__file__).parent / "prompts" / "report_formatter" / "semantic_analysis.yaml"),
                "linguistic_analysis": str(Path(__file__).parent / "prompts" / "report_formatter" / "linguistic_analysis.yaml"),
                "cultural_sensitivity": str(Path(__file__).parent / "prompts" / "report_formatter" / "cultural_sensitivity.yaml"),
                "translation_analysis": str(Path(__file__).parent / "prompts" / "report_formatter" / "translation_analysis.yaml"),
                "market_research": str(Path(__file__).parent / "prompts" / "report_formatter" / "market_research.yaml"),
                "competitor_analysis": str(Path(__file__).parent / "prompts" / "report_formatter" / "competitor_analysis.yaml"),
                "name_evaluation": str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_name_evaluation.yaml"),
                "domain_analysis": str(Path(__file__).parent / "prompts" / "report_formatter" / "domain_analysis.yaml"),
                "survey_simulation": str(Path(__file__).parent / "prompts" / "report_formatter" / "survey_simulation.yaml"),
                "system": str(Path(__file__).parent / "prompts" / "report_formatter" / "system.yaml"),
                "shortlisted_names_summary": str(Path(__file__).parent / "prompts" / "report_formatter" / "shortlisted_names_summary.yaml"),
                "format_section": str(Path(__file__).parent / "prompts" / "report_formatter" / "format_section.yaml")
            }
            
            # Create format_section fallback in case it doesn't load
            format_section_fallback = PromptTemplate.from_template(
                "Format the section '{section_name}' based on this data:\n"
                "```\n{section_data}\n```\n\n"
                "Include a title, main content, and sections with headings."
            )
            
            self.prompts = {}
            
            # Load each prompt with basic error handling
            for prompt_name, prompt_path in prompt_paths.items():
                try:
                    self.prompts[prompt_name] = _safe_load_prompt(prompt_path)
                    logger.debug(f"Loaded prompt: {prompt_name}")
                except Exception as e:
                    logger.warning(f"Could not load prompt {prompt_name}: {str(e)}")
                    # Only create fallback for format_section as it's critical
                    if prompt_name == "format_section":
                        self.prompts[prompt_name] = format_section_fallback
                        logger.info("Using fallback template for format_section")
            
            # Verify format_section prompt is available
            if "format_section" not in self.prompts:
                logger.warning("Critical prompt 'format_section' not found, using fallback")
                self.prompts["format_section"] = format_section_fallback
                
            logger.info(f"Loaded {len(self.prompts)} prompts")
                
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            # Minimum fallback prompts
            self.prompts = {
                "format_section": format_section_fallback
            }
    
    def _initialize_llm(self):
        """Initialize the LLM with Google Generative AI."""
        try:
            # Create LLM instance
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=settings.model_temperature,
                google_api_key=settings.gemini_api_key,
                convert_system_message_to_human=True,
                callbacks=settings.get_langsmith_callbacks()
            )
            logger.info(f"Initialized LLM: {type(self.llm).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise ValueError(f"Could not initialize LLM: {str(e)}")
            
    def _format_template(self, template_name: str, format_data: Dict[str, Any], section_name: str = None) -> str:
        """Helper method to standardize template formatting across the class.
        
        Args:
            template_name: Name of the template in self.prompts
            format_data: Dictionary of variables to format the template with
            section_name: Optional section name for logging
            
        Returns:
            Formatted template string
        """
        context = f" for {section_name}" if section_name else ""
        logger.debug(f"Formatting template {template_name}{context}")
        
        # Get template content
        template_content = ""
        if template_name in self.prompts:
            if hasattr(self.prompts[template_name], "template"):
                template_content = self.prompts[template_name].template
            else:
                # Try to access as dictionary
                template_content = self.prompts[template_name].get("template", "")
        
        if not template_content:
            logger.warning(f"Template {template_name} not found or empty{context}")
            return f"Please format the following data: {format_data.get('data', '')}"
        
        # Format the template with data
        formatted_template = template_content
        original_template = template_content
        
        # Try Jinja2-style placeholders first
        for key, value in format_data.items():
            placeholder = "{{" + key + "}}"
            if placeholder in formatted_template:
                formatted_template = formatted_template.replace(placeholder, str(value))
                logger.debug(f"Replaced placeholder '{placeholder}' in template {template_name}{context}")
        
        # Then try Python-style placeholders
        for key, value in format_data.items():
            placeholder = "{" + key + "}"
            if placeholder in formatted_template:
                formatted_template = formatted_template.replace(placeholder, str(value))
                logger.debug(f"Replaced placeholder '{placeholder}' in template {template_name}{context}")
        
        # Check if any replacements were made
        if original_template == formatted_template:
            logger.warning(f"No placeholders were replaced in {template_name} template{context}!")
        else:
            logger.info(f"Template placeholders were successfully replaced for {template_name}{context}")
            
        return formatted_template
    
    def _get_system_content(self, fallback: str = None) -> str:
        """Get system prompt content with fallback.
        
        Args:
            fallback: Fallback content if system prompt not found
            
        Returns:
            System prompt content
        """
        if "system" in self.prompts:
            if hasattr(self.prompts["system"], "template"):
                return self.prompts["system"].template
            else:
                return self.prompts["system"].get("template", "")
        return fallback or "You are an expert report formatter."
            
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
        
        try:
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
            
            logger.info(f"Query result for run_id {run_id}: {len(result) if result else 0} rows found")
            
            if not result or len(result) == 0:
                logger.warning(f"No data found for run_id: {run_id}")
                return {}
            
            # Debug log to see what's in the result
            section_names = [row['section_name'] for row in result]
            logger.info(f"Found sections: {section_names}")
            
            # Track data quality issues
            data_quality_issues = {}
            
            # Transform results into a dictionary with section_name as keys
            sections_data = {}
            for row in result:
                db_section_name = row['section_name']
                raw_data = row['raw_data']
                
                # Debug each row
                logger.debug(f"Processing row for section: {db_section_name} (data length: {len(str(raw_data))})")
                
                # Use the DB section name as the key directly since that's what we use in SECTION_ORDER
                # This way we maintain the exact mapping between database names and our processing
                formatter_section_name = db_section_name
                
                # Make sure raw_data is a properly formatted dict
                if isinstance(raw_data, str):
                    try:
                        raw_data = json.loads(raw_data)
                        logger.debug(f"Converted string raw_data to dict for {formatter_section_name}")
                    except json.JSONDecodeError:
                        logger.warning(f"Raw data for {formatter_section_name} is a string but not valid JSON")
                
                if not isinstance(raw_data, dict) and not isinstance(raw_data, list):
                    logger.warning(f"Raw data for {formatter_section_name} is not a dict or list: {type(raw_data)}")
                    # Convert to dict to avoid crashes
                    raw_data = {"raw_content": str(raw_data)}
                
                # Apply transformation if available
                if formatter_section_name in self.transformers:
                    transformer = self.transformers[formatter_section_name]
                    try:
                        logger.info(f"Applying transformer for {formatter_section_name}")
                        transformed_data = transformer(raw_data)
                        
                        # Validate transformed data
                        is_valid, issues = self._validate_section_data(formatter_section_name, transformed_data)
                        if not is_valid or issues:
                            for issue in issues:
                                logger.warning(f"Data quality issue in {formatter_section_name}: {issue}")
                            if issues:
                                data_quality_issues[formatter_section_name] = issues
                        
                        sections_data[formatter_section_name] = transformed_data
                        logger.info(f"Successfully transformed data for {formatter_section_name}")
                    except Exception as e:
                        logger.error(f"Error transforming data for section {formatter_section_name}: {str(e)}")
                        logger.error(f"Transformation error details: {traceback.format_exc()}")
                        logger.debug(f"Raw data for failed transformation: {str(raw_data)[:500]}...")
                        # Store error but continue with other sections
                        data_quality_issues[formatter_section_name] = [f"Transformation error: {str(e)}"]
                        # Still store raw data for potential fallback handling
                        sections_data[formatter_section_name] = raw_data
                else:
                    # Use raw data if no transformer is available
                    logger.info(f"No transformer available for {formatter_section_name}, using raw data")
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
                
            logger.info(f"Successfully processed {len(sections_data)} sections for run_id: {run_id}")
            
            # Log the keys to verify what we're returning
            logger.info(f"Returning data for sections: {sorted(list(sections_data.keys()))}")
            
            # Check for expected sections
            missing_sections = [section for section in self.SECTION_ORDER if section not in sections_data]
            if missing_sections:
                logger.warning(f"Missing expected sections: {missing_sections}")
            
            return sections_data
            
        except Exception as e:
            logger.error(f"Error fetching raw data for run_id {run_id}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return {}

    def _setup_document_styles(self, doc: Document) -> None:
        """Set up document styles."""
        try:
            # Add styles
            styles = doc.styles
            
            # Add a style for bullet points
            if 'List Bullet' not in styles:
                bullet_style = styles.add_style('List Bullet', 1)
                bullet_style.base_style = styles['Normal']
                bullet_style.font.size = Pt(11)
                bullet_style.paragraph_format.left_indent = Inches(0.5)
            
            # Add a style for quotes
            if 'Intense Quote' not in styles:
                quote_style = styles.add_style('Intense Quote', 1)
                quote_style.base_style = styles['Normal']
                quote_style.font.size = Pt(11)
                quote_style.font.italic = True
                quote_style.paragraph_format.left_indent = Inches(0.5)
                quote_style.paragraph_format.right_indent = Inches(0.5)
            
            # Add a style for code blocks
            if 'Code Block' not in styles:
                code_style = styles.add_style('Code Block', 1)
                code_style.base_style = styles['Normal']
                code_style.font.name = 'Courier New'
                code_style.font.size = Pt(10)
                code_style.paragraph_format.left_indent = Inches(0.5)
                code_style.paragraph_format.right_indent = Inches(0.5)
            
            # Add a style for tables
            if 'Table Grid' not in styles:
                table_style = styles.add_style('Table Grid', 1)
                table_style.base_style = styles['Table Grid']
                table_style.font.size = Pt(10)
                table_style.paragraph_format.space_before = Pt(6)
                table_style.paragraph_format.space_after = Pt(6)
                
            # Add a style for TOC entries
            if 'TOC 1' not in styles:
                toc_style = styles.add_style('TOC 1', 1)
                toc_style.base_style = styles['Normal']
                toc_style.font.size = Pt(12)
                toc_style.paragraph_format.left_indent = Inches(0.25)
                toc_style.paragraph_format.space_after = Pt(6)
                
        except Exception as e:
            logger.error(f"Error setting up document styles: {str(e)}")
            # Add a basic error message to the document
            doc.add_paragraph("Error setting up document styles. Some formatting may be incorrect.", style='Intense Quote')
    
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
            response = await self._safe_llm_invoke([
                SystemMessage(content="You are an expert report formatter creating a professional section on cultural sensitivity analysis for brand names."),
                HumanMessage(content=str(self.prompts["format_section"].format(
                    section_name="Cultural Sensitivity Analysis",
                    section_data=data
                )))
            ], section_name="Cultural Sensitivity Analysis")
            
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
            response = await self._safe_llm_invoke([
                SystemMessage(content="You are an expert report formatter creating a professional section on brand name evaluation."),
                HumanMessage(content=str(self.prompts["format_section"].format(
                    section_name="Brand Name Evaluation",
                    section_data=data
                )))
            ], section_name="Brand Name Evaluation")
            
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
            response = await self._safe_llm_invoke([
                SystemMessage(content="You are an expert report formatter creating a professional section on SEO and online discoverability analysis for brand names."),
                HumanMessage(content=str(self.prompts["format_section"].format(
                    section_name="SEO Analysis",
                    section_data=data
                )))
            ], section_name="SEO Analysis")
            
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

    async def _format_brand_context(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the brand context section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section provides the foundational context for the brand naming process, "
                "including information about the company, industry, target audience, and brand values."
            )
            
            # Extract brand context information
            brand_context = data.get("brand_context", data)  # Handle both wrapped and unwrapped data
            
            if isinstance(brand_context, dict):
                # Add company information
                if "company_name" in brand_context or "company_description" in brand_context:
                    doc.add_heading("Company Information", level=2)
                    
                    if "company_name" in brand_context:
                        p = doc.add_paragraph()
                        p.add_run("Company Name: ").bold = True
                        p.add_run(brand_context["company_name"])
                        
                    if "company_description" in brand_context:
                        doc.add_paragraph(brand_context["company_description"])
                
                # Add industry information
                if "industry" in brand_context or "industry_description" in brand_context:
                    doc.add_heading("Industry Context", level=2)
                    
                    if "industry" in brand_context:
                        p = doc.add_paragraph()
                        p.add_run("Industry: ").bold = True
                        p.add_run(brand_context["industry"])
                        
                    if "industry_description" in brand_context:
                        doc.add_paragraph(brand_context["industry_description"])
                
                # Add target audience
                if "target_audience" in brand_context:
                    doc.add_heading("Target Audience", level=2)
                    doc.add_paragraph(brand_context["target_audience"])
                
                # Add brand values
                if "brand_values" in brand_context:
                    doc.add_heading("Brand Values", level=2)
                    values = brand_context["brand_values"]
                    if isinstance(values, list):
                        for value in values:
                            bullet = doc.add_paragraph(style='List Bullet')
                            bullet.add_run(value)
                    else:
                        doc.add_paragraph(values)
                
                # Add user prompt - access from state instead of brand_context
                if "state" in data and "user_prompt" in data["state"]:
                    doc.add_heading("Original User Prompt", level=2)
                    doc.add_paragraph(data["state"]["user_prompt"])
                
                # Add any other fields
                for key, value in brand_context.items():
                    if key not in ["company_name", "company_description", "industry", 
                                  "industry_description", "target_audience", "brand_values", 
                                  "user_prompt"] and isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
            else:
                # Fallback for non-dict data
                if isinstance(brand_context, str):
                    doc.add_paragraph(brand_context)
                else:
                    doc.add_paragraph("Brand context data is not in expected format.")
                    
        except Exception as e:
            logger.error(f"Error formatting brand context: {str(e)}")
            doc.add_paragraph(f"Error formatting brand context section: {str(e)}", style='Intense Quote')

    async def _format_name_generation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name generation section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section presents the generated brand name options categorized by naming approach. "
                "Each name includes its rationale and relevant characteristics."
            )
            
            # Process name generations
            if "name_generations" in data and isinstance(data["name_generations"], list):
                # Group names by category
                categories = {}
                for name in data["name_generations"]:
                    if "brand_name" in name:
                        category = name.get("naming_category", "General")
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(name)
                
                # Add each category and its names
                for category, names in categories.items():
                    doc.add_heading(category, level=2)
                    
                    for name in names:
                        # Add name as subheading
                        doc.add_heading(name["brand_name"], level=3)
                        
                        # Add rationale if available
                        if "rationale" in name:
                            doc.add_paragraph(name["rationale"])
                        
                        # Add additional info in a structured format
                        for key, value in name.items():
                            if key not in ["brand_name", "rationale", "naming_category"] and value:
                                p = doc.add_paragraph()
                                p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                                p.add_run(str(value))
            else:
                # Fallback for unstructured data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting name generation: {str(e)}")
            doc.add_paragraph(f"Error formatting name generation section: {str(e)}", style='Intense Quote')

    async def _format_semantic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the semantic analysis section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the semantic aspects of the brand name options, "
                "including meaning associations, connotations, and semantic fields."
            )
            
            # Process semantic analyses
            if "semantic_analyses" in data and isinstance(data["semantic_analyses"], list):
                analyses = data["semantic_analyses"]
                
                for analysis in analyses:
                    # Add a heading for each brand name
                    if "brand_name" in analysis:
                        doc.add_heading(analysis["brand_name"], level=2)
                        
                        # Add meaning analysis
                        if "meaning" in analysis:
                            doc.add_heading("Meaning Analysis", level=3)
                            doc.add_paragraph(analysis["meaning"])
                        
                        # Add connotations
                        if "connotations" in analysis:
                            doc.add_heading("Connotations", level=3)
                            connotations = analysis["connotations"]
                            if isinstance(connotations, list):
                                for connotation in connotations:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(connotation)
                            else:
                                doc.add_paragraph(connotations)
                        
                        # Add semantic fields
                        if "semantic_fields" in analysis:
                            doc.add_heading("Semantic Fields", level=3)
                            fields = analysis["semantic_fields"]
                            if isinstance(fields, list):
                                for field in fields:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(field)
                            else:
                                doc.add_paragraph(fields)
                        
                        # Add other analyses
                        for key, value in analysis.items():
                            if key not in ["brand_name", "meaning", "connotations", "semantic_fields"] and value:
                                doc.add_heading(key.replace("_", " ").title(), level=3)
                                if isinstance(value, list):
                                    for item in value:
                                        bullet = doc.add_paragraph(style='List Bullet')
                                        bullet.add_run(str(item))
                                else:
                                    doc.add_paragraph(str(value))
            else:
                # Fallback for unstructured data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting semantic analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting semantic analysis section: {str(e)}", style='Intense Quote')

    async def _format_translation_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the translation analysis section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section examines how the brand name options translate across different languages, "
                "identifying potential issues or advantages in global contexts."
            )
            
            # Process translation analyses
            if "translation_analyses" in data and isinstance(data["translation_analyses"], list):
                analyses = data["translation_analyses"]
                
                for analysis in analyses:
                    # Add a heading for each brand name
                    if "brand_name" in analysis:
                        doc.add_heading(analysis["brand_name"], level=2)
                        
                        # Add translations
                        if "translations" in analysis and isinstance(analysis["translations"], list):
                            doc.add_heading("Translations", level=3)
                            
                            # Create a table for translations
                            table = doc.add_table(rows=1, cols=2)
                            table.style = 'Table Grid'
                            
                            # Add headers
                            header_cells = table.rows[0].cells
                            header_cells[0].text = "Language"
                            header_cells[1].text = "Translation"
                            
                            for cell in header_cells:
                                for paragraph in cell.paragraphs:
                                    for run in paragraph.runs:
                                        run.font.bold = True
                            
                            # Add data rows
                            for translation in analysis["translations"]:
                                if "language" in translation and "text" in translation:
                                    row = table.add_row().cells
                                    row[0].text = translation["language"]
                                    row[1].text = translation["text"]
                        
                        # Add cultural considerations
                        if "cultural_considerations" in analysis:
                            doc.add_heading("Cultural Considerations", level=3)
                            considerations = analysis["cultural_considerations"]
                            if isinstance(considerations, list):
                                for consideration in considerations:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(consideration)
                            else:
                                doc.add_paragraph(considerations)
                        
                        # Add recommendations
                        if "recommendations" in analysis:
                            doc.add_heading("Recommendations", level=3)
                            doc.add_paragraph(analysis["recommendations"])
            else:
                # Fallback for unstructured data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting translation analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting translation analysis section: {str(e)}", style='Intense Quote')

    async def _format_domain_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the domain analysis section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section evaluates the domain name availability and alternatives for each brand name option, "
                "providing insights into online presence opportunities."
            )
            
            # Process domain analyses
            if "domain_analyses" in data and isinstance(data["domain_analyses"], list):
                analyses = data["domain_analyses"]
                
                for analysis in analyses:
                    # Add a heading for each brand name
                    if "brand_name" in analysis:
                        doc.add_heading(analysis["brand_name"], level=2)
                        
                        # Add primary domain availability
                        if "primary_domain" in analysis:
                            doc.add_heading("Primary Domain", level=3)
                            primary = analysis["primary_domain"]
                            p = doc.add_paragraph()
                            p.add_run(f"{primary.get('domain', '')}: ").bold = True
                            
                            # Format availability with color
                            availability = primary.get('available', False)
                            if availability:
                                run = p.add_run("Available")
                                run.font.color.rgb = RGBColor(0, 128, 0)  # Green
                            else:
                                run = p.add_run("Not Available")
                                run.font.color.rgb = RGBColor(255, 0, 0)  # Red
                            
                            # Add price if available
                            if "price" in primary:
                                p.add_run(f" (Price: {primary['price']})")
                        
                        # Add alternative domains
                        if "alternative_domains" in analysis and isinstance(analysis["alternative_domains"], list):
                            doc.add_heading("Alternative Domains", level=3)
                            
                            for domain in analysis["alternative_domains"]:
                                p = doc.add_paragraph(style='List Bullet')
                                p.add_run(f"{domain.get('domain', '')}: ").bold = True
                                
                                # Format availability with color
                                availability = domain.get('available', False)
                                if availability:
                                    run = p.add_run("Available")
                                    run.font.color.rgb = RGBColor(0, 128, 0)  # Green
                                else:
                                    run = p.add_run("Not Available")
                                    run.font.color.rgb = RGBColor(255, 0, 0)  # Red
                                
                                # Add price if available
                                if "price" in domain:
                                    p.add_run(f" (Price: {domain['price']})")
                        
                        # Add recommendations
                        if "recommendations" in analysis:
                            doc.add_heading("Domain Recommendations", level=3)
                            doc.add_paragraph(analysis["recommendations"])
            else:
                # Fallback for unstructured data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting domain analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting domain analysis section: {str(e)}", style='Intense Quote')

    async def _format_competitor_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the competitor analysis section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section analyzes competitors' brand names to provide context and differentiation "
                "strategies for the proposed brand name options."
            )
            
            # Process competitor analyses
            if "competitor_analyses" in data and isinstance(data["competitor_analyses"], list):
                analyses = data["competitor_analyses"]
                
                # Add overview first
                doc.add_heading("Competitive Landscape Overview", level=2)
                overview_added = False
                
                for analysis in analyses:
                    if "overview" in analysis:
                        doc.add_paragraph(analysis["overview"])
                        overview_added = True
                        break
                
                if not overview_added:
                    doc.add_paragraph("No overview information available.")
                
                # Add individual competitor analyses
                doc.add_heading("Competitor Brand Names", level=2)
                
                for analysis in analyses:
                    if "competitor_name" in analysis:
                        # Add a heading for each competitor
                        doc.add_heading(analysis["competitor_name"], level=3)
                        
                        # Add brand name analysis
                        if "brand_name_analysis" in analysis:
                            doc.add_paragraph(analysis["brand_name_analysis"])
                        
                        # Add strengths
                        if "strengths" in analysis:
                            doc.add_heading("Strengths", level=4)
                            strengths = analysis["strengths"]
                            if isinstance(strengths, list):
                                for strength in strengths:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(strength)
                            else:
                                doc.add_paragraph(strengths)
                        
                        # Add weaknesses
                        if "weaknesses" in analysis:
                            doc.add_heading("Weaknesses", level=4)
                            weaknesses = analysis["weaknesses"]
                            if isinstance(weaknesses, list):
                                for weakness in weaknesses:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(weakness)
                            else:
                                doc.add_paragraph(weaknesses)
                
                # Add differentiation strategy
                doc.add_heading("Differentiation Strategy", level=2)
                strategy_added = False
                
                for analysis in analyses:
                    if "differentiation_strategy" in analysis:
                        doc.add_paragraph(analysis["differentiation_strategy"])
                        strategy_added = True
                        break
                
                if not strategy_added:
                    doc.add_paragraph("No differentiation strategy information available.")
            else:
                # Fallback for unstructured data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting competitor analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting competitor analysis section: {str(e)}", style='Intense Quote')

    async def _format_market_research(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the market research section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section presents market research findings related to the brand naming process, "
                "including target audience insights and market trends."
            )
            
            # Process market research data
            if "market_researches" in data and isinstance(data["market_researches"], list):
                researches = data["market_researches"]
                
                # Add overview first
                doc.add_heading("Market Research Overview", level=2)
                overview_added = False
                
                for research in researches:
                    if "overview" in research:
                        doc.add_paragraph(research["overview"])
                        overview_added = True
                        break
                
                if not overview_added:
                    doc.add_paragraph("No overview information available.")
                
                # Add target audience insights
                doc.add_heading("Target Audience Insights", level=2)
                insights_added = False
                
                for research in researches:
                    if "target_audience_insights" in research:
                        doc.add_paragraph(research["target_audience_insights"])
                        insights_added = True
                        break
                
                if not insights_added:
                    doc.add_paragraph("No target audience insights available.")
                
                # Add market trends
                doc.add_heading("Market Trends", level=2)
                trends_added = False
                
                for research in researches:
                    if "market_trends" in research:
                        trends = research["market_trends"]
                        if isinstance(trends, list):
                            for trend in trends:
                                bullet = doc.add_paragraph(style='List Bullet')
                                bullet.add_run(trend)
                        else:
                            doc.add_paragraph(trends)
                        trends_added = True
                        break
                
                if not trends_added:
                    doc.add_paragraph("No market trends information available.")
                
                # Add implications for naming
                doc.add_heading("Implications for Brand Naming", level=2)
                implications_added = False
                
                for research in researches:
                    if "implications_for_naming" in research:
                        doc.add_paragraph(research["implications_for_naming"])
                        implications_added = True
                        break
                
                if not implications_added:
                    doc.add_paragraph("No implications for naming information available.")
            else:
                # Fallback for unstructured data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                        
        except Exception as e:
            logger.error(f"Error formatting market research: {str(e)}")
            doc.add_paragraph(f"Error formatting market research section: {str(e)}", style='Intense Quote')

    async def generate_report(self, run_id: str, upload_to_storage: bool = True) -> str:
        """
        Generate a formatted report document for the specified run_id.
        
        Args:
            run_id: The run ID to generate a report for
            upload_to_storage: Whether to upload the report to Supabase storage
            
        Returns:
            The path to the generated report file
        """
        logger.info(f"Generating report for run_id: {run_id}")
        
        # Store the current run ID
        self.current_run_id = run_id
        
        try:
            # Create a new document
            doc = Document()
            
            # Set up document styles
            self._setup_document_styles(doc)
            
            # Fetch raw data for the run
            logger.info(f"Fetching raw data for run_id: {run_id}")
            sections_data = await self.fetch_raw_data(run_id)
            
            if not sections_data:
                logger.error(f"No data found for run_id: {run_id}")
                # Create an error document
                doc.add_heading("Error: No Data Found", level=1)
                doc.add_paragraph(f"No data was found for run ID: {run_id}")
                doc.add_paragraph("Please check that the run ID is correct and that data has been stored for this run.")
            else:
                logger.info(f"Found {len(sections_data)} sections for run_id: {run_id}")
                logger.debug(f"Available sections: {list(sections_data.keys())}")
                
                # Add title page
                await self._add_title_page(doc, sections_data)
                
                # Add table of contents
                await self._add_table_of_contents(doc)
                
                # Add executive summary if available
                if "exec_summary" in sections_data:
                    logger.info("Adding executive summary")
                    await self._add_executive_summary(doc, sections_data["exec_summary"])
                else:
                    logger.warning("Executive summary data not found")
                    doc.add_heading("Executive Summary", level=1)
                    doc.add_paragraph("Executive summary data not available for this report.", style='Intense Quote')
                
                # Process each section in the defined order
                for section_name in self.SECTION_ORDER:
                    # Skip executive summary as it's already added
                    if section_name == "exec_summary":
                        continue
                        
                    # Get the display name for the section
                    section_title = self.SECTION_MAPPING.get(section_name, section_name.replace("_", " ").title())
                    
                    logger.info(f"Processing section: {section_name} (Title: {section_title})")
                    
                    # Check if we have data for this section
                    if section_name in sections_data:
                        section_data = sections_data[section_name]
                        
                        # Add a page break before each main section
                        doc.add_page_break()
                        
                        # Check if we have a specific formatter for this section
                        formatter_method_name = f"_format_{section_name}"
                        if hasattr(self, formatter_method_name):
                            logger.info(f"Using specific formatter for {section_name}")
                            formatter_method = getattr(self, formatter_method_name)
                            try:
                                await formatter_method(doc, section_data)
                            except Exception as e:
                                logger.error(f"Error formatting section {section_name}: {str(e)}")
                                logger.error(traceback.format_exc())
                                # Add error message to document
                                doc.add_heading(section_title, level=1)
                                doc.add_paragraph(f"Error formatting section: {str(e)}", style='Intense Quote')
                                # Try generic formatter as fallback
                                try:
                                    await self._format_generic_section(doc, section_name, section_data)
                                except Exception as e2:
                                    logger.error(f"Generic formatter also failed for {section_name}: {str(e2)}")
                                    # Use the most basic fallback
                                    self._format_generic_section_fallback(doc, section_name, section_data)
                        else:
                            # Use generic formatter
                            logger.info(f"Using generic formatter for {section_name}")
                            try:
                                await self._format_generic_section(doc, section_name, section_data)
                            except Exception as e:
                                logger.error(f"Error in generic formatting for {section_name}: {str(e)}")
                                logger.error(traceback.format_exc())
                                # Use the most basic fallback
                                self._format_generic_section_fallback(doc, section_name, section_data)
                    else:
                        logger.warning(f"No data found for section: {section_name}")
                        # Add a placeholder for missing sections
                        doc.add_page_break()
                        doc.add_heading(section_title, level=1)
                        doc.add_paragraph(f"No data available for this section.", style='Intense Quote')
                
                # Add recommendations if available
                if "final_recommendations" in sections_data:
                    logger.info("Adding recommendations")
                    await self._add_recommendations(doc, sections_data["final_recommendations"])
                else:
                    logger.warning("Recommendations data not found")
            
            # Save the document to a temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("./output")
            output_dir.mkdir(exist_ok=True)
            
            output_filename = f"brand_naming_report_{run_id}_{timestamp}.docx"
            output_path = str(output_dir / output_filename)
            
            logger.info(f"Saving report to: {output_path}")
            doc.save(output_path)
            
            # Upload to storage if requested
            if upload_to_storage and self.supabase:
                try:
                    logger.info("Uploading report to storage")
                    report_url = await self.upload_report_to_storage(output_path, run_id)
                    logger.info(f"Report uploaded to: {report_url}")
                except Exception as e:
                    logger.error(f"Error uploading report to storage: {str(e)}")
                    logger.error(traceback.format_exc())
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    async def _add_title_page(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add a title page to the document."""
        logger.info("Adding title page")
        
        try:
            # Get brand context data if available
            brand_context = data.get("brand_context", {})
            brand_name = brand_context.get("brand_name", "")
            industry = brand_context.get("industry", "")
            
            # Create comprehensive format data for the template
            format_data = {
                "run_id": self.current_run_id,
                "brand_name": brand_name,
                "industry": industry,
                "brand_context": json.dumps(brand_context, indent=2)
            }
            
            # Format the template using the helper method
            formatted_template = self._format_template("title_page", format_data, "Title Page")
            
            # Try to get title page content from LLM
            title_content = await self._safe_llm_invoke([
                SystemMessage(content="You are a professional report formatter creating a title page for a brand naming report."),
                HumanMessage(content=formatted_template)
            ], section_name="Title Page")
            
            # Try to parse the response
            try:
                content = json.loads(title_content.content)
                title = content.get("title", "Brand Naming Report")
                subtitle = content.get("subtitle", "Generated by Mae Brand Naming Expert")
            except (json.JSONDecodeError, AttributeError):
                # Default values if parsing fails
                title = "Brand Naming Report"
                subtitle = "Generated by Mae Brand Naming Expert"
                
            # Add title page content
            doc.add_paragraph()  # Add some space at the top
            
            # Add logo if available
            # logo_path = Path(__file__).parent / "assets" / "logo.png"
            # if logo_path.exists():
            #     doc.add_picture(str(logo_path), width=Inches(2.5))
            
            # Add title
            title_para = doc.add_paragraph()
            title_run = title_para.add_run(title)
            title_run.font.size = Pt(24)
            title_run.font.bold = True
            title_run.font.color.rgb = RGBColor(0, 70, 127)  # Dark blue
            
            # Add subtitle
            subtitle_para = doc.add_paragraph()
            subtitle_run = subtitle_para.add_run(subtitle)
            subtitle_run.font.size = Pt(16)
            subtitle_run.font.italic = True
            
            # Add date
            date_para = doc.add_paragraph()
            date_run = date_para.add_run(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
            date_run.font.size = Pt(12)
            
            # Add run ID
            run_id_para = doc.add_paragraph()
            run_id_run = run_id_para.add_run(f"Report ID: {self.current_run_id}")
            run_id_run.font.size = Pt(10)
            
            # Add page break
            doc.add_page_break()
            
        except Exception as e:
            logger.error(f"Error adding title page: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Add a simple title page as fallback
            doc.add_heading("Brand Naming Report", level=0)
            doc.add_paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
            doc.add_paragraph(f"Report ID: {self.current_run_id}")
            doc.add_page_break()
            
    async def _add_table_of_contents(self, doc: Document) -> None:
        """Add a table of contents to the document."""
        logger.info("Adding table of contents")
        
        try:
            # Create format data for the template
            format_data = {
                "run_id": self.current_run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Format the template
            formatted_template = self._format_template("table_of_contents", format_data, "Table of Contents")
            
            # Get system content
            system_content = self._get_system_content("You are a professional report formatter creating a table of contents.")
            
            # Optional: Get any additional TOC information from LLM
            try:
                toc_content = await self._safe_llm_invoke([
                    SystemMessage(content=system_content),
                    HumanMessage(content=formatted_template)
                ], section_name="Table of Contents")
                
                # Add a heading for the TOC
                doc.add_heading("Table of Contents", level=1)
                
                # Add a paragraph before the TOC (optional)
                if hasattr(toc_content, 'content') and toc_content.content:
                    try:
                        # Try to parse as JSON if formatted that way
                        content = json.loads(toc_content.content)
                        if "introduction" in content:
                            doc.add_paragraph(content["introduction"])
                    except json.JSONDecodeError:
                        # Just use the raw content
                        doc.add_paragraph(toc_content.content)
                
                # Add the actual table of contents field
                paragraph = doc.add_paragraph()
                run = paragraph.add_run()
                fld_char = OxmlElement('w:fldChar')
                fld_char.set(qn('w:fldCharType'), 'begin')
                
                instr_text = OxmlElement('w:instrText')
                instr_text.set(qn('xml:space'), 'preserve')
                instr_text.text = 'TOC \\o "1-3" \\h \\z \\u'
                
                fld_char_end = OxmlElement('w:fldChar')
                fld_char_end.set(qn('w:fldCharType'), 'end')
                
                r_element = run._r
                r_element.append(fld_char)
                r_element.append(instr_text)
                r_element.append(fld_char_end)
                
                # Add a page break after TOC
                doc.add_page_break()
                
            except Exception as e:
                logger.error(f"Error getting TOC descriptions from LLM: {str(e)}")
                
                # Fallback: Add a basic TOC without LLM content
                doc.add_heading("Table of Contents", level=1)
                
                paragraph = doc.add_paragraph()
                run = paragraph.add_run()
                fld_char = OxmlElement('w:fldChar')
                fld_char.set(qn('w:fldCharType'), 'begin')
                
                instr_text = OxmlElement('w:instrText')
                instr_text.set(qn('xml:space'), 'preserve')
                instr_text.text = 'TOC \\o "1-3" \\h \\z \\u'
                
                fld_char_end = OxmlElement('w:fldChar')
                fld_char_end.set(qn('w:fldCharType'), 'end')
                
                r_element = run._r
                r_element.append(fld_char)
                r_element.append(instr_text)
                r_element.append(fld_char_end)
                
                # Add a page break after TOC
                doc.add_page_break()
                
        except Exception as e:
            logger.error(f"Error adding table of contents: {str(e)}")
            doc.add_paragraph("Error generating table of contents", style='Intense Quote')
    
    async def _add_executive_summary(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add an executive summary to the document."""
        logger.info("Adding executive summary")
        
        try:
            # Extract brand context
            brand_context = data.get("brand_context", {})
            brand_name = brand_context.get("brand_name", "")
            industry = brand_context.get("industry", "")
            
            # Create format data
            format_data = {
                "run_id": self.current_run_id,
                "brand_name": brand_name,
                "industry": industry,
                "brand_context": json.dumps(brand_context, indent=2),
                "data": json.dumps(data, indent=2)
            }
            
            # Format the template using the helper method
            formatted_prompt = self._format_template("executive_summary", format_data, "Executive Summary")
            
            # Get system content
            system_content = self._get_system_content("You are an expert report formatter creating a professional executive summary.")
            
            # Call LLM to generate executive summary
            response = await self._safe_llm_invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=formatted_prompt)
            ], section_name="Executive Summary")
            
            # Add the executive summary to the document
            doc.add_heading("Executive Summary", level=1)
            doc.add_paragraph(response.content)
            
            # Add sections if available
            if "sections" in data and isinstance(data["sections"], list):
                for section in data["sections"]:
                    if "heading" in section and "content" in section:
                        doc.add_heading(section["heading"], level=2)
                        doc.add_paragraph(section["content"])
            
            # Add key points if available
            if "key_points" in data and isinstance(data["key_points"], list):
                doc.add_heading("Key Points", level=2)
                for point in data["key_points"]:
                    bullet = doc.add_paragraph(style='List Bullet')
                    bullet.add_run(point)
            
            # Add recommendations if available
            if "recommendations" in data and isinstance(data["recommendations"], list):
                doc.add_heading("Recommendations", level=2)
                for rec in data["recommendations"]:
                    bullet = doc.add_paragraph(style='List Bullet')
                    bullet.add_run(rec)
            
        except Exception as e:
            logger.error(f"Error adding executive summary: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Add a simple executive summary as fallback
            doc.add_heading("Executive Summary", level=1)
            doc.add_paragraph("An error occurred while generating the executive summary.")
            
            # Try to add some basic information from the raw data
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_paragraph(f"{key}: {value}")
                        
    async def _add_recommendations(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add strategic recommendations to the document."""
        logger.info("Adding strategic recommendations")
        
        try:
            # Extract brand context
            brand_context = data.get("brand_context", {})
            brand_name = brand_context.get("brand_name", "")
            industry = brand_context.get("industry", "")
            
            # Create format data
            format_data = {
                "run_id": self.current_run_id,
                "brand_name": brand_name,
                "industry": industry,
                "brand_context": json.dumps(brand_context, indent=2),
                "data": json.dumps(data, indent=2)
            }
            
            # Format the template using the helper method
            formatted_prompt = self._format_template("recommendations", format_data, "Strategic Recommendations")
            
            # Get system content
            system_content = self._get_system_content("You are an expert report formatter creating professional strategic recommendations.")
            
            # Call LLM to generate recommendations
            response = await self._safe_llm_invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=formatted_prompt)
            ], section_name="Strategic Recommendations")
            
            # Add the recommendations to the document
            doc.add_heading("Strategic Recommendations", level=1)
            
            # Parse the response
            try:
                content = json.loads(response.content)
                
                # Add introduction if available
                if "introduction" in content and content["introduction"]:
                    doc.add_paragraph(content["introduction"])
                
                # Add recommendations if available
                if "recommendations" in content and isinstance(content["recommendations"], list):
                    for i, rec in enumerate(content["recommendations"]):
                        if isinstance(rec, dict) and "title" in rec and "details" in rec:
                            # Rich recommendation format
                            doc.add_heading(f"{i+1}. {rec['title']}", level=2)
                            doc.add_paragraph(rec["details"])
                            
                            # Add sub-recommendations if available
                            if "steps" in rec and isinstance(rec["steps"], list):
                                for step in rec["steps"]:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(step)
                        else:
                            # Simple recommendation format
                            doc.add_heading(f"Recommendation {i+1}", level=2)
                            doc.add_paragraph(str(rec))
                
                # Add conclusion if available
                if "conclusion" in content and content["conclusion"]:
                    doc.add_heading("Conclusion", level=2)
                    doc.add_paragraph(content["conclusion"])
                    
            except json.JSONDecodeError:
                # Fallback to using raw text
                doc.add_paragraph(response.content)
                
        except Exception as e:
            logger.error(f"Error adding recommendations: {str(e)}")
            doc.add_paragraph(f"Error occurred while adding recommendations: {str(e)}", style='Intense Quote')

    async def _format_generic_section(self, doc: Document, section_name: str, data: Dict[str, Any]) -> None:
        """Format a section using LLM when no specific formatter is available."""
        try:
            # Generate a section prompt template key based on section name
            prompt_key = section_name.lower().replace(" ", "_")
            
            # Try to find a matching prompt template
            if prompt_key in self.prompts:
                logger.info(f"Using prompt template '{prompt_key}' for section {section_name}")
                
                # Create a comprehensive format data dictionary
                format_data = {
                    "run_id": self.current_run_id,
                    "data": json.dumps(data, indent=2),
                    "section_data": json.dumps(data, indent=2),
                    section_name.lower().replace(" ", "_"): json.dumps(data, indent=2)
                }
                
                # Special handling for survey_simulation
                if prompt_key == "survey_simulation":
                    survey_data = data.get("survey_simulation", {})
                    if survey_data:
                        # Extract brand names from the survey_simulation data
                        brand_names = list(survey_data.keys())
                        format_data["brand_names"] = ", ".join(brand_names)
                        format_data["survey_data"] = json.dumps(survey_data, indent=2)
                        logger.info(f"Special handling for survey_simulation: found {len(brand_names)} brand names: {format_data['brand_names']}")
                    else:
                        logger.warning(f"survey_simulation key not found in section data")
                        # Try to find brand names in the top level if possible
                        if isinstance(data, dict) and len(data) > 0:
                            keys = list(data.keys())
                            if not any(k for k in keys if k.startswith("_") or k in ["metadata", "info", "config"]):
                                # These might be brand names
                                format_data["brand_names"] = ", ".join(keys)
                                format_data["survey_data"] = json.dumps(data, indent=2)
                                logger.info(f"Fallback for survey_simulation: using top-level keys as brand names: {format_data['brand_names']}")
                
                # Special handling for translation_analysis
                elif prompt_key == "translation_analysis":
                    translation_data = data.get("translation_analysis", {})
                    if translation_data:
                        # Extract brand names from the translation_analysis data
                        brand_names = list(translation_data.keys()) if isinstance(translation_data, dict) else []
                        format_data["brand_names"] = ", ".join(brand_names)
                        format_data["translation_analysis"] = json.dumps(translation_data, indent=2)
                        logger.info(f"Special handling for translation_analysis: found {len(brand_names)} brand names: {format_data['brand_names']}")
                    else:
                        logger.warning(f"translation_analysis key not found in section data")
                        # Try to find brand names in the top level if possible
                        if isinstance(data, dict) and len(data) > 0:
                            keys = list(data.keys())
                            if not any(k for k in keys if k.startswith("_") or k in ["metadata", "info", "config"]):
                                # These might be brand names
                                format_data["brand_names"] = ", ".join(keys)
                                format_data["translation_analysis"] = json.dumps(data, indent=2)
                                logger.info(f"Fallback for translation_analysis: using top-level keys as brand names: {format_data['brand_names']}")
                
                # For more complex data structures, extract specific fields
                if isinstance(data, dict):
                    for key, value in data.items():
                        # Only include scalar values or simple lists
                        if isinstance(value, (str, int, float, bool)) or (
                            isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value)
                        ):
                            format_data[key] = value
                
                # Format the template using the helper method
                try:
                    formatted_prompt = self._format_template(prompt_key, format_data, section_name)
                    
                    # Log before LLM call
                    logger.info(f"Making LLM call for section: {section_name}")
                    
                    # Call LLM with the formatted prompt
                    system_content = self._get_system_content(f"You are an expert report formatter creating a professional {section_name} section.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=formatted_prompt)
                    ]
                    
                    # Make the LLM call with improved error handling
                    response = await self._safe_llm_invoke(messages, section_name=section_name)
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
                    
                    # Add the section title
                    section_title = self.SECTION_MAPPING.get(section_name, section_name.replace("_", " ").title())
                    if "title" in content and content["title"]:
                        doc.add_heading(content["title"], level=1)
                    else:
                        doc.add_heading(section_title, level=1)
                    
                    # Add the main content if available
                    if "content" in content and content["content"]:
                        doc.add_paragraph(content["content"])
                    
                    # Add subsections if available
                    if "sections" in content and isinstance(content["sections"], list):
                        for section in content["sections"]:
                            if "heading" in section and "content" in section:
                                doc.add_heading(section["heading"], level=2)
                                doc.add_paragraph(section["content"])
                    
                    # Process other structured content
                    for key, value in content.items():
                        # Skip already processed keys
                        if key in ["title", "content", "sections"]:
                            continue
                            
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
                    
                    # Add the section title
                    section_title = self.SECTION_MAPPING.get(section_name, section_name.replace("_", " ").title())
                    doc.add_heading(section_title, level=1)
                    
                    # Add the content
                    doc.add_paragraph(cleaned_content)
            else:
                logger.warning(f"No prompt template found for section {section_name}, using fallback formatting")
                # If no specific prompt template is found, use a generic approach
                self._format_generic_section_fallback(doc, section_name, data)
                    
        except Exception as e:
            logger.error(f"Error in generic section formatting for {section_name}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            doc.add_paragraph(f"Error formatting section: {str(e)}", style='Intense Quote')

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
            logger.error(f"Error details: {traceback.format_exc()}")
            raise
            
    async def store_report_metadata(self, run_id: str, report_url: str, file_size_kb: float, format: str) -> None:
        """
        Store metadata about the generated report in the report_metadata table.
        
        Args:
            run_id: The run ID associated with the report
            report_url: The URL where the report is accessible
            file_size_kb: The size of the report file in kilobytes
            format: The format of the report (e.g., 'docx')
        """
        logger.info(f"Storing report metadata for run_id: {run_id}")
        
        try:
            # Get the current timestamp
            timestamp = datetime.now().isoformat()
            
            # Get the current version number
            version = 1
            try:
                # Query for existing versions
                query = f"""
                SELECT MAX(version) as max_version FROM report_metadata 
                WHERE run_id = '{run_id}'
                """
                result = await self.supabase.execute_with_retry(query, {})
                if result and len(result) > 0 and result[0]['max_version'] is not None:
                    version = int(result[0]['max_version']) + 1
                    logger.info(f"Found existing reports, using version: {version}")
            except Exception as e:
                logger.warning(f"Error getting version number: {str(e)}")
                # Continue with default version 1
            
            # Insert metadata into the report_metadata table
            metadata = {
                "run_id": run_id,
                "report_url": report_url,
                "created_at": timestamp,
                "last_updated": timestamp,
                "format": format,
                "file_size_kb": file_size_kb,
                "version": version,
                "notes": f"Generated by ReportFormatter v{settings.version}"
            }
            
            result = await self.supabase.execute_with_retry(
                "insert",
                "report_metadata",
                metadata
            )
            
            logger.info(f"Report metadata stored successfully for run_id: {run_id}, version: {version}")
            
        except Exception as e:
            logger.error(f"Error storing report metadata: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Don't raise the exception - this is not critical for report generation

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