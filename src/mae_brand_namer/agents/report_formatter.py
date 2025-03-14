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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

logger = get_logger(__name__)


class ReportFormatter:
    """
    Handles formatting and generation of reports using raw data from the report_raw_data table.
    This is the second step in the two-step report generation process.
    """

    # Mapping between DB section names and formatter section names
    SECTION_MAPPING = {
        # DB section name -> Formatter section name
        "brand_context": "brand_context",
        "brand_name_generation": "name_generation",
        "linguistic_analysis": "linguistic_analysis",
        "semantic_analysis": "semantic_analysis",
        "cultural_sensitivity_analysis": "cultural_sensitivity",
        "translation_analysis": "translation_analysis",
        "survey_simulation": "survey_simulation",
        "brand_name_evaluation": "name_evaluation",
        "domain_analysis": "domain_analysis",
        "seo_online_discoverability": "seo_analysis",  # Updated to match our section name
        "competitor_analysis": "competitor_analysis",
        "market_research": "market_research",
        "exec_summary": "executive_summary",
        "final_recommendations": "recommendations"
    }

    # Reverse mapping for convenience
    REVERSE_SECTION_MAPPING = {v: k for k, v in SECTION_MAPPING.items()}

    # Default storage bucket for reports
    STORAGE_BUCKET = "agent_reports"
    
    # Report formats
    FORMAT_DOCX = "docx"
    
    def __init__(self, dependencies: Optional[Dependencies] = None):
        """Initialize the ReportFormatter with dependencies."""
        # Extract clients from dependencies if available
        if dependencies:
            self.supabase = dependencies.supabase
            # Safely try to get LLM from dependencies, with fallback if not available
            try:
                self.llm = dependencies.llm
            except AttributeError:
                # Log the issue and create a new LLM instance
                print("Warning: 'llm' attribute not found in Dependencies, creating new instance")
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0.5,
                    google_api_key=settings.gemini_api_key,
                    convert_system_message_to_human=True
                )
        else:
            # Create clients if not provided
            self.supabase = SupabaseManager()
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.2,
                google_api_key=settings.gemini_api_key,
                convert_system_message_to_human=True
            )
            
        # Initialize error tracking
        self.formatting_errors = {}
        self.missing_sections = set()
        
        # Initialize current run ID
        self.current_run_id = None
        
        # Define section mapping from DB to formatter
        self.SECTION_MAPPING = {
            "brand_context": "brand_context",
            "brand_name_generation": "name_generation",
            "linguistic_analysis": "linguistic_analysis",
            "semantic_analysis": "semantic_analysis",
            "cultural_sensitivity": "cultural_sensitivity",
            "translation_analysis": "translation_analysis",
            "survey_simulation": "survey_simulation",
            "name_evaluation": "name_evaluation",
            "domain_analysis": "domain_analysis",
            "seo_online_discoverability": "seo_analysis",  # Updated to match our section name
            "competitor_analysis": "competitor_analysis",
            "market_research": "market_research",
            "executive_summary": "executive_summary",
            "recommendations": "recommendations"
        }
        
        # Create transformers mapping
        self.transformers = {
            "brand_context": self._transform_brand_context,
            "name_generation": self._transform_name_generation,
            "semantic_analysis": self._transform_semantic_analysis,
            "linguistic_analysis": self._transform_linguistic_analysis,
            "cultural_sensitivity": self._transform_cultural_sensitivity,
            "name_evaluation": self._transform_name_evaluation,
            "translation_analysis": self._transform_translation_analysis,
            "market_research": self._transform_market_research,
            "competitor_analysis": self._transform_competitor_analysis,
            "domain_analysis": self._transform_domain_analysis,
            "survey_simulation": self._transform_survey_simulation,
            # Add transformers as needed
        }
        
        # Load prompts from YAML files
        try:
            self.prompts = {
                # Title page and TOC
                "title_page": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "title_page.yaml")),
                "table_of_contents": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "table_of_contents.yaml")),
                
                # Main section prompts
                "executive_summary": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "executive_summary.yaml")),
                "recommendations": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "recommendations.yaml")),
                "seo_analysis": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "seo_analysis.yaml")),
                "brand_context": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_context.yaml")),
                "brand_name_generation": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_name_generation.yaml")),
                "semantic_analysis": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "semantic_analysis.yaml")),
                "linguistic_analysis": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "linguistic_analysis.yaml")),
                "cultural_sensitivity": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "cultural_sensitivity.yaml")),
                "translation_analysis": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "translation_analysis.yaml")),
                "market_research": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "market_research.yaml")),
                "competitor_analysis": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "competitor_analysis.yaml")),
                "name_evaluation": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "brand_name_evaluation.yaml")),
                "domain_analysis": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "domain_analysis.yaml")),
                "survey_simulation": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "survey_simulation.yaml")),
                "system": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "system.yaml")),
                "shortlisted_names_summary": load_prompt(str(Path(__file__).parent / "prompts" / "report_formatter" / "shortlisted_names_summary.yaml"))
            }
            
            logger.info("Successfully loaded report formatter prompts")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            # Create an empty dictionary as fallback
            self.prompts = {}
        
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
            
            response = await self.llm.ainvoke(messages)
            
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
            
            # Map DB section name to formatter section name
            formatter_section_name = self.SECTION_MAPPING.get(db_section_name, db_section_name)
            
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
            title_content = await self.llm.ainvoke([
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
        toc_sections = [
            "Executive Summary",
            "Brand Context",
            "Name Generation",
            "Linguistic Analysis",
            "Semantic Analysis",
            "Cultural Sensitivity",
            "Translation Analysis",
            "Name Evaluation",
            "Domain Analysis",
            "SEO Analysis",
            "Competitor Analysis",
            "Market Research",
            "Survey Simulation",
            "Strategic Recommendations"
        ]
        
        # Add TOC entries with consistent styling
        for i, section in enumerate(toc_sections, 1):
            p = doc.add_paragraph(style='TOC 1')
            p.add_run(f"{i}. {section}")
            
            # Add page number placeholder (would be replaced in a real TOC)
            tab = p.add_run("\t")
            p.add_run("___")
        
        # Optional: Get any additional TOC information from LLM for section descriptions
        try:
            toc_content = await self.llm.ainvoke([
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
        
        # Define section order
        major_sections = {
            "executive_summary",
            "brand_context",
            "name_generation",
            "linguistic_analysis",
            "semantic_analysis",
            "cultural_sensitivity",
            "translation_analysis",
            "name_evaluation",
            "domain_analysis",
            "seo_online_discoverability",
            "competitor_analysis",
            "market_research",
            "survey_simulation",
            "recommendations"
        }
        
        section_order = [
            "executive_summary",
            "brand_context",
            "name_generation",
            "linguistic_analysis",
            "semantic_analysis",
            "cultural_sensitivity",
            "translation_analysis",
            "name_evaluation",
            "domain_analysis",
            "seo_online_discoverability",
            "competitor_analysis",
            "market_research",
            "survey_simulation",
            "recommendations"
        ]
        
        # Track data quality issues
        data_quality_issues = {}
        
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
                        await self._add_executive_summary(doc, sections_data[section_name])
                    elif section_name == "recommendations":
                        # Generate recommendations using LLM
                        await self._add_recommendations(doc, sections_data[section_name])
                    else:
                        await self._format_section(doc, section_name, sections_data[section_name])
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
        """Add executive summary section to the document."""
        # Extract relevant data for the executive summary
        brand_context = raw_data.get("brand_context", {})
        total_names = len(raw_data.get("name_generation", {}).get("brand_names", []))
        shortlisted_names = [name for name in raw_data.get("name_evaluation", {}).values() 
                            if name.get("shortlist_status") == True]
        
        # Get executive summary from LLM
        summary_content = await self.llm.ainvoke([
            SystemMessage(content="You are an expert report writer creating an executive summary for a brand naming report."),
            HumanMessage(content=str(self.prompts["executive_summary"].format(
                run_id=self.current_run_id,
                sections_data=raw_data,
                brand_context=brand_context,
                total_names=total_names,
                shortlisted_names=shortlisted_names,
                user_prompt=brand_context.get("user_prompt", "No user prompt available")
            )))
        ])
        
        # Parse the response
        try:
            content = json.loads(summary_content.content)
        except json.JSONDecodeError:
            # Fallback to error message if parsing fails
            content = {
                "summary": "Error generating executive summary. Please check the raw data and try again.",
                "key_points": []
            }
        
        # Add executive summary heading
        doc.add_heading("Executive Summary", level=1)
        
        # Add summary text
        summary_para = doc.add_paragraph()
        summary_para.add_run(content["summary"])
        
        # Add key points if available
        if content.get("key_points"):
            doc.add_paragraph()
            doc.add_heading("Key Points", level=2)
            for point in content["key_points"]:
                p = doc.add_paragraph(style="List Bullet")
                p.add_run(point)
        
        # Add page break
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    async def _add_recommendations(self, doc: Document, raw_data: Dict[str, Any]) -> None:
        """Add recommendations section to the document."""
        # Extract relevant data for recommendations
        brand_context = raw_data.get("brand_context", {})
        name_evaluation = raw_data.get("name_evaluation", {})
        shortlisted_names = [name for name, data in name_evaluation.items() 
                            if data.get("shortlist_status") == True]
        
        # Get recommendations from LLM
        recommendations_content = await self.llm.ainvoke([
            SystemMessage(content="You are an expert brand consultant providing strategic recommendations based on a brand naming analysis."),
            HumanMessage(content=str(self.prompts["recommendations"].format(
                run_id=self.current_run_id,
                sections_data=raw_data,
                brand_context=brand_context,
                shortlisted_names=shortlisted_names,
                user_prompt=brand_context.get("user_prompt", "No user prompt available")
            )))
        ])
        
        # Parse the response
        try:
            content = json.loads(recommendations_content.content)
        except json.JSONDecodeError:
            # Fallback to error message if parsing fails
            content = {
                "overview": "Error generating recommendations. Please check the raw data and try again.",
                "recommendations": []
            }
        
        # Add recommendations heading
        doc.add_heading("Recommendations", level=1)
        
        # Add overview
        overview_para = doc.add_paragraph()
        overview_para.add_run(content["overview"])
        
        # Add recommendations
        if content.get("recommendations"):
            doc.add_paragraph()
            for i, rec in enumerate(content["recommendations"], 1):
                # Add recommendation heading
                doc.add_heading(f"Recommendation {i}: {rec['title']}", level=2)
                
                # Add recommendation details
                details_para = doc.add_paragraph()
                details_para.add_run(rec["details"])
                
                # Add rationale if available
                if "rationale" in rec:
                    rationale_para = doc.add_paragraph()
                    rationale_para.add_run("Rationale: ").bold = True
                    rationale_para.add_run(rec["rationale"])
                
                # Add next steps if available
                if "next_steps" in rec:
                    next_steps_para = doc.add_paragraph()
                    next_steps_para.add_run("Next Steps: ").bold = True
                    next_steps_para.add_run(rec["next_steps"])
        
        # Add page break
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

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

    async def _format_seo_online_discoverability(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the SEO online discoverability section."""
        # Extract SEO analysis data
        analyses = data.get("seo_online_discoverability", {})
        if not analyses:
            doc.add_paragraph("No SEO online discoverability data available.", style='Quote')
            return
            
        # Add introduction
        doc.add_paragraph(
            f"SEO and online discoverability analysis was performed for {len(analyses)} brand names "
            "to evaluate search engine performance potential and online presence opportunities."
        )
        
        # Process each brand
        for brand_name, analysis in analyses.items():
            doc.add_heading(brand_name, level=2)
            
            # Create a summary table with key metrics
            summary_table = doc.add_table(rows=1, cols=4)
            summary_table.style = 'Table Grid'
            
            # Add headers
            headers = summary_table.rows[0].cells
            headers[0].text = "SEO Viability Score"
            headers[1].text = "Search Volume"
            headers[2].text = "Keyword Competition"
            headers[3].text = "Branded Keyword Potential"
            
            # Add summary row
            summary_row = summary_table.add_row().cells
            summary_row[0].text = str(analysis.get("seo_viability_score", "N/A"))
            summary_row[1].text = str(analysis.get("search_volume", "N/A"))
            summary_row[2].text = str(analysis.get("keyword_competition", "N/A"))
            summary_row[3].text = str(analysis.get("branded_keyword_potential", "N/A"))
            
            # Generate enhanced SEO content using LLM
            try:
                seo_content = await self._generate_seo_content(brand_name, analysis)
                
                # Add LLM-generated content
                if seo_content:
                    # Add viability assessment
                    doc.add_heading("SEO Viability Assessment", level=3)
                    doc.add_paragraph(seo_content.get("seo_viability_assessment", "No assessment available."))
                    
                    # Add search landscape analysis
                    doc.add_heading("Search Landscape Analysis", level=3)
                    doc.add_paragraph(seo_content.get("search_landscape_analysis", "No analysis available."))
                    
                    # Add content strategy recommendations
                    doc.add_heading("Content Strategy Recommendations", level=3)
                    doc.add_paragraph(seo_content.get("content_strategy_recommendations", "No recommendations available."))
                    
                    # Add technical SEO considerations
                    doc.add_heading("Technical SEO Considerations", level=3)
                    doc.add_paragraph(seo_content.get("technical_seo_considerations", "No considerations available."))
                    
                    # Add action plan
                    doc.add_heading("Action Plan", level=3)
                    action_plan = seo_content.get("action_plan", [])
                    if action_plan:
                        for action in action_plan:
                            doc.add_paragraph(action, style='List Bullet')
                    else:
                        doc.add_paragraph("No action plan available.", style='Quote')
            except Exception as e:
                logger.error(f"Error generating enhanced SEO content for {brand_name}: {str(e)}")
                doc.add_paragraph(f"Error generating enhanced SEO content: {str(e)}", style='Intense Quote')
                
                # Fall back to original data
                doc.add_heading("Detailed Analysis", level=3)
                
                # Create a table for detailed metrics
                detail_table = doc.add_table(rows=6, cols=2)
                detail_table.style = 'Table Grid'
                
                # Add rows with data
                rows = detail_table.rows
                rows[0].cells[0].text = "Keyword Alignment"
                rows[0].cells[1].text = str(analysis.get("keyword_alignment", "N/A"))
                
                rows[1].cells[0].text = "Non-Branded Keyword Potential"
                rows[1].cells[1].text = str(analysis.get("non_branded_keyword_potential", "N/A"))
                
                rows[2].cells[0].text = "Exact Match Search Results"
                rows[2].cells[1].text = str(analysis.get("exact_match_search_results", "N/A"))
                
                rows[3].cells[0].text = "Competitor Domain Strength"
                rows[3].cells[1].text = str(analysis.get("competitor_domain_strength", "N/A"))
                
                rows[4].cells[0].text = "Name Length Searchability"
                rows[4].cells[1].text = str(analysis.get("name_length_searchability", "N/A"))
                
                rows[5].cells[0].text = "Content Marketing Opportunities"
                rows[5].cells[1].text = str(analysis.get("content_marketing_opportunities", "N/A"))
            
            # Add recommendations
            doc.add_heading("SEO Recommendations", level=3)
            recommendations = analysis.get("seo_recommendations", [])
            if isinstance(recommendations, list) and recommendations:
                for rec in recommendations:
                    doc.add_paragraph(rec, style='List Bullet')
            elif isinstance(recommendations, str) and recommendations:
                doc.add_paragraph(recommendations)
            else:
                doc.add_paragraph("No specific SEO recommendations available.", style='Quote')
            
            # Add section divider
            doc.add_paragraph("", style='Normal')
            doc.add_paragraph("* * *", style='Normal').alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_paragraph("", style='Normal')

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

    async def _format_section(self, doc: Document, section_name: str, sections_data: dict) -> tuple:
        """Format a section of the report.
        
        Args:
            doc: The document to add the section to.
            section_name: The name of the section to format.
            sections_data: The data for all sections.
            
        Returns:
            tuple: (success_flag, error_message)
        """
        logger.info(f"Formatting section: {section_name}")
        
        # Add section header with consistent formatting
        section_title = section_name.replace("_", " ").title()
        doc.add_heading(section_title, level=1)
        
        try:
            section_data = sections_data.get(section_name, {})
            
            # Handle empty or missing section data
            if not section_data:
                logger.warning(f"No data found for section {section_name}")
                doc.add_paragraph(f"No data available for {section_title}.")
                return False, f"Missing data for {section_name}"
            
            # Special handling for different section types
            if section_name == "executive_summary":
                # Get formatted executive summary from LLM
                try:
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an expert brand consultant creating an executive summary for a brand naming report."),
                        HumanMessage(content=str(self.prompts["executive_summary"]))
                    ])
                    content = json.loads(response.content)
                    
                    # Add the executive summary content to the document
                    for subsection in content.get("subsections", []):
                        doc.add_heading(subsection["title"], level=2)
                        doc.add_paragraph(subsection["content"])
                    
                    # Add any additional insights
                    if "additional_insights" in content:
                        doc.add_heading("Additional Insights", level=2)
                        doc.add_paragraph(content["additional_insights"])
                        
                except Exception as e:
                    logger.error(f"Error formatting executive summary: {str(e)}")
                    doc.add_paragraph("Error generating executive summary. Please check the data and try again.")
                    return False, f"Error formatting executive summary: {str(e)}"

            elif section_name == "brand_context":
                # Format brand context information
                if "description" in section_data:
                    doc.add_paragraph(section_data["description"])
                
                if "target_audience" in section_data:
                    doc.add_heading("Target Audience", level=2)
                    doc.add_paragraph(section_data["target_audience"])
                
                if "brand_values" in section_data:
                    doc.add_heading("Brand Values", level=2)
                    values_list = doc.add_paragraph(style='List Bullet')
                    for value in section_data["brand_values"]:
                        values_list.add_run(f"{value}\n")
                
                if "industry" in section_data:
                    doc.add_heading("Industry", level=2)
                    doc.add_paragraph(section_data["industry"])
                
                if "brand_promise" in section_data:
                    doc.add_heading("Brand Promise", level=2)
                    doc.add_paragraph(section_data["brand_promise"])

            elif section_name == "name_generation":
                # Format name generation process
                doc.add_heading("Name Generation Methodology", level=2)
                if "methodology" in section_data:
                    doc.add_paragraph(section_data["methodology"])
                else:
                    doc.add_paragraph("Names were generated using a combination of creative techniques, linguistic analysis, and AI algorithms to ensure alignment with brand values and goals.")
                
                doc.add_heading("Generated Names", level=2)
                if "names" in section_data and section_data["names"]:
                    table = doc.add_table(rows=1, cols=3)
                    table.style = 'Table Grid'
                    
                    # Add header row
                    header_cells = table.rows[0].cells
                    header_cells[0].text = "Name"
                    header_cells[1].text = "Category"
                    header_cells[2].text = "Description"
                    
                    # Add name rows
                    for name_data in section_data["names"]:
                        row_cells = table.add_row().cells
                        row_cells[0].text = name_data.get("name", "")
                        row_cells[1].text = name_data.get("category", "")
                        row_cells[2].text = name_data.get("description", "")
                else:
                    doc.add_paragraph("No generated names data available.")

            elif section_name == "linguistic_analysis":
                # Format linguistic analysis using LLM
                try:
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an expert linguist analyzing brand names."),
                        HumanMessage(content=str(self.prompts["linguistic_analysis"]))
                    ])
                    content = json.loads(response.content)
                    
                    # Add the linguistic analysis content
                    if "overview" in content:
                        doc.add_paragraph(content["overview"])
                    
                    if "analysis_by_name" in content and content["analysis_by_name"]:
                        for name_analysis in content["analysis_by_name"]:
                            doc.add_heading(name_analysis["name"], level=2)
                            
                            # Pronunciation
                            if "pronunciation" in name_analysis:
                                doc.add_heading("Pronunciation", level=3)
                                doc.add_paragraph(name_analysis["pronunciation"])
                            
                            # Etymology
                            if "etymology" in name_analysis:
                                doc.add_heading("Etymology", level=3)
                                doc.add_paragraph(name_analysis["etymology"])
                            
                            # Sound symbolism
                            if "sound_symbolism" in name_analysis:
                                doc.add_heading("Sound Symbolism", level=3)
                                doc.add_paragraph(name_analysis["sound_symbolism"])
                            
                            # Other details
                            if "additional_notes" in name_analysis:
                                doc.add_heading("Additional Notes", level=3)
                                doc.add_paragraph(name_analysis["additional_notes"])
                
                except Exception as e:
                    logger.error(f"Error formatting linguistic analysis: {str(e)}")
                    doc.add_paragraph("Error generating linguistic analysis. Using available data instead.")
                    
                    # Fallback to raw data display
                    if section_data.get("names"):
                        for name_data in section_data["names"]:
                            doc.add_heading(name_data.get("name", "Unknown Name"), level=2)
                            for key, value in name_data.items():
                                if key != "name" and value:
                                    doc.add_paragraph(f"{key.replace('_', ' ').title()}: {value}")

            elif section_name == "semantic_analysis":
                # Format semantic analysis
                doc.add_heading("Semantic Analysis Overview", level=2)
                if "overview" in section_data:
                    doc.add_paragraph(section_data["overview"])
                else:
                    doc.add_paragraph("This section analyzes the meaning and connotations of each name option.")
                
                if "names" in section_data and section_data["names"]:
                    for name_data in section_data["names"]:
                        name = name_data.get("name", "Unknown Name")
                        doc.add_heading(name, level=2)
                        
                        if "meaning" in name_data:
                            doc.add_heading("Meaning", level=3)
                            doc.add_paragraph(name_data["meaning"])
                        
                        if "connotations" in name_data:
                            doc.add_heading("Connotations", level=3)
                            doc.add_paragraph(name_data["connotations"])
                        
                        if "brand_alignment" in name_data:
                            doc.add_heading("Brand Alignment", level=3)
                            doc.add_paragraph(name_data["brand_alignment"])

            elif section_name == "cultural_sensitivity":
                # Format cultural sensitivity using LLM
                try:
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an expert in cultural sensitivity analysis for brand names."),
                        HumanMessage(content=str(self.prompts["cultural_sensitivity"]))
                    ])
                    content = json.loads(response.content)
                    
                    # Add the cultural sensitivity content
                    if "overview" in content:
                        doc.add_paragraph(content["overview"])
                    
                    if "analysis_by_name" in content and content["analysis_by_name"]:
                        for name_analysis in content["analysis_by_name"]:
                            doc.add_heading(name_analysis["name"], level=2)
                            
                            if "cultural_appropriateness" in name_analysis:
                                doc.add_heading("Cultural Appropriateness", level=3)
                                doc.add_paragraph(name_analysis["cultural_appropriateness"])
                            
                            if "regional_considerations" in name_analysis:
                                doc.add_heading("Regional Considerations", level=3)
                                doc.add_paragraph(name_analysis["regional_considerations"])
                            
                            if "recommendations" in name_analysis:
                                doc.add_heading("Recommendations", level=3)
                                doc.add_paragraph(name_analysis["recommendations"])
                
                except Exception as e:
                    logger.error(f"Error formatting cultural sensitivity analysis: {str(e)}")
                    doc.add_paragraph("Error generating cultural sensitivity analysis. Using available data instead.")
                    
                    # Fallback to raw data display
                    if section_data.get("names"):
                        for name_data in section_data["names"]:
                            doc.add_heading(name_data.get("name", "Unknown Name"), level=2)
                            if "cultural_sensitivity" in name_data:
                                doc.add_paragraph(name_data["cultural_sensitivity"])

            elif section_name == "translation_analysis":
                # Format translation analysis
                doc.add_heading("Translation Analysis Overview", level=2)
                if "overview" in section_data:
                    doc.add_paragraph(section_data["overview"])
                else:
                    doc.add_paragraph("This section examines how each name option translates and is perceived in different languages.")
                
                if "names" in section_data and section_data["names"]:
                    for name_data in section_data["names"]:
                        name = name_data.get("name", "Unknown Name")
                        doc.add_heading(name, level=2)
                        
                        if "translations" in name_data and name_data["translations"]:
                            table = doc.add_table(rows=1, cols=3)
                            table.style = 'Table Grid'
                            
                            # Add header row
                            header_cells = table.rows[0].cells
                            header_cells[0].text = "Language"
                            header_cells[1].text = "Translation"
                            header_cells[2].text = "Notes"
                            
                            # Add translation rows
                            for trans in name_data["translations"]:
                                row_cells = table.add_row().cells
                                row_cells[0].text = trans.get("language", "")
                                row_cells[1].text = trans.get("translation", "")
                                row_cells[2].text = trans.get("notes", "")
                        else:
                            doc.add_paragraph("No translation data available for this name.")

            elif section_name == "name_evaluation":
                # Format name evaluation
                doc.add_heading("Name Evaluation Overview", level=2)
                if "overview" in section_data:
                    doc.add_paragraph(section_data["overview"])
                else:
                    doc.add_paragraph("This section provides a comprehensive evaluation of each name option based on key branding criteria.")
                
                if "names" in section_data and section_data["names"]:
                    # Create summary evaluation table
                    doc.add_heading("Evaluation Summary", level=2)
                    table = doc.add_table(rows=1, cols=5)
                    table.style = 'Table Grid'
                    
                    # Add header row
                    header_cells = table.rows[0].cells
                    header_cells[0].text = "Name"
                    header_cells[1].text = "Overall Score"
                    header_cells[2].text = "Strengths"
                    header_cells[3].text = "Weaknesses"
                    header_cells[4].text = "Recommendation"
                    
                    # Add name rows
                    for name_data in section_data["names"]:
                        row_cells = table.add_row().cells
                        row_cells[0].text = name_data.get("name", "")
                        row_cells[1].text = str(name_data.get("overall_score", "N/A"))
                        row_cells[2].text = name_data.get("strengths", "")
                        row_cells[3].text = name_data.get("weaknesses", "")
                        row_cells[4].text = name_data.get("recommendation", "")
                    
                    # Add detailed evaluation for each name
                    doc.add_heading("Detailed Evaluation", level=2)
                    for name_data in section_data["names"]:
                        name = name_data.get("name", "Unknown Name")
                        doc.add_heading(name, level=3)
                        
                        # Create criteria table
                        if "criteria_scores" in name_data and name_data["criteria_scores"]:
                            criteria_table = doc.add_table(rows=1, cols=2)
                            criteria_table.style = 'Table Grid'
                            
                            # Add header row
                            header_cells = criteria_table.rows[0].cells
                            header_cells[0].text = "Criterion"
                            header_cells[1].text = "Score"
                            
                            # Add criteria rows
                            for criterion, score in name_data["criteria_scores"].items():
                                row_cells = criteria_table.add_row().cells
                                row_cells[0].text = criterion.replace("_", " ").title()
                                row_cells[1].text = str(score)
                        
                        # Add comments
                        if "comments" in name_data:
                            doc.add_heading("Comments", level=4)
                            doc.add_paragraph(name_data["comments"])

            elif section_name == "domain_analysis":
                # Format domain analysis using LLM
                try:
                    brand_names = []
                    for name_data in section_data.get("names", []):
                        if "name" in name_data:
                            brand_names.append(name_data["name"])
                    
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an expert in domain analysis for brand names."),
                        HumanMessage(content=str(self.prompts["domain_analysis"].format(
                            run_id=self.current_run_id,
                            brand_names=brand_names,
                            domain_data=section_data
                        )))
                    ])
                    content = json.loads(response.content)
                    
                    # Add the domain analysis content
                    if "overview" in content:
                        doc.add_paragraph(content["overview"])
                    
                    if "analysis_by_name" in content and content["analysis_by_name"]:
                        for name_analysis in content["analysis_by_name"]:
                            doc.add_heading(name_analysis["name"], level=2)
                            
                            if "domain_availability" in name_analysis:
                                doc.add_heading("Domain Availability", level=3)
                                doc.add_paragraph(name_analysis["domain_availability"])
                            
                            if "alternative_tlds" in name_analysis:
                                doc.add_heading("Alternative TLDs", level=3)
                                doc.add_paragraph(name_analysis["alternative_tlds"])
                            
                            if "domain_history" in name_analysis:
                                doc.add_heading("Domain History", level=3)
                                doc.add_paragraph(name_analysis["domain_history"])
                            
                            if "seo_potential" in name_analysis:
                                doc.add_heading("SEO Potential", level=3)
                                doc.add_paragraph(name_analysis["seo_potential"])
                            
                            if "social_media_availability" in name_analysis:
                                doc.add_heading("Social Media Availability", level=3)
                                doc.add_paragraph(name_analysis["social_media_availability"])
                            
                            if "technical_considerations" in name_analysis:
                                doc.add_heading("Technical Considerations", level=3)
                                doc.add_paragraph(name_analysis["technical_considerations"])
                            
                            if "future_proofing" in name_analysis:
                                doc.add_heading("Future-proofing Strategy", level=3)
                                doc.add_paragraph(name_analysis["future_proofing"])
                    
                    if "comparative_analysis" in content:
                        doc.add_heading("Comparative Analysis", level=2)
                        doc.add_paragraph(content["comparative_analysis"])
                    
                    if "recommendations" in content:
                        doc.add_heading("Domain Recommendations", level=2)
                        doc.add_paragraph(content["recommendations"])
                
                except Exception as e:
                    logger.error(f"Error formatting domain analysis: {str(e)}")
                    doc.add_paragraph("Error generating domain analysis. Using available data instead.")
                    
                    # Fallback to raw data display
                    if "names" in section_data and section_data["names"]:
                        for name_data in section_data["names"]:
                            doc.add_heading(name_data.get("name", "Unknown Name"), level=2)
                            if "domains" in name_data and name_data["domains"]:
                                table = doc.add_table(rows=1, cols=3)
                                table.style = 'Table Grid'
                                
                                # Add header row
                                header_cells = table.rows[0].cells
                                header_cells[0].text = "Domain"
                                header_cells[1].text = "Availability"
                                header_cells[2].text = "Notes"
                                
                                # Add domain rows
                                for domain in name_data["domains"]:
                                    row_cells = table.add_row().cells
                                    row_cells[0].text = domain.get("name", "")
                                    row_cells[1].text = domain.get("available", "Unknown")
                                    row_cells[2].text = domain.get("notes", "")
                            else:
                                doc.add_paragraph("No domain data available for this name.")

            elif section_name == "seo_analysis":
                # Format SEO analysis using LLM
                try:
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an SEO expert analyzing brand name options."),
                        HumanMessage(content=str(self.prompts["seo_analysis"]))
                    ])
                    content = json.loads(response.content)
                    
                    # Add the SEO analysis content
                    if "overview" in content:
                        doc.add_paragraph(content["overview"])
                    
                    if "analysis_by_name" in content and content["analysis_by_name"]:
                        for name_analysis in content["analysis_by_name"]:
                            doc.add_heading(name_analysis["name"], level=2)
                            
                            if "search_volume" in name_analysis:
                                doc.add_heading("Search Volume", level=3)
                                doc.add_paragraph(name_analysis["search_volume"])
                            
                            if "keyword_competition" in name_analysis:
                                doc.add_heading("Keyword Competition", level=3)
                                doc.add_paragraph(name_analysis["keyword_competition"])
                            
                            if "online_discoverability" in name_analysis:
                                doc.add_heading("Online Discoverability", level=3)
                                doc.add_paragraph(name_analysis["online_discoverability"])
                            
                            if "recommendations" in name_analysis:
                                doc.add_heading("SEO Recommendations", level=3)
                                doc.add_paragraph(name_analysis["recommendations"])
                
                except Exception as e:
                    logger.error(f"Error formatting SEO analysis: {str(e)}")
                    doc.add_paragraph("Error generating SEO analysis. Using available data instead.")
                    
                    # Fallback to raw data display
                    if section_data.get("names"):
                        for name_data in section_data["names"]:
                            doc.add_heading(name_data.get("name", "Unknown Name"), level=2)
                            if "seo_analysis" in name_data:
                                doc.add_paragraph(name_data["seo_analysis"])

            elif section_name == "competitor_analysis":
                # Format competitor analysis
                doc.add_heading("Competitor Analysis Overview", level=2)
                if "overview" in section_data:
                    doc.add_paragraph(section_data["overview"])
                else:
                    doc.add_paragraph("This section examines how the proposed names compare to competitors in the market.")
                
                if "competitors" in section_data and section_data["competitors"]:
                    # Create competitor table
                    table = doc.add_table(rows=1, cols=3)
                    table.style = 'Table Grid'
                    
                    # Add header row
                    header_cells = table.rows[0].cells
                    header_cells[0].text = "Competitor"
                    header_cells[1].text = "Brand Name"
                    header_cells[2].text = "Analysis"
                    
                    # Add competitor rows
                    for comp in section_data["competitors"]:
                        row_cells = table.add_row().cells
                        row_cells[0].text = comp.get("company", "")
                        row_cells[1].text = comp.get("brand_name", "")
                        row_cells[2].text = comp.get("analysis", "")
                
                if "name_comparisons" in section_data and section_data["name_comparisons"]:
                    doc.add_heading("Name Comparisons", level=2)
                    for comparison in section_data["name_comparisons"]:
                        name = comparison.get("name", "Unknown Name")
                        doc.add_heading(name, level=3)
                        doc.add_paragraph(comparison.get("competitive_advantage", "No competitive advantage information available."))

            elif section_name == "market_research":
                # Format market research
                doc.add_heading("Market Research Overview", level=2)
                if "overview" in section_data:
                    doc.add_paragraph(section_data["overview"])
                else:
                    doc.add_paragraph("This section presents market research related to the brand naming process.")
                
                # Industry trends
                if "industry_trends" in section_data:
                    doc.add_heading("Industry Trends", level=2)
                    doc.add_paragraph(section_data["industry_trends"])
                
                # Target audience insights
                if "target_audience_insights" in section_data:
                    doc.add_heading("Target Audience Insights", level=2)
                    doc.add_paragraph(section_data["target_audience_insights"])
                
                # Naming trends
                if "naming_trends" in section_data:
                    doc.add_heading("Naming Trends", level=2)
                    doc.add_paragraph(section_data["naming_trends"])
                
                # Research methodology
                if "methodology" in section_data:
                    doc.add_heading("Research Methodology", level=2)
                    doc.add_paragraph(section_data["methodology"])
                
                # Key findings
                if "key_findings" in section_data:
                    doc.add_heading("Key Findings", level=2)
                    findings_list = doc.add_paragraph(style='List Bullet')
                    for finding in section_data["key_findings"]:
                        findings_list.add_run(f"{finding}\n")

            elif section_name == "survey_simulation":
                # Format survey simulation using LLM
                try:
                    brand_names = []
                    for name_data in section_data.get("names", []):
                        if "name" in name_data:
                            brand_names.append(name_data["name"])
                    
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an expert in brand market research and survey analysis."),
                        HumanMessage(content=str(self.prompts["survey_simulation"].format(
                            run_id=self.current_run_id,
                            brand_names=brand_names,
                            survey_data=section_data
                        )))
                    ])
                    content = json.loads(response.content)
                    
                    # Add methodology information
                    if "methodology" in content:
                        doc.add_heading("Methodology", level=2)
                        doc.add_paragraph(content["methodology"])
                    
                    # Add demographics information
                    if "demographics" in content:
                        doc.add_heading("Simulated Demographics", level=2)
                        doc.add_paragraph(content["demographics"])
                    
                    # Add overview insights
                    if "overview" in content:
                        doc.add_heading("Survey Overview", level=2)
                        doc.add_paragraph(content["overview"])
                    
                    # Add per-name analysis
                    if "analysis_by_name" in content and content["analysis_by_name"]:
                        doc.add_heading("Brand Name Feedback", level=2)
                        for name_analysis in content["analysis_by_name"]:
                            doc.add_heading(name_analysis["name"], level=3)
                            
                            if "brand_promise_alignment" in name_analysis:
                                doc.add_heading("Brand Promise Perception", level=4)
                                doc.add_paragraph(name_analysis["brand_promise_alignment"])
                            
                            if "personality_fit" in name_analysis:
                                doc.add_heading("Personality Fit", level=4)
                                doc.add_paragraph(name_analysis["personality_fit"])
                            
                            if "emotional_impact" in name_analysis:
                                doc.add_heading("Emotional Associations", level=4)
                                doc.add_paragraph(name_analysis["emotional_impact"])
                            
                            if "competitive_positioning" in name_analysis:
                                doc.add_heading("Competitive Differentiation", level=4)
                                doc.add_paragraph(name_analysis["competitive_positioning"])
                            
                            if "market_receptivity" in name_analysis:
                                doc.add_heading("Market Adoption Potential", level=4)
                                doc.add_paragraph(name_analysis["market_receptivity"])
                            
                            if "participant_feedback" in name_analysis:
                                doc.add_heading("Qualitative Feedback", level=4)
                                doc.add_paragraph(name_analysis["participant_feedback"])
                            
                            # Handle recommendations - check for new field name first, then fall back to old name
                            if "final_recommendations" in name_analysis:
                                doc.add_heading("Final Recommendations", level=4)
                                doc.add_paragraph(name_analysis["final_recommendations"])
                            elif "recommendations" in name_analysis:
                                doc.add_heading("Recommendations", level=4)
                                doc.add_paragraph(name_analysis["recommendations"])
                    
                    # Add comparative analysis
                    if "comparative_analysis" in content:
                        doc.add_heading("Comparative Analysis", level=2)
                        doc.add_paragraph(content["comparative_analysis"])
                    
                    # Add strategic implications
                    if "strategic_implications" in content:
                        doc.add_heading("Strategic Implications", level=2)
                        doc.add_paragraph(content["strategic_implications"])
                
                except Exception as e:
                    logger.error(f"Error formatting survey simulation: {str(e)}")
                    doc.add_paragraph("Error generating survey simulation analysis. Using available data instead.")
                    
                    # Fallback to raw data display
                    doc.add_heading("Survey Methodology", level=2)
                    if "methodology" in section_data:
                        doc.add_paragraph(section_data["methodology"])
                    else:
                        doc.add_paragraph("No methodology data available.")
                    
                    # Display survey results if available
                    if "results" in section_data:
                        doc.add_heading("Survey Results", level=2)
                        for name, result in section_data["results"].items():
                            doc.add_heading(name, level=3)
                            for metric, value in result.items():
                                doc.add_paragraph(f"{metric.replace('_', ' ').title()}: {value}")
                    
                    # Display insights if available
                    if "insights" in section_data:
                        doc.add_heading("Survey Insights", level=2)
                        doc.add_paragraph(section_data["insights"])

            elif section_name == "recommendations":
                # Format recommendations using LLM
                try:
                    response = await self.llm.ainvoke([
                        SystemMessage(content="You are an expert brand consultant providing strategic recommendations based on a comprehensive brand naming analysis."),
                        HumanMessage(content=str(self.prompts["recommendations"]))
                    ])
                    content = json.loads(response.content)
                    
                    # Add the recommendations content
                    if "summary" in content:
                        doc.add_paragraph(content["summary"])
                    
                    if "top_names" in content and content["top_names"]:
                        doc.add_heading("Top Recommended Names", level=2)
                        for idx, name_rec in enumerate(content["top_names"], 1):
                            doc.add_heading(f"{idx}. {name_rec['name']}", level=3)
                            doc.add_paragraph(name_rec["rationale"])
                    
                    if "implementation_steps" in content:
                        doc.add_heading("Implementation Steps", level=2)
                        steps_list = doc.add_paragraph(style='List Number')
                        for step in content["implementation_steps"]:
                            steps_list.add_run(f"{step}\n")
                    
                    if "additional_recommendations" in content:
                        doc.add_heading("Additional Recommendations", level=2)
                        doc.add_paragraph(content["additional_recommendations"])
                
                except Exception as e:
                    logger.error(f"Error formatting recommendations: {str(e)}")
                    doc.add_paragraph("Error generating recommendations. Using available data instead.")
                    
                    # Fallback to simple recommendations
                    if "top_names" in section_data:
                        doc.add_heading("Top Recommended Names", level=2)
                        for idx, name in enumerate(section_data["top_names"], 1):
                            doc.add_paragraph(f"{idx}. {name}")
                    
                    if "rationale" in section_data:
                        doc.add_heading("Recommendation Rationale", level=2)
                        doc.add_paragraph(section_data["rationale"])
                    
                    if "next_steps" in section_data:
                        doc.add_heading("Next Steps", level=2)
                        steps_list = doc.add_paragraph(style='List Number')
                        for step in section_data["next_steps"]:
                            steps_list.add_run(f"{step}\n")
            
            else:
                # Generic section formatting for any other sections
                for key, value in section_data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                    elif isinstance(value, list) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        bullet_list = doc.add_paragraph(style='List Bullet')
                        for item in value:
                            if isinstance(item, str):
                                bullet_list.add_run(f"{item}\n")
                            elif isinstance(item, dict):
                                # For list of dictionaries, create a more complex structure
                                for subkey, subvalue in item.items():
                                    if isinstance(subvalue, str):
                                        doc.add_paragraph(f"{subkey.replace('_', ' ').title()}: {subvalue}")
            
            # Add page break after each section
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            return True, None
            
        except Exception as e:
            error_msg = f"Error formatting section {section_name}: {str(e)}"
            logger.error(error_msg)
            doc.add_paragraph(f"Error formatting section. Please check the data and try again.")
            doc.add_paragraph(f"Technical details: {str(e)}")
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            return False, error_msg

    async def _generate_brand_context_content(self, brand_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured brand context content using the LLM."""
        try:
            # Prepare the prompt
            prompt = self.prompts["brand_context"].format(
                run_id=self.current_run_id,
                brand_context=brand_context
            )
            
            # Get response from LLM
            messages = [
                SystemMessage(content="You are an expert brand strategist providing context for a brand naming report."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
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
                    logger.warning("Could not extract JSON from LLM response for brand context")
                    return {
                        "brand_promise_section": "Error extracting structured content from LLM response.",
                        "brand_personality_section": "",
                        "brand_tone_section": "",
                        "brand_values_section": "",
                        "brand_purpose_section": "",
                        "brand_mission_section": "",
                        "target_audience_section": "",
                        "customer_needs_section": "",
                        "market_positioning_section": "",
                        "competitive_landscape_section": "",
                        "industry_focus_section": "",
                        "industry_trends_section": "",
                        "brand_identity_summary": ""
                    }
                    
        except Exception as e:
            logger.error(f"Error generating brand context content: {str(e)}")
            return {
                "brand_promise_section": f"Error generating brand context content: {str(e)}",
                "brand_personality_section": "",
                "brand_tone_section": "",
                "brand_values_section": "",
                "brand_purpose_section": "",
                "brand_mission_section": "",
                "target_audience_section": "",
                "customer_needs_section": "",
                "market_positioning_section": "",
                "competitive_landscape_section": "",
                "industry_focus_section": "",
                "industry_trends_section": "",
                "brand_identity_summary": ""
            }

    async def _format_brand_context(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the brand context section using LLM."""
        if not data:
            doc.add_paragraph("No brand context data available.", style='Quote')
            return
            
        # Generate enhanced content using LLM
        try:
            enhanced_content = await self._generate_brand_context_content(data)
            
            # Add each section from the enhanced content
            for section_title, section_key in [
                ("Brand Promise", "brand_promise_section"),
                ("Brand Personality", "brand_personality_section"),
                ("Brand Tone of Voice", "brand_tone_section"),
                ("Brand Values", "brand_values_section"),
                ("Brand Purpose", "brand_purpose_section"),
                ("Brand Mission", "brand_mission_section"),
                ("Target Audience", "target_audience_section"),
                ("Customer Needs", "customer_needs_section"),
                ("Market Positioning", "market_positioning_section"),
                ("Competitive Landscape", "competitive_landscape_section"),
                ("Industry Focus", "industry_focus_section"),
                ("Industry Trends", "industry_trends_section")
            ]:
                if section_key in enhanced_content and enhanced_content[section_key]:
                    doc.add_heading(section_title, level=2)
                    doc.add_paragraph(enhanced_content[section_key])
                
            # Add brand identity summary at the end
            if "brand_identity_summary" in enhanced_content and enhanced_content["brand_identity_summary"]:
                doc.add_heading("Brand Identity Summary", level=2)
                doc.add_paragraph(enhanced_content["brand_identity_summary"])
                
        except Exception as e:
            logger.error(f"Error formatting brand context with LLM: {str(e)}")
            doc.add_paragraph(f"⚠️ Error enhancing brand context with LLM: {str(e)}", style='Intense Quote')
            
            # Fall back to basic formatting
            self._format_brand_context_basic(doc, data)
    
    def _format_brand_context_basic(self, doc: Document, data: Dict[str, Any]) -> None:
        """Basic formatting for brand context when LLM enhancement fails."""
        # Extract brand context data (might be nested)
        if "brand_context" in data and isinstance(data["brand_context"], dict):
            context = data["brand_context"]
        else:
            context = data
        
        # Define fields to display
        fields = [
            ("Brand Promise", "brand_promise"),
            ("Brand Personality", "brand_personality"),
            ("Brand Tone of Voice", "brand_tone_of_voice"),
            ("Brand Values", "brand_values"),
            ("Brand Purpose", "brand_purpose"),
            ("Brand Mission", "brand_mission"),
            ("Target Audience", "target_audience"),
            ("Customer Needs", "customer_needs"),
            ("Market Positioning", "market_positioning"),
            ("Competitive Landscape", "competitive_landscape"),
            ("Industry Focus", "industry_focus"),
            ("Industry Trends", "industry_trends"),
            ("Brand Identity Brief", "brand_identity_brief")
        ]
        
        # Add each field
        for display_name, field_name in fields:
            if field_name in context and context[field_name]:
                doc.add_heading(display_name, level=2)
                doc.add_paragraph(context[field_name])

    async def _generate_name_generation_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced name generation content using the LLM."""
        try:
            # Prepare the prompt
            prompt = self.prompts["brand_name_generation"].format(
                run_id=self.current_run_id,
                brand_name_generation=data
            )
            
            # Get response from LLM
            messages = [
                SystemMessage(content="You are an expert brand naming specialist presenting brand name options."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
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
                    logger.warning("Could not extract JSON from LLM response for name generation")
                    return {
                        "introduction": "Error extracting structured content from LLM response.",
                        "categories": [],
                        "summary": ""
                    }
                    
        except Exception as e:
            logger.error(f"Error generating name generation content: {str(e)}")
            return {
                "introduction": f"Error generating name generation content: {str(e)}",
                "categories": [],
                "summary": ""
            }

    async def _format_name_generation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name generation section using LLM."""
        if not data:
            doc.add_paragraph("No name generation data available.", style='Quote')
            return
            
        # Generate enhanced content using LLM
        try:
            enhanced_content = await self._generate_name_generation_content(data)
            
            # Add introduction
            if "introduction" in enhanced_content and enhanced_content["introduction"]:
                doc.add_paragraph(enhanced_content["introduction"])
            
            # Add categories
            if "categories" in enhanced_content and enhanced_content["categories"]:
                for category in enhanced_content["categories"]:
                    # Add category heading
                    if "category_name" in category and category["category_name"]:
                        doc.add_heading(category["category_name"], level=2)
                    
                    # Add category description
                    if "category_description" in category and category["category_description"]:
                        doc.add_paragraph(category["category_description"])
                    
                    # Add names in this category
                    if "names" in category and category["names"]:
                        for name_data in category["names"]:
                            if "name" in name_data and name_data["name"]:
                                # Create a heading for the name
                                doc.add_heading(name_data["name"], level=3)
                                
                                # Create a table for the name details
                                table = doc.add_table(rows=8, cols=2)
                                table.style = 'Table Grid'
                                
                                # Define fields to display in the table
                                fields = [
                                    ("Personality Alignment", "personality_alignment"),
                                    ("Promise Alignment", "promise_alignment"),
                                    ("Methodology", "methodology"),
                                    ("Memorability", "memorability"),
                                    ("Pronounceability", "pronounceability"),
                                    ("Visual Potential", "visual_potential"),
                                    ("Audience Relevance", "audience_relevance"),
                                    ("Market Differentiation", "market_differentiation")
                                ]
                                
                                # Add fields to table
                                for i, (display_name, field_name) in enumerate(fields):
                                    if i < len(table.rows):
                                        row = table.rows[i].cells
                                        row[0].text = display_name
                                        row[0].paragraphs[0].runs[0].bold = True
                                        row[1].text = name_data.get(field_name, "")
                                
                                # Add some space after each name
                                doc.add_paragraph()
            
            # Add summary
            if "summary" in enhanced_content and enhanced_content["summary"]:
                doc.add_heading("Summary", level=2)
                doc.add_paragraph(enhanced_content["summary"])
                
        except Exception as e:
            logger.error(f"Error formatting name generation with LLM: {str(e)}")
            doc.add_paragraph(f"⚠️ Error enhancing name generation with LLM: {str(e)}", style='Intense Quote')
            
            # Fall back to basic formatting
            self._format_name_generation_basic(doc, data)
    
    def _format_name_generation_basic(self, doc: Document, data: Dict[str, Any]) -> None:
        """Basic formatting for name generation when LLM enhancement fails."""
        # Extract name generation data
        if "brand_name_generation" in data and isinstance(data["brand_name_generation"], dict):
            categories = data["brand_name_generation"]
        else:
            categories = data
        
        # Add introduction
        doc.add_paragraph(
            f"This section presents the generated brand names organized by naming category. "
            f"Each name is evaluated on multiple criteria including brand alignment, memorability, "
            f"pronounceability, and market differentiation."
        )
        
        # Process each category
        if isinstance(categories, dict):
            for category_name, names in categories.items():
                if isinstance(names, list) and names:
                    # Add category heading
                    doc.add_heading(category_name, level=2)
                    
                    # Process each name in this category
                    for name_data in names:
                        if isinstance(name_data, dict) and "brand_name" in name_data:
                            # Create a heading for the name
                            doc.add_heading(name_data["brand_name"], level=3)
                            
                            # Create a table for the name details
                            table = doc.add_table(rows=5, cols=2)
                            table.style = 'Table Grid'
                            
                            # Define fields to display in the table
                            fields = [
                                ("Brand Personality Alignment", "brand_personality_alignment"),
                                ("Brand Promise Alignment", "brand_promise_alignment"),
                                ("Methodology", "name_generation_methodology"),
                                ("Memorability", "memorability_score_details"),
                                ("Pronounceability", "pronounceability_score_details")
                            ]
                            
                            # Add fields to table
                            for i, (display_name, field_name) in enumerate(fields):
                                if i < len(table.rows):
                                    row = table.rows[i].cells
                                    row[0].text = display_name
                                    row[1].text = name_data.get(field_name, "")
                            
                            # Add some space after each name
                            doc.add_paragraph()


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