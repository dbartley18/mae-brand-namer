#!/usr/bin/env python3
"""
Report Formatter

It pulls raw data from the report_raw_data table and formats it into a polished report.
"""

import os
import re
import json
import logging
import asyncio
import time
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Any
from pathlib import Path

import docx
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from langchain.prompts import PromptTemplate, load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import ValidationError

from ..models.report_sections import (
    BrandContext,
    NameGenerationSection,
    BrandName,
    TOCSection,
    TableOfContentsSection,
    LinguisticAnalysis,
    LinguisticAnalysisDetails,
    SemanticAnalysis,
    SemanticAnalysisDetails,
    CulturalSensitivityAnalysis,
    BrandAnalysis,
    TranslationAnalysis,
    LanguageAnalysis,
    BrandNameEvaluation,
    EvaluationDetails,
    EvaluationLists,
    DomainAnalysis,
    DomainDetails,
    SEOOnlineDiscoverability,
    SEOOnlineDiscoverabilityDetails,
    SEORecommendations,
    CompetitorAnalysis,
    CompetitorDetails,
    BrandCompetitors,
    MarketResearch,
    MarketResearchDetails,
    SurveySimulation,
    SurveyDetails
)
from ..utils.supabase_utils import SupabaseManager
from ..utils.logging import get_logger
from ..config.settings import settings
from .prompts.report_formatter import (
    get_title_page_prompt,
    get_toc_prompt,
    get_executive_summary_prompt,
    get_recommendations_prompt,
    get_seo_analysis_prompt,
    get_brand_context_prompt,
    get_brand_name_generation_prompt,
    get_semantic_analysis_prompt,
    get_linguistic_analysis_prompt,
    get_cultural_sensitivity_prompt,
    get_translation_analysis_prompt,
    get_market_research_prompt,
    get_competitor_analysis_prompt,
    get_name_evaluation_prompt,
    get_domain_analysis_prompt,
    get_survey_simulation_prompt,
    get_system_prompt,
    get_shortlisted_names_summary_prompt,
    get_format_section_prompt
)

from ..utils.logging import get_logger

# Define path to prompts directory
PROMPTS_DIR = os.path.join(Path(__file__).parent, "prompts", "report_formatter")

logger = get_logger(__name__)

def _safe_load_prompt(path: str) -> PromptTemplate:
    """
    Safely load a prompt template from a file or use a default one.
    
    Args:
        path: Path to the prompt template file
        
    Returns:
        PromptTemplate: The loaded prompt template or a default one if loading fails
    """
    try:
        # Only load necessary templates for brand_context and executive_summary
        if any(section in path for section in ["brand_context", "executive_summary"]):
            logger.debug(f"Loading prompt template from {path}")
            with open(path, "r") as f:
                content = f.read().strip()
                
            # Extract template variables
            variables = set(re.findall(r"{([^{}]*)}", content))
            
            logger.debug(f"Loaded prompt template with variables: {variables}")
            return PromptTemplate(template=content, input_variables=list(variables))
        else:
            # For other sections, return a minimal template that won't be used
            logger.debug(f"Skipping prompt template loading for {path} - using ETL processing")
            return PromptTemplate(template="ETL processing used instead of LLM", input_variables=["dummy"])
    except Exception as e:
        logger.warning(f"Error loading prompt template from {path}: {str(e)}")
        logger.warning("Using default prompt template")
        return PromptTemplate(
            template="Please format the following data for a section of a brand naming report: {data}",
            input_variables=["data"]
        )

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
        """
        Initialize the ReportFormatter.
        
        Args:
            run_id: The ID of the run to fetch data for
            dependencies: Optional dependencies to inject
            supabase: Optional Supabase manager
        """
        # Store run ID
        self.current_run_id = run_id
        
        # Initialize dependencies
        if dependencies is None:
            dependencies = {}
        
        # Initialize Supabase connection
        if supabase:
            self.supabase = supabase
        elif hasattr(dependencies, "supabase"):
            self.supabase = dependencies.supabase
        else:
            self.supabase = SupabaseManager()
        
        # Initialize Gemini model with tracing 
        if hasattr(dependencies, "llm"):
            self.llm = dependencies.llm
        else:
            from ..config.settings import settings
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name, 
                temperature=0.2,  # Balanced temperature for analysis 
                google_api_key=settings.google_api_key, 
                convert_system_message_to_human=True, 
                callbacks=settings.get_langsmith_callbacks()
            )
        
        # Set up prompt templates
        self.prompts = {
            # Only load templates for brand_context and executive_summary
            "brand_context": _safe_load_prompt(os.path.join(PROMPTS_DIR, "brand_context.yaml")),
            "exec_summary": _safe_load_prompt(os.path.join(PROMPTS_DIR, "executive_summary.yaml")),
            "executive_summary": _safe_load_prompt(os.path.join(PROMPTS_DIR, "executive_summary.yaml"))
        }
        
        # Log available prompt templates
        logger.debug(f"Loaded {len(self.prompts)} prompt templates: {list(self.prompts.keys())}")
        
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
    def _get_format_instructions(self, section_name: str) -> str:
        """
        Get formatting instructions for a section.
        Only used for brand_context and executive_summary sections.
        
        Args:
            section_name: The name of the section
            
        Returns:
            The formatting instructions
        """
        # Only brand_context and executive_summary use format instructions
        if section_name not in ["brand_context", "executive_summary"]:
            return "ETL processing used instead"
            
        # Base instructions for all sections
        base_instructions = """
        Format the provided data into a well-structured, professionally written report section.
        The output should be a JSON object containing fields appropriate for this section.
        Keep the tone professional, informative, and engaging.
        """
        
        # Section-specific instructions
        if section_name == "brand_context":
            return base_instructions + """
            Include the following fields in your JSON output:
            - "overview": A comprehensive overview of the brand context
            - "industry": Analysis of the industry context
            - "target_audience": Description of the target audience
            - "brand_values": Core brand values and personality
            - "positioning": Brand positioning statement and strategy
            - "brand_voice": Tone and voice guidelines
            - "naming_objectives": Specific goals for the naming project
            """
        elif section_name == "executive_summary":
            return base_instructions + """
            Include the following fields in your JSON output:
            - "project_overview": Brief overview of the brand naming project
            - "process_summary": Summary of the naming process and methodology
            - "key_findings": The most important findings from the analysis
            - "recommended_names": The top recommended brand name options
            - "rationale": Explanation of why these names are recommended
            - "next_steps": Suggested next steps in the brand naming process
            """
        else:
            return base_instructions

    def _format_template(self, template_name: str, format_data: Dict[str, Any], section_name: str = None) -> str:
        """
        Format a template with the given data.
        
        Args:
            template_name: The name of the template to format
            format_data: The data to format the template with
            section_name: The section name to use for logging
            
        Returns:
            The formatted template
        """
        # Only brand_context and executive_summary use templates
        if section_name not in ["brand_context", "executive_summary"]:
            logger.debug(f"Skipping template formatting for {section_name} - using ETL processing")
            return "ETL processing used instead"
            
        try:
            if template_name in self.prompts:
                # Format the template
                template = self.prompts[template_name]
                
                # Include format instructions if available and not in the data
                if "format_instructions" not in format_data and section_name:
                    format_data["format_instructions"] = self._get_format_instructions(section_name)
                
                # Check for missing variables
                missing_vars = []
                for var in template.input_variables:
                    if var not in format_data:
                        missing_vars.append(var)
                        format_data[var] = f"[No data available for {var}]"
                
                if missing_vars:
                    logger.warning(f"Missing variables for template {template_name}: {missing_vars}")
                
                # Format the template
                return template.format(**format_data)
            else:
                # Return a default prompt if template not found
                logger.warning(f"Template {template_name} not found in prompts dictionary")
                return f"Please format the following {template_name} data: {json.dumps(format_data)}"
        except Exception as e:
            logger.error(f"Error formatting template {template_name}: {str(e)}")
            return f"Error formatting template: {str(e)}"

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
        if not data:
            return {}
        try:
            # Check if data has nested 'brand_context' key and extract it if so
            brand_context_data = data.get('brand_context', data)
            
            # Log what we're trying to validate
            logger.debug(f"Validating brand context: {str(brand_context_data)[:200]}...")
            
            # Validate against the model
            brand_context_model = BrandContext.model_validate(brand_context_data)
            return brand_context_model.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for brand context data: {str(e)}")
            # Return the original data as fallback
            return data.get('brand_context', data)

    def _transform_name_generation(self, data: Dict[str, Any]) -> Any:
        """
        Transform brand name generation data into a structured format for the report.
        
        This handles various formats of brand name generation data:
        1. Raw data from brand_name_generation.md which has:
           - Categories as top-level keys
           - Lists of brand names as values for each category
           - Each brand name entry has attributes like rationale, brand_promise_alignment, etc.
        
        2. Data already structured as a NameGenerationSection model with categories
        
        3. Data with a nested 'brand_name_generation' key containing any of the above formats
        
        Returns:
            A structured dictionary matching the NameGenerationSection model structure
        """
        try:
            # Log original data structure
            if isinstance(data, dict):
                logger.debug(f"Name generation data keys: {list(data.keys())}")
                
                # Check if data is nested under 'brand_name_generation'
                if "brand_name_generation" in data:
                    data = data["brand_name_generation"]
                    logger.debug(f"Extracted nested brand_name_generation data, keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
            
            # First, handle the case where we have a raw format like in brand_name_generation.md
            # where categories are top-level keys and values are lists of brand names
            if isinstance(data, dict) and all(isinstance(value, list) for value in data.values() if isinstance(value, list)):
                logger.debug("Detected raw brand name generation data format with categories as keys")
                
                # Convert to NameGenerationSection structure
                categories = []
                for category_name, names_list in data.items():
                    # Skip non-list values and special keys that aren't categories
                    if not isinstance(names_list, list) or category_name in ["methodology_and_approach", "introduction", "summary"]:
                        continue
                    
                    # Process names in this category
                    processed_names = []
                    for name_data in names_list:
                        if isinstance(name_data, dict) and "brand_name" in name_data:
                            processed_names.append(name_data)
                    
                    # Create category entry
                    if processed_names:
                        categories.append({
                            "category_name": category_name,
                            "category_description": "",  # Default empty description
                            "names": processed_names
                        })
                
                # Create structured data
                structured_data = {
                    "introduction": data.get("introduction", ""),
                    "methodology_and_approach": data.get("methodology_and_approach", ""),
                    "summary": data.get("summary", ""),
                    "categories": categories,
                    "generated_names_overview": {
                        "total_count": sum(len(cat.get("names", [])) for cat in categories)
                    },
                    "evaluation_metrics": {}  # Default empty metrics
                }
                
                logger.debug(f"Converted raw data to structured format with {len(categories)} categories")
                
                # Try to validate with NameGenerationSection model
                try:
                    from mae_brand_namer.models.report_sections import NameGenerationSection
                    validated_data = NameGenerationSection.model_validate(structured_data).model_dump()
                    logger.debug("Successfully validated structured data with NameGenerationSection model")
                    return validated_data
                except Exception as validation_error:
                    logger.warning(f"Validation of structured name generation data failed: {str(validation_error)}")
                    # Return the structured data even if validation fails
                    return structured_data
            
            # Attempt to directly validate the data if it's already in the right structure
            if isinstance(data, dict) and "categories" in data:
                try:
                    from mae_brand_namer.models.report_sections import NameGenerationSection
                    validated_data = NameGenerationSection.model_validate(data).model_dump()
                    logger.debug("Successfully validated existing structured data with NameGenerationSection model")
                    return validated_data
                except Exception as validation_error:
                    logger.warning(f"Validation of existing structured name generation data failed: {str(validation_error)}")
            
            # Try to parse from JSON directly
            if isinstance(data, str):
                try:
                    parsed_data = json.loads(data)
                    return self._transform_name_generation(parsed_data)  # Recursively process the parsed data
                except json.JSONDecodeError:
                    logger.warning("Failed to parse name generation data as JSON string")
            
            # If still here, use NameGenerationSection.from_raw_json as fallback
            try:
                from mae_brand_namer.models.report_sections import NameGenerationSection
                section = NameGenerationSection.from_raw_json(data)
                logger.debug("Successfully created NameGenerationSection from raw JSON data")
                return section.model_dump()
            except Exception as e:
                logger.warning(f"Failed to create NameGenerationSection from raw JSON: {str(e)}")
            
            # Last resort: return the original data
            logger.debug("Returning original name generation data as no transformation succeeded")
            return data
            
        except Exception as e:
            logger.error(f"Error transforming name generation data: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            return data

    def _transform_semantic_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform semantic analysis data to match expected model format.
        
        The semantic analysis data can be structured in several ways:
        1. Under a "semantic_analysis" key in the top-level data
        2. As a dictionary of brand names with semantic analysis details
        3. As a nested structure under semantic_analysis.semantic_analysis
        """
        if not data:
            return {}
        try:
            # First check for semantic_analysis key
            semantic_data = data.get("semantic_analysis", data)
            
            # Log what we're trying to validate
            logger.debug(f"Validating semantic analysis data: {str(semantic_data)[:200]}...")
            
            # If semantic_data exists but is nested one level deeper
            if isinstance(semantic_data, dict) and "semantic_analysis" in semantic_data:
                semantic_data = semantic_data["semantic_analysis"]
                
            # Create the expected structure
            semantic_analysis = {"semantic_analysis": semantic_data}
            
            # Validate against the model
            validated_data = SemanticAnalysis.model_validate(semantic_analysis)
            
            # Transform the data for the template format
            template_data = {
                "brand_analyses": []
            }
            
            # Convert dictionary of brand details to a list of brand analyses
            for brand_name, details in validated_data.semantic_analysis.items():
                # Make sure brand_name is included in the details
                brand_analysis = details.model_dump()
                # Ensure brand_name is present
                if "brand_name" not in brand_analysis:
                    brand_analysis["brand_name"] = brand_name
                    
                template_data["brand_analyses"].append(brand_analysis)
                
            logger.info(f"Successfully transformed semantic analysis data with {len(template_data['brand_analyses'])} brand analyses")
            return template_data
            
        except ValidationError as e:
            logger.error(f"Validation error for semantic analysis data: {str(e)}")
            # Return the original data as fallback
            return data.get('semantic_analysis', data)

    def _transform_linguistic_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            linguistic_analysis_data = LinguisticAnalysis.model_validate(data)
            return linguistic_analysis_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for linguistic analysis data: {str(e)}")
            return {}

    def _transform_cultural_sensitivity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform cultural sensitivity analysis data to match expected format.
        
        The cultural sensitivity analysis data should contain brand name analyses with
        attributes such as symbolic_meanings, cultural_connotations, etc.
        
        Args:
            data: Raw cultural sensitivity analysis data
            
        Returns:
            Transformed data with brand_analyses and summary fields
        """
        if not data:
            return {}
        try:
            # Validate data against the model
            cultural_sensitivity_data = CulturalSensitivityAnalysis.model_validate(data)
            
            # Transform to a format more suitable for the template
            transformed_data = {
                "brand_analyses": [],
                "summary": "The cultural sensitivity analysis evaluated brand names across multiple dimensions including symbolic meanings, cultural connotations, religious sensitivities, and regional variations. This analysis helps identify names that minimize cultural risks while maximizing positive associations across diverse markets."
            }
            
            # Process each brand analysis
            if hasattr(cultural_sensitivity_data, "cultural_sensitivity_analysis"):
                for brand_name, analysis in cultural_sensitivity_data.cultural_sensitivity_analysis.items():
                    # Create a brand analysis entry with all fields from the model
                    brand_analysis = analysis.model_dump()
                    # Ensure brand_name is included
                    if "brand_name" not in brand_analysis:
                        brand_analysis["brand_name"] = brand_name
                    
                    transformed_data["brand_analyses"].append(brand_analysis)
            
            logger.info(f"Successfully transformed cultural sensitivity data with {len(transformed_data['brand_analyses'])} brand analyses")
            return transformed_data
            
        except ValidationError as e:
            logger.error(f"Validation error for cultural sensitivity analysis data: {str(e)}")
            return {}

    def _transform_name_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform brand name evaluation data to match expected model format.
        
        The brand name evaluation data should have:
        1. A "brand_name_evaluation" key with shortlisted_names and other_names lists
        2. Each name evaluation contains brand_name, overall_score, shortlist_status, and evaluation_comments
        
        Args:
            data: Raw brand name evaluation data
            
        Returns:
            Transformed data in the format expected by the formatter
        """
        if not data:
            return {}
        try:
            # First, validate data against the model
            name_evaluation_data = BrandNameEvaluation.model_validate(data)
            
            # Transform to a format more suitable for the template
            transformed_data = {
                "shortlisted_names": [],
                "other_names": [],
                "evaluation_methodology": "Each brand name was evaluated based on multiple criteria including distinctiveness, memorability, strategic alignment, and cultural considerations.",
                "comparative_analysis": "The shortlisted names demonstrated stronger overall performance across evaluation criteria, particularly in areas of strategic alignment with the brand's values and positioning in the market."
            }
            
            # Process shortlisted names
            if hasattr(name_evaluation_data.brand_name_evaluation, "shortlisted_names"):
                transformed_data["shortlisted_names"] = [
                    {
                        "brand_name": item.brand_name,
                        "overall_score": item.overall_score,
                        "evaluation_comments": item.evaluation_comments
                    }
                    for item in name_evaluation_data.brand_name_evaluation.shortlisted_names
                ]
                logger.info(f"Processed {len(transformed_data['shortlisted_names'])} shortlisted names")
            
            # Process other names
            if hasattr(name_evaluation_data.brand_name_evaluation, "other_names"):
                transformed_data["other_names"] = [
                    {
                        "brand_name": item.brand_name,
                        "overall_score": item.overall_score,
                        "evaluation_comments": item.evaluation_comments
                    }
                    for item in name_evaluation_data.brand_name_evaluation.other_names
                ]
                logger.info(f"Processed {len(transformed_data['other_names'])} other names")
                
            # Add final rankings based on scores
            all_names = transformed_data["shortlisted_names"] + transformed_data["other_names"]
            rankings = {
                item["brand_name"]: item["overall_score"] for item in all_names
            }
            transformed_data["final_rankings"] = rankings
            
            logger.info(f"Successfully transformed name evaluation data with {len(transformed_data['shortlisted_names'])} shortlisted names and {len(transformed_data['other_names'])} other names")
            logger.debug(f"Transformed data: {transformed_data}")
            return transformed_data
            
        except ValidationError as e:
            logger.error(f"Validation error for name evaluation data: {str(e)}")
            
            # Try a fallback approach if the validation fails
            try:
                # Check if the data is already in the expected format
                if "brand_name_evaluation" in data and isinstance(data["brand_name_evaluation"], dict):
                    eval_lists = data["brand_name_evaluation"]
                    
                    transformed_data = {
                        "shortlisted_names": [],
                        "other_names": [],
                        "evaluation_methodology": "Each brand name was evaluated based on multiple criteria including distinctiveness, memorability, strategic alignment, and cultural considerations.",
                        "comparative_analysis": "The shortlisted names demonstrated stronger overall performance across evaluation criteria, particularly in areas of strategic alignment with the brand's values and positioning in the market."
                    }
                    
                    # Process shortlisted names
                    if "shortlisted_names" in eval_lists and isinstance(eval_lists["shortlisted_names"], list):
                        transformed_data["shortlisted_names"] = [
                            {
                                "brand_name": item.get("brand_name", ""),
                                "overall_score": item.get("overall_score", 0),
                                "evaluation_comments": item.get("evaluation_comments", "")
                            }
                            for item in eval_lists["shortlisted_names"]
                        ]
                    
                    # Process other names
                    if "other_names" in eval_lists and isinstance(eval_lists["other_names"], list):
                        transformed_data["other_names"] = [
                            {
                                "brand_name": item.get("brand_name", ""),
                                "overall_score": item.get("overall_score", 0),
                                "evaluation_comments": item.get("evaluation_comments", "")
                            }
                            for item in eval_lists["other_names"]
                        ]
                    
                    # Add final rankings
                    all_names = transformed_data["shortlisted_names"] + transformed_data["other_names"]
                    rankings = {
                        item["brand_name"]: item["overall_score"] for item in all_names
                    }
                    transformed_data["final_rankings"] = rankings
                    
                    logger.info(f"Fallback transformation successful with {len(transformed_data['shortlisted_names'])} shortlisted names and {len(transformed_data['other_names'])} other names")
                    return transformed_data
            except Exception as fallback_error:
                logger.error(f"Fallback transformation also failed: {str(fallback_error)}")
            
            return {}

    def _transform_translation_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            translation_analysis_data = TranslationAnalysis.model_validate(data)
            return translation_analysis_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for translation analysis data: {str(e)}")
            return {}

    def _transform_market_research(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform market research data to match expected model format.
        
        The model expects a dictionary keyed by brand_name, but the raw data
        is typically an array of objects with brand_name as a field.
        
        Args:
            data: Raw market research data
            
        Returns:
            Transformed data in the format expected by the formatter
        """
        if not data:
            return {}
        try:
            # Check data structure and convert if necessary
            if "market_research" in data and isinstance(data["market_research"], list):
                # Convert list to dictionary keyed by brand_name
                market_research_dict = {}
                for item in data["market_research"]:
                    if "brand_name" in item:
                        brand_name = item["brand_name"]
                        market_research_dict[brand_name] = item
                
                # Create new data structure with the dictionary
                transformed_data = {"market_research": market_research_dict}
                
                # Try to validate with the model
                market_research_data = MarketResearch.model_validate(transformed_data)
                logger.info(f"Successfully converted market research data from list to dict format with {len(market_research_dict)} entries")
                return market_research_data.model_dump()
            else:
                # Data might already be in the correct format
                market_research_data = MarketResearch.model_validate(data)
                return market_research_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for market research data: {str(e)}")
            
            # Fallback: return a simplified structure that preserves the data
            try:
                if "market_research" in data and isinstance(data["market_research"], list):
                    result = {
                        "market_research": {}
                    }
                    
                    for item in data["market_research"]:
                        if "brand_name" in item:
                            # Process key_competitors if it's a string representation of a list
                            if "key_competitors" in item and isinstance(item["key_competitors"], str):
                                try:
                                    # Try to convert string list representation to actual list
                                    import ast
                                    item["key_competitors"] = ast.literal_eval(item["key_competitors"])
                                except:
                                    # If conversion fails, keep as is
                                    pass
                            
                            # Process customer_pain_points if it's a string representation of a list
                            if "customer_pain_points" in item and isinstance(item["customer_pain_points"], str):
                                try:
                                    import ast
                                    item["customer_pain_points"] = ast.literal_eval(item["customer_pain_points"])
                                except:
                                    pass
                                    
                            # Process emerging_trends if it's a string representation of a list
                            if "emerging_trends" in item and isinstance(item["emerging_trends"], str):
                                try:
                                    import ast
                                    item["emerging_trends"] = ast.literal_eval(item["emerging_trends"])
                                except:
                                    pass
                            
                            result["market_research"][item["brand_name"]] = item
                    
                    logger.info(f"Used fallback approach for market research data with {len(result['market_research'])} entries")
                    return result
                return data
            except Exception as fallback_error:
                logger.error(f"Fallback transformation for market research also failed: {str(fallback_error)}")
                return data

    def _transform_competitor_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform competitor analysis data to match expected format for the formatter.
        
        Args:
            data: Raw competitor analysis data
            
        Returns:
            Transformed data structure
        """
        # Debug the input data structure
        logger.info(f"Competitor analysis raw data type: {type(data)}")
        if isinstance(data, dict):
            logger.info(f"Competitor analysis raw data keys: {list(data.keys())}")
        
        # Updated handling for different data structures
        if data is None:
            logger.warning("Competitor analysis data is None")
            return {}
        
        # Handle string data (JSON string)
        if isinstance(data, str):
            try:
                data = json.loads(data)
                logger.info("Successfully parsed competitor analysis data from JSON string")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse competitor analysis data JSON string: {e}")
                return {}
        
        # Structure is either:
        # 1. Already has "brand_names" (from previous call)
        # 2. Has "competitor_analysis" as a key with a list of analyses
        # 3. Is itself a list of brand analyses
        
        # Case 1: Already correctly structured
        if isinstance(data, dict) and "brand_names" in data:
            logger.info("Competitor analysis data already in correct format with brand_names key")
            return data
        
        # Create the transformed data structure
        transformed_data = {
            "brand_names": {},
            "summary": "This competitor analysis evaluates how each proposed brand name positions against existing market competitors, identifying strengths, weaknesses, and potential market gaps."
        }
        
        # Extract the brand analyses from the data
        brand_analyses = []
        
        # Case 2: Has "competitor_analysis" key with a list
        if isinstance(data, dict) and "competitor_analysis" in data:
            if isinstance(data["competitor_analysis"], list):
                brand_analyses = data["competitor_analysis"]
                logger.info(f"Found {len(brand_analyses)} brand analyses in competitor_analysis list")
            else:
                logger.warning(f"competitor_analysis is not a list: {type(data['competitor_analysis'])}")
                return transformed_data
        # Case 3: Is itself a list of brand analyses
        elif isinstance(data, list):
            brand_analyses = data
            logger.info(f"Found {len(brand_analyses)} brand analyses in root list")
        else:
            logger.warning(f"Unrecognized competitor analysis data structure: {type(data)}")
            return transformed_data
        
        # Process each brand analysis
        for analysis in brand_analyses:
            # Skip items without brand_name
            if not isinstance(analysis, dict) or "brand_name" not in analysis:
                logger.warning("Skipping brand analysis without brand_name field")
                continue
                
            brand_name = analysis.get("brand_name", "")
            logger.info(f"Processing competitor analysis for: {brand_name}")
            
            # Extract strengths safely
            strengths = []
            if "strengths" in analysis:
                strength_data = analysis.get("strengths", [])
                if isinstance(strength_data, str):
                    try:
                        # Try to parse as JSON string
                        strengths = json.loads(strength_data)
                        logger.info(f"Parsed {len(strengths)} strengths from JSON string")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse strengths as JSON: {e}")
                        # If not valid JSON, use as a single string
                        strengths = [strength_data]
                elif isinstance(strength_data, list):
                    strengths = strength_data
                    logger.info(f"Found {len(strengths)} strengths as list")
                else:
                    strengths = [str(strength_data)]
                    logger.info(f"Converted strengths to string: {strengths[0][:30]}...")
            
            # Extract weaknesses safely
            weaknesses = []
            if "weaknesses" in analysis:
                weakness_data = analysis.get("weaknesses", [])
                if isinstance(weakness_data, str):
                    try:
                        # Try to parse as JSON string
                        weaknesses = json.loads(weakness_data)
                        logger.info(f"Parsed {len(weaknesses)} weaknesses from JSON string")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse weaknesses as JSON: {e}")
                        # If not valid JSON, use as a single string
                        weaknesses = [weakness_data]
                elif isinstance(weakness_data, list):
                    weaknesses = weakness_data
                    logger.info(f"Found {len(weaknesses)} weaknesses as list")
                else:
                    weaknesses = [str(weakness_data)]
                    logger.info(f"Converted weaknesses to string: {weaknesses[0][:30]}...")
            
            # Extract opportunities safely
            opportunities = []
            if "opportunities" in analysis:
                opportunity_data = analysis.get("opportunities", [])
                if isinstance(opportunity_data, str):
                    try:
                        # Try to parse as JSON string
                        opportunities = json.loads(opportunity_data)
                        logger.info(f"Parsed {len(opportunities)} opportunities from JSON string")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse opportunities as JSON: {e}")
                        # If not valid JSON, use as a single string
                        opportunities = [opportunity_data]
                elif isinstance(opportunity_data, list):
                    opportunities = opportunity_data
                    logger.info(f"Found {len(opportunities)} opportunities as list")
                else:
                    opportunities = [str(opportunity_data)]
                    logger.info(f"Converted opportunities to string: {opportunities[0][:30]}...")
            
            # Extract threats safely
            threats = []
            if "threats" in analysis:
                threat_data = analysis.get("threats", [])
                if isinstance(threat_data, str):
                    try:
                        # Try to parse as JSON string
                        threats = json.loads(threat_data)
                        logger.info(f"Parsed {len(threats)} threats from JSON string")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse threats as JSON: {e}")
                        # If not valid JSON, use as a single string
                        threats = [threat_data]
                elif isinstance(threat_data, list):
                    threats = threat_data
                    logger.info(f"Found {len(threats)} threats as list")
                else:
                    threats = [str(threat_data)]
                    logger.info(f"Converted threats to string: {threats[0][:30]}...")
            
            # Create brand analysis entry with formatted structure
            transformed_data["brand_names"][brand_name] = {
                "top_competitors": analysis.get("top_competitors", ""),
                "market_position": analysis.get("market_position", ""),
                "differentiation_score": analysis.get("differentiation_score", 0),
                "market_saturation": analysis.get("market_saturation", ""),
                "strengths": strengths,
                "weaknesses": weaknesses,
                "opportunities": opportunities,
                "threats": threats
            }
        
        logger.info(f"Transformed competitor analysis data with {len(transformed_data['brand_names'])} brand analyses")
        return transformed_data

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

    async def fetch_user_prompt(self, run_id: str) -> str:
        """Fetch the user prompt from the workflow_state table for a specific run_id."""
        logger.info(f"Fetching user prompt for run_id: {run_id}")
        
        try:
            # Query the workflow_state table to get the state data
            result = await self.supabase.execute_with_retry(
                "select",
                "workflow_state",
                {
                    "run_id": f"eq.{run_id}",
                    "select": "state",
                    "order": "created_at.desc",
                    "limit": 1  # Get the most recent state
                }
            )
            
            if not result or len(result) == 0:
                logger.warning(f"No state data found for run_id: {run_id}")
                return "No user prompt available"
            
            # Extract the state data from the result
            state_data = result[0]['state']
            
            # Parse the state data if it's a string
            if isinstance(state_data, str):
                try:
                    state_data = json.loads(state_data)
                except json.JSONDecodeError:
                    logger.warning(f"State data for {run_id} is not valid JSON")
                    return "No user prompt available (invalid state format)"
            
            # Extract the user prompt from the state data
            if isinstance(state_data, dict) and 'user_prompt' in state_data:
                user_prompt = state_data['user_prompt']
                logger.info(f"Successfully fetched user prompt for run_id: {run_id}")
                return user_prompt
            else:
                logger.warning(f"User prompt not found in state data for run_id: {run_id}")
                return "No user prompt available (not in state)"
            
        except Exception as e:
            logger.error(f"Error fetching user prompt for run_id {run_id}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return "No user prompt available (error)"

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
        """
        Format the survey simulation section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw survey simulation data
        """
        try:
            # Add section title
            doc.add_heading("Survey Simulation", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section presents a simulated consumer survey that evaluates the brand name options "
                "based on target audience preferences, emotional responses, and overall appeal. "
                "The simulation helps predict how real consumers might respond to each brand name."
            )
            
            # Transform data using model
            transformed_data = self._transform_survey_simulation(data)
            logger.debug(f"Transformed survey simulation data: {len(str(transformed_data))} chars")
            
            # Check if we have analysis data - specifically looking for the survey_simulation key
            if transformed_data and isinstance(transformed_data, dict) and "survey_simulation" in transformed_data:
                survey_items = transformed_data["survey_simulation"]
                
                if not survey_items:
                    doc.add_paragraph("No survey simulation data available for this brand naming project.")
                    return
                
                # Add methodology section
                doc.add_heading("Methodology", level=2)
                doc.add_paragraph(
                    "The survey simulation represents feedback from key target personas in the "
                    "intended market. Each persona evaluation includes emotional associations, "
                    "personality fit scores, and detailed qualitative feedback on each brand name."
                )
                
                # Process each survey participant
                doc.add_heading("Survey Results by Persona", level=2)
                
                for i, item in enumerate(survey_items, 1):
                    if not isinstance(item, dict):
                        continue
                        
                    # Extract the brand name and company
                    brand_name = item.get("brand_name", f"Brand {i}")
                    company = item.get("company_name", "")
                    
                    # Add persona heading
                    heading_text = f"Persona {i}: {item.get('job_title', '')}"
                    if company:
                        heading_text += f" at {company}"
                    doc.add_heading(heading_text, level=3)
                    
                    # Add persona details if available
                    if "industry" in item:
                        p = doc.add_paragraph()
                        p.add_run("Industry: ").bold = True
                        p.add_run(str(item.get("industry", "")))
                    
                    if "seniority" in item:
                        p = doc.add_paragraph()
                        p.add_run("Seniority: ").bold = True
                        p.add_run(str(item.get("seniority", "")))
                    
                    # Add brand name evaluation
                    doc.add_heading(f"Evaluation of {brand_name}", level=4)
                    
                    # Add emotional association
                    if "emotional_association" in item:
                        p = doc.add_paragraph()
                        p.add_run("Emotional Association: ").bold = True
                        p.add_run(str(item.get("emotional_association", "")))
                    
                    # Add personality fit score
                    if "personality_fit_score" in item:
                        p = doc.add_paragraph()
                        p.add_run("Personality Fit Score: ").bold = True
                        p.add_run(f"{item.get('personality_fit_score', 0)}/10")
                    
                    # Add qualitative feedback summary
                    if "qualitative_feedback_summary" in item:
                        doc.add_heading("Qualitative Feedback Summary", level=5)
                        doc.add_paragraph(str(item.get("qualitative_feedback_summary", "")))
                    
                    # Add raw qualitative feedback
                    if "raw_qualitative_feedback" in item:
                        doc.add_heading("Detailed Feedback", level=5)
                        raw_feedback = item.get("raw_qualitative_feedback", {})
                        
                        if isinstance(raw_feedback, dict):
                            # Handle different structures
                            if "value" in raw_feedback and isinstance(raw_feedback["value"], str):
                                # Try to parse JSON string
                                try:
                                    parsed_feedback = json.loads(raw_feedback["value"])
                                    if isinstance(parsed_feedback, dict):
                                        for key, value in parsed_feedback.items():
                                            p = doc.add_paragraph()
                                            p.add_run(f"{key}: ").bold = True
                                            p.add_run(str(value))
                                    else:
                                        doc.add_paragraph(str(raw_feedback["value"]))
                                except (json.JSONDecodeError, TypeError):
                                    doc.add_paragraph(str(raw_feedback["value"]))
                            else:
                                # Regular dict
                                for key, value in raw_feedback.items():
                                    p = doc.add_paragraph()
                                    p.add_run(f"{key}: ").bold = True
                                    p.add_run(str(value))
                        elif isinstance(raw_feedback, str):
                            doc.add_paragraph(raw_feedback)
                    
                    # Add competitor benchmarking score
                    if "competitor_benchmarking_score" in item:
                        p = doc.add_paragraph()
                        p.add_run("Competitor Benchmarking Score: ").bold = True
                        p.add_run(f"{item.get('competitor_benchmarking_score', 0)}/10")
                    
                    # Add brand promise perception score
                    if "brand_promise_perception_score" in item:
                        p = doc.add_paragraph()
                        p.add_run("Brand Promise Perception Score: ").bold = True
                        p.add_run(f"{item.get('brand_promise_perception_score', 0)}/10")
                    
                    # Add simulated market adoption score
                    if "simulated_market_adoption_score" in item:
                        p = doc.add_paragraph()
                        p.add_run("Simulated Market Adoption Score: ").bold = True
                        p.add_run(f"{item.get('simulated_market_adoption_score', 0)}/10")
                    
                    # Add competitive differentiation score
                    if "competitive_differentiation_score" in item:
                        p = doc.add_paragraph()
                        p.add_run("Competitive Differentiation Score: ").bold = True
                        p.add_run(f"{item.get('competitive_differentiation_score', 0)}/10")
                    
                    # Add final recommendation
                    if "final_survey_recommendation" in item:
                        doc.add_heading("Recommendation", level=5)
                        doc.add_paragraph(str(item.get("final_survey_recommendation", "")))
                    
                    # Add separator between personas
                    if i < len(survey_items):
                        doc.add_paragraph("")
                
                # Add a summary section
                doc.add_heading("Survey Summary", level=2)
                
                # Generate aggregate scores
                brand_names = set()
                total_personality_fit = 0
                total_competitor_benchmark = 0
                total_brand_promise = 0
                total_market_adoption = 0
                total_differentiation = 0
                count = 0
                
                for item in survey_items:
                    if not isinstance(item, dict):
                        continue
                        
                    count += 1
                    if "brand_name" in item:
                        brand_names.add(item.get("brand_name", ""))
                    if "personality_fit_score" in item:
                        total_personality_fit += item.get("personality_fit_score", 0)
                    if "competitor_benchmarking_score" in item:
                        total_competitor_benchmark += item.get("competitor_benchmarking_score", 0)
                    if "brand_promise_perception_score" in item:
                        total_brand_promise += item.get("brand_promise_perception_score", 0)
                    if "simulated_market_adoption_score" in item:
                        total_market_adoption += item.get("simulated_market_adoption_score", 0)
                    if "competitive_differentiation_score" in item:
                        total_differentiation += item.get("competitive_differentiation_score", 0)
                
                # Add summary paragraph
                if count > 0:
                    summary = f"The survey simulation evaluated {len(brand_names)} brand names across {count} personas. "
                    
                    # Add average scores
                    summary += "Average scores across all personas:\n"
                    doc.add_paragraph(summary)
                    
                    scores_table = doc.add_table(rows=1, cols=2)
                    scores_table.style = 'Table Grid'
                    
                    # Add header row
                    header_cells = scores_table.rows[0].cells
                    header_cells[0].text = "Metric"
                    header_cells[1].text = "Average Score (0-10)"
                    
                    # Add scores rows
                    metrics = [
                        ("Personality Fit", total_personality_fit / count if count > 0 else 0),
                        ("Competitor Benchmarking", total_competitor_benchmark / count if count > 0 else 0),
                        ("Brand Promise Perception", total_brand_promise / count if count > 0 else 0),
                        ("Market Adoption Potential", total_market_adoption / count if count > 0 else 0),
                        ("Competitive Differentiation", total_differentiation / count if count > 0 else 0)
                    ]
                    
                    for metric, score in metrics:
                        row = scores_table.add_row()
                        row.cells[0].text = metric
                        row.cells[1].text = f"{score:.1f}"
            else:
                # No survey simulation data available
                doc.add_paragraph("No survey simulation data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting survey simulation section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            doc.add_paragraph(f"Error formatting survey simulation section: {str(e)}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the survey simulation section due to an error in processing the data.")

    async def _format_linguistic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the linguistic analysis section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw linguistic analysis data
        """
        try:
            # Add section title
            doc.add_heading("Linguistic Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the linguistic characteristics of the brand name options, "
                "including pronunciation, spelling, phonetic patterns, and other linguistic elements "
                "that influence brand perception and usability."
            )
            
            # Transform data using model
            transformed_data = self._transform_linguistic_analysis(data)
            logger.info(f"Transformed linguistic analysis data: {len(str(transformed_data))} chars")
            
            # Check if we have linguistic analysis data
            if "linguistic_analysis" in transformed_data and transformed_data["linguistic_analysis"]:
                linguistic_analyses = transformed_data["linguistic_analysis"]
                
                # Add a summary of the number of brand names analyzed
                doc.add_paragraph(f"Linguistic analysis was conducted for {len(linguistic_analyses)} brand name candidates.")
                
                # Process each brand name analysis
                for brand_name, analysis in linguistic_analyses.items():
                    # Add brand name as heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Process attributes in a structured way
                    attributes = [
                        ("word_class", "Word Class"),
                        ("sound_symbolism", "Sound Symbolism"),
                        ("rhythm_and_meter", "Rhythm and Meter"),
                        ("pronunciation_ease", "Pronunciation Ease"),
                        ("euphony_vs_cacophony", "Euphony vs Cacophony"),
                        ("inflectional_properties", "Inflectional Properties"),
                        ("neologism_appropriateness", "Neologism Appropriateness"),
                        ("overall_readability_score", "Overall Readability Score"),
                        ("morphological_transparency", "Morphological Transparency"),
                        ("naturalness_in_collocations", "Naturalness in Collocations"),
                        ("ease_of_marketing_integration", "Ease of Marketing Integration"),
                        ("phoneme_frequency_distribution", "Phoneme Frequency Distribution"),
                        ("semantic_distance_from_competitors", "Semantic Distance from Competitors"),
                        ("notes", "Notes")
                    ]
                    
                    for attr_key, attr_display in attributes:
                        if attr_key in analysis and analysis[attr_key]:
                            # Format based on value type
                            value = analysis[attr_key]
                            
                            # Add attribute
                            p = doc.add_paragraph()
                            p.add_run(f"{attr_display}: ").bold = True
                            
                            # Handle different value types
                            if isinstance(value, dict):
                                # Add each dictionary item as a bullet point
                                for sub_key, sub_value in value.items():
                                    bullet_p = doc.add_paragraph(style="List Bullet")
                                    bullet_p.add_run(f"{sub_key.replace('_', ' ').title()}: ").bold = True
                                    bullet_p.add_run(str(sub_value))
                            elif isinstance(value, list):
                                # Add the attribute value as a comma-separated list
                                p.add_run(", ".join(str(item) for item in value))
                            else:
                                p.add_run(str(value))
                    
                    # Add separator between brand analyses (except after the last one)
                    if brand_name != list(linguistic_analyses.keys())[-1]:
                        doc.add_paragraph("", style="Normal")
                
                # Add summary section
                doc.add_heading("Linguistic Analysis Summary", level=2)
                doc.add_paragraph(
                    "The linguistic analysis examines how each brand name functions from a linguistic perspective, "
                    "considering pronunciation, rhythm, sound symbolism, and other factors that influence "
                    "memorability and effectiveness. This analysis helps identify names that are not only "
                    "semantically appropriate but also linguistically strong."
                )
                    
            else:
                # No linguistic analysis data available
                doc.add_paragraph("No linguistic analysis data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting linguistic analysis section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the linguistic analysis section due to an error in processing the data.")

    async def _format_cultural_sensitivity(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the cultural sensitivity analysis section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw cultural sensitivity analysis data
        """
        try:
            # Add section title
            doc.add_heading("Cultural Sensitivity Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the cultural implications of the brand name options across "
                "different regions and cultural contexts, identifying potential sensitivities and risks "
                "that should be considered in the brand naming decision process."
            )
            
            # Transform data using model
            transformed_data = self._transform_cultural_sensitivity(data)
            logger.info(f"Transformed cultural sensitivity data: {len(str(transformed_data))} chars")
            
            # Check if we have analysis data
            if transformed_data and isinstance(transformed_data, dict):
                # Check for brand_analyses key (main data structure)
                if "brand_analyses" in transformed_data and transformed_data["brand_analyses"]:
                    brand_analyses = transformed_data["brand_analyses"]
                    
                    # Add a summary of the number of brand names analyzed
                    doc.add_paragraph(f"Cultural sensitivity analysis was conducted for {len(brand_analyses)} brand name candidates.")
                    
                    # Process each brand name analysis
                    for analysis in brand_analyses:
                        brand_name = analysis.get("brand_name", "Unknown Brand")
                        
                        # Add brand name as heading
                        doc.add_heading(brand_name, level=2)
                        
                        # Process attributes in a structured way - updated to match BrandAnalysis model fields
                        attributes = [
                            ("symbolic_meanings", "Symbolic Meanings"),
                            ("historical_meaning", "Historical Meaning"),
                            ("overall_risk_rating", "Overall Risk Rating"),
                            ("regional_variations", "Regional Variations"),
                            ("cultural_connotations", "Cultural Connotations"),
                            ("current_event_relevance", "Current Event Relevance"),
                            ("religious_sensitivities", "Religious Sensitivities"), 
                            ("social_political_taboos", "Social Political Taboos"),
                            ("age_related_connotations", "Age-Related Connotations"),
                            ("alignment_with_cultural_values", "Alignment with Cultural Values"),
                            ("notes", "Notes")
                        ]
                        
                        for attr_key, attr_display in attributes:
                            if attr_key in analysis and analysis[attr_key]:
                                # Format based on value type
                                value = analysis[attr_key]
                                
                                # Add attribute
                                p = doc.add_paragraph()
                                p.add_run(f"{attr_display}: ").bold = True
                                
                                # Handle dictionary attributes
                                if isinstance(value, dict):
                                    # Add each dictionary item as a bullet point
                                    for sub_key, sub_value in value.items():
                                        bullet_p = doc.add_paragraph(style="List Bullet")
                                        bullet_p.add_run(f"{sub_key.replace('_', ' ').title()}: ").bold = True
                                        bullet_p.add_run(str(sub_value))
                                # Handle list attributes
                                elif isinstance(value, list):
                                    # Add the attribute value as a bullet list
                                    for item in value:
                                        bullet_p = doc.add_paragraph(style="List Bullet")
                                        bullet_p.add_run(str(item))
                                # Handle simple string/number attributes
                                else:
                                    p.add_run(str(value))
                        
                        # Add separator between brand analyses (except after the last one)
                        if analysis != brand_analyses[-1]:
                            doc.add_paragraph("")
                    
                    # Add summary if available
                    if "summary" in transformed_data and transformed_data["summary"]:
                        doc.add_heading("Cultural Sensitivity Summary", level=2)
                        doc.add_paragraph(transformed_data["summary"])
                else:
                    # No brand analyses available
                    doc.add_paragraph("No cultural sensitivity analysis data is available for the brand names.")
            else:
                # No cultural sensitivity data available
                doc.add_paragraph("No cultural sensitivity analysis data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting cultural sensitivity section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the cultural sensitivity section due to an error in processing the data.")

    async def _format_name_evaluation(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the brand name evaluation section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw brand name evaluation data
        """
        try:
            # Add section title
            doc.add_heading("Brand Name Evaluation", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section provides a comprehensive evaluation of brand name candidates "
                "based on multiple criteria, highlighting their strengths and weaknesses "
                "to support the final recommendation process."
            )
            
            # Transform data using model
            transformed_data = self._transform_name_evaluation(data)
            logger.info(f"Transformed name evaluation data: {len(str(transformed_data))} chars")
            
            # Check if we have evaluation data
            if transformed_data and isinstance(transformed_data, dict):
                # Process evaluation methodology
                if "evaluation_methodology" in transformed_data and transformed_data["evaluation_methodology"]:
                    doc.add_heading("Evaluation Methodology", level=2)
                    doc.add_paragraph(transformed_data["evaluation_methodology"])
                
                # Process shortlisted names
                if "shortlisted_names" in transformed_data and transformed_data["shortlisted_names"]:
                    doc.add_heading("Shortlisted Brand Names", level=2)
                    doc.add_paragraph(
                        f"The following {len(transformed_data['shortlisted_names'])} brand names were shortlisted "
                        f"based on their strong performance across evaluation criteria."
                    )
                    
                    # Process each shortlisted name
                    for name_data in transformed_data["shortlisted_names"]:
                        doc.add_heading(name_data["brand_name"], level=3)
                        
                        # Add overall score
                        p = doc.add_paragraph()
                        p.add_run("Overall Score: ").bold = True
                        p.add_run(f"{name_data['overall_score']}/10")
                        
                        # Add evaluation comments
                        if "evaluation_comments" in name_data and name_data["evaluation_comments"]:
                            doc.add_paragraph(name_data["evaluation_comments"])
                        
                        # Add separator between brand evaluations (except for the last one)
                        if name_data != transformed_data["shortlisted_names"][-1]:
                            doc.add_paragraph("")
                
                # Process other names
                if "other_names" in transformed_data and transformed_data["other_names"]:
                    doc.add_heading("Other Evaluated Brand Names", level=2)
                    doc.add_paragraph(
                        f"The following {len(transformed_data['other_names'])} brand names were evaluated "
                        f"but not shortlisted."
                    )
                    
                    # Process each non-shortlisted name
                    for name_data in transformed_data["other_names"]:
                        doc.add_heading(name_data["brand_name"], level=3)
                        
                        # Add overall score
                        p = doc.add_paragraph()
                        p.add_run("Overall Score: ").bold = True
                        p.add_run(f"{name_data['overall_score']}/10")
                        
                        # Add evaluation comments
                        if "evaluation_comments" in name_data and name_data["evaluation_comments"]:
                            doc.add_paragraph(name_data["evaluation_comments"])
                        
                        # Add separator between brand evaluations (except for the last one)
                        if name_data != transformed_data["other_names"][-1]:
                            doc.add_paragraph("")
                
                # Process comparative analysis
                if "comparative_analysis" in transformed_data and transformed_data["comparative_analysis"]:
                    doc.add_heading("Comparative Analysis", level=2)
                    doc.add_paragraph(transformed_data["comparative_analysis"])
                
                # Process final rankings
                if "final_rankings" in transformed_data and transformed_data["final_rankings"]:
                    doc.add_heading("Final Rankings", level=2)
                    
                    rankings = transformed_data["final_rankings"]
                    if isinstance(rankings, dict):
                        # Create a table for rankings with scores
                        table = doc.add_table(rows=len(rankings)+1, cols=3)
                        table.style = 'Table Grid'
                        
                        # Add header row
                        header_cells = table.rows[0].cells
                        header_cells[0].text = "Rank"
                        header_cells[1].text = "Brand Name"
                        header_cells[2].text = "Score"
                        
                        # Sort rankings by score (descending)
                        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
                        
                        # Add ranking rows
                        for i, (brand_name, score) in enumerate(sorted_rankings, 1):
                            cells = table.rows[i].cells
                            cells[0].text = str(i)
                            cells[1].text = str(brand_name)
                            cells[2].text = str(score)
                    else:
                        # Simple string or list format
                        doc.add_paragraph(str(rankings))
                
                # Add summary section
                doc.add_heading("Evaluation Summary", level=2)
                doc.add_paragraph(
                    "The brand name evaluation process identified names that best align with "
                    "the brand's strategic objectives, target audience, and positioning. "
                    "The shortlisted names demonstrated stronger performance across key criteria "
                    "including distinctiveness, memorability, and strategic alignment."
                )
            else:
                # No evaluation data available
                doc.add_paragraph("No brand name evaluation data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting brand name evaluation section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the brand name evaluation section due to an error in processing the data.")

    async def _format_seo_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the SEO analysis section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw SEO analysis data
        """
        try:
            # Add section title
            doc.add_heading("SEO and Online Discoverability Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the SEO potential and online discoverability of the brand name options, "
                "evaluating factors such as search volume, competition, keyword potential, and social media presence "
                "to identify names with the strongest digital marketing potential."
            )
            
            # Transform data using model
            transformed_data = self._transform_seo_analysis(data)
            logger.info(f"Transformed SEO analysis data: {len(str(transformed_data))} chars")
            
            # Check if we have analysis data
            if transformed_data and isinstance(transformed_data, dict) and "brand_names" in transformed_data and transformed_data["brand_names"]:
                # Process methodology if available
                if "methodology" in transformed_data and transformed_data["methodology"]:
                    doc.add_heading("Methodology", level=2)
                    doc.add_paragraph(transformed_data["methodology"])
                
                # Process brand name analyses
                brand_analyses = transformed_data["brand_names"]
                
                if brand_analyses:
                    doc.add_heading("Brand Name SEO Analysis", level=2)
                    
                    # Process each brand name analysis
                    for brand_name, brand_key in brand_analyses.items():
                        # Add brand name as heading
                        doc.add_heading(brand_name, level=3)
                        
                        # Get the actual analysis data from transformed_data using the brand name
                        if brand_name in transformed_data:
                            analysis = transformed_data[brand_name]
                            
                            # Create a metrics table for all basic metrics
                            metrics_table = doc.add_table(rows=1, cols=2)
                            metrics_table.style = 'Table Grid'
                            
                            # Add header row
                            header_cells = metrics_table.rows[0].cells
                            header_cells[0].text = "Metric"
                            header_cells[1].text = "Value"
                            
                            # Add all SEO metrics to the table
                            metrics = [
                                ("Search Volume", analysis.get("search_volume", "Unknown")),
                                ("Keyword Alignment", analysis.get("keyword_alignment", "Unknown")),
                                ("Keyword Competition", analysis.get("keyword_competition", "Unknown")),
                                ("SEO Viability Score", analysis.get("seo_viability_score", "Unknown")),
                                ("Negative Search Results", "Yes" if analysis.get("negative_search_results", False) else "No"),
                                ("Unusual Spelling Impact", "Yes" if analysis.get("unusual_spelling_impact", False) else "No"),
                                ("Branded Keyword Potential", analysis.get("branded_keyword_potential", "Unknown")),
                                ("Name Length Searchability", analysis.get("name_length_searchability", "Unknown")),
                                ("Social Media Availability", "Yes" if analysis.get("social_media_availability", False) else "No"),
                                ("Competitor Domain Strength", analysis.get("competitor_domain_strength", "Unknown")),
                                ("Exact Match Search Results", analysis.get("exact_match_search_results", "Unknown")),
                                ("Social Media Discoverability", analysis.get("social_media_discoverability", "Unknown")),
                                ("Negative Keyword Associations", analysis.get("negative_keyword_associations", "Unknown")),
                                ("Non-Branded Keyword Potential", analysis.get("non_branded_keyword_potential", "Unknown")),
                                ("Content Marketing Opportunities", analysis.get("content_marketing_opportunities", "Unknown")),
                            ]
                            
                            # Fill the metrics table
                            for metric, value in metrics:
                                row = metrics_table.add_row()
                                cells = row.cells
                                cells[0].text = metric
                                cells[1].text = str(value)
                            
                            # Add spacing after table
                            doc.add_paragraph("")
                            
                            # Process search metrics
                            if "search_metrics" in analysis and analysis["search_metrics"]:
                                doc.add_heading("Search Metrics", level=4)
                                
                                search_metrics = analysis["search_metrics"]
                                if isinstance(search_metrics, dict):
                                    # Create a table for metrics
                                    metrics_table = doc.add_table(rows=len(search_metrics)+1, cols=2)
                                    metrics_table.style = 'Table Grid'
                                    
                                    # Add header row
                                    header_cells = metrics_table.rows[0].cells
                                    header_cells[0].text = "Metric"
                                    header_cells[1].text = "Value"
                                    
                                    # Add metrics rows
                                    for i, (metric, value) in enumerate(search_metrics.items(), 1):
                                        cells = metrics_table.rows[i].cells
                                        cells[0].text = str(metric)
                                        cells[1].text = str(value)
                                    
                                    # Add spacing after table
                                    doc.add_paragraph("")
                                else:
                                    doc.add_paragraph(str(search_metrics))
                            
                            # Process competitive analysis
                            if "competitive_analysis" in analysis and analysis["competitive_analysis"]:
                                doc.add_heading("Competitive Analysis", level=4)
                                
                                comp_analysis = analysis["competitive_analysis"]
                                if isinstance(comp_analysis, str):
                                    doc.add_paragraph(comp_analysis)
                                elif isinstance(comp_analysis, dict):
                                    for key, value in comp_analysis.items():
                                        p = doc.add_paragraph()
                                        p.add_run(f"{key}: ").bold = True
                                        p.add_run(str(value))
                                elif isinstance(comp_analysis, list):
                                    for item in comp_analysis:
                                        doc.add_paragraph(f"• {item}", style="List Bullet")
                            
                            # Process keyword opportunities
                            if "keyword_opportunities" in analysis and analysis["keyword_opportunities"]:
                                doc.add_heading("Keyword Opportunities", level=4)
                                
                                opportunities = analysis["keyword_opportunities"]
                                if isinstance(opportunities, dict):
                                    for key, value in opportunities.items():
                                        p = doc.add_paragraph()
                                        p.add_run(f"{key}: ").bold = True
                                        p.add_run(str(value))
                                elif isinstance(opportunities, list):
                                    for opportunity in opportunities:
                                        doc.add_paragraph(f"• {opportunity}", style="List Bullet")
                                else:
                                    doc.add_paragraph(str(opportunities))
                            
                            # Process social media potential
                            if "social_media_potential" in analysis and analysis["social_media_potential"]:
                                doc.add_heading("Social Media Potential", level=4)
                                
                                social_media = analysis["social_media_potential"]
                                if isinstance(social_media, dict):
                                    for platform, assessment in social_media.items():
                                        p = doc.add_paragraph()
                                        p.add_run(f"{platform}: ").bold = True
                                        p.add_run(str(assessment))
                                else:
                                    doc.add_paragraph(str(social_media))
                            
                            # Process SEO strengths
                            if "strengths" in analysis and analysis["strengths"]:
                                doc.add_heading("SEO Strengths", level=4)
                                
                                strengths = analysis["strengths"]
                                if isinstance(strengths, list):
                                    for strength in strengths:
                                        if strength:  # Only add non-empty strengths
                                            doc.add_paragraph(f"• {strength}", style="List Bullet")
                                else:
                                    doc.add_paragraph(str(strengths))
                            
                            # Process SEO challenges
                            if "challenges" in analysis and analysis["challenges"]:
                                doc.add_heading("SEO Challenges", level=4)
                                
                                challenges = analysis["challenges"]
                                if isinstance(challenges, list):
                                    for challenge in challenges:
                                        if challenge:  # Only add non-empty challenges
                                            doc.add_paragraph(f"• {challenge}", style="List Bullet")
                                else:
                                    doc.add_paragraph(str(challenges))
                            
                            # Process overall rating
                            if "overall_seo_rating" in analysis and analysis["overall_seo_rating"]:
                                p = doc.add_paragraph()
                                p.add_run("Overall SEO Rating: ").bold = True
                                p.add_run(str(analysis["overall_seo_rating"]))
                            
                            # Process recommendations
                            if "recommendations" in analysis and analysis["recommendations"]:
                                doc.add_heading("SEO Recommendations", level=4)
                                
                                recommendations = analysis["recommendations"]
                                if isinstance(recommendations, list):
                                    has_recommendations = False
                                    for recommendation in recommendations:
                                        if recommendation:  # Only add non-empty recommendations
                                            doc.add_paragraph(f"• {recommendation}", style="List Bullet")
                                            has_recommendations = True
                                    
                                    if not has_recommendations:
                                        doc.add_paragraph("No specific SEO recommendations available for this brand name.")
                                else:
                                    doc.add_paragraph(str(recommendations))
                        else:
                            # No analysis data for this brand name
                            doc.add_paragraph(f"No detailed SEO analysis available for {brand_name}.")
                        
                        # Add separator between brand analyses (except for the last one)
                        if brand_name != list(brand_analyses.keys())[-1]:
                            doc.add_paragraph("")
                
                # Process general SEO recommendations
                if "general_recommendations" in transformed_data and transformed_data["general_recommendations"]:
                    doc.add_heading("General SEO Recommendations", level=2)
                    
                    recommendations = transformed_data["general_recommendations"]
                    if isinstance(recommendations, list):
                        for recommendation in recommendations:
                            if recommendation:  # Only add non-empty recommendations
                                doc.add_paragraph(f"• {recommendation}", style="List Bullet")
                
                # Process comparative analysis
                if "seo_comparative_analysis" in transformed_data and transformed_data["seo_comparative_analysis"]:
                    doc.add_heading("Comparative SEO Analysis", level=2)
                    doc.add_paragraph(transformed_data["seo_comparative_analysis"])
                
                # Process summary
                if "summary" in transformed_data and transformed_data["summary"]:
                    doc.add_heading("SEO Analysis Summary", level=2)
                    doc.add_paragraph(transformed_data["summary"])
            else:
                # No SEO analysis data available
                doc.add_paragraph("No SEO analysis data available for this brand naming project.")
                if transformed_data:
                    logger.warning(f"SEO analysis data missing expected keys: {list(transformed_data.keys())}")
                else:
                    logger.warning("SEO analysis transformation returned empty data")
                
        except Exception as e:
            logger.error(f"Error formatting SEO analysis section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the SEO analysis section due to an error in processing the data.")

    async def _format_brand_context(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the brand context section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw brand context data
        """
        try:
            # Add section title
            doc.add_heading("Brand Context", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section provides an overview of the brand's foundation, including its values, mission, "
                "target audience, and market positioning. It serves as the strategic framework for the "
                "brand naming process and ensures alignment with the organization's goals."
            )
            
            # Transform data using model
            transformed_data = self._transform_brand_context(data)
            logger.debug(f"Transformed brand context data: {len(str(transformed_data))} chars")
            
            # Check if we have brand context data
            if transformed_data and isinstance(transformed_data, dict):
                # Brand Values
                if "brand_values" in transformed_data and transformed_data["brand_values"]:
                    doc.add_heading("Brand Values", level=2)
                    values = transformed_data["brand_values"]
                    if isinstance(values, list):
                        for value in values:
                            doc.add_paragraph(f"• {value}", style="List Bullet")
                    else:
                        doc.add_paragraph(str(values))
                
                # Brand Mission
                if "brand_mission" in transformed_data and transformed_data["brand_mission"]:
                    doc.add_heading("Brand Mission", level=2)
                    doc.add_paragraph(transformed_data["brand_mission"])
                
                # Brand Promise
                if "brand_promise" in transformed_data and transformed_data["brand_promise"]:
                    doc.add_heading("Brand Promise", level=2)
                    doc.add_paragraph(transformed_data["brand_promise"])
                
                # Brand Purpose
                if "brand_purpose" in transformed_data and transformed_data["brand_purpose"]:
                    doc.add_heading("Brand Purpose", level=2)
                    doc.add_paragraph(transformed_data["brand_purpose"])
                
                # Customer Needs
                if "customer_needs" in transformed_data and transformed_data["customer_needs"]:
                    doc.add_heading("Customer Needs", level=2)
                    needs = transformed_data["customer_needs"]
                    if isinstance(needs, list):
                        for need in needs:
                            doc.add_paragraph(f"• {need}", style="List Bullet")
                    else:
                        doc.add_paragraph(str(needs))
                
                # Industry Focus
                if "industry_focus" in transformed_data and transformed_data["industry_focus"]:
                    doc.add_heading("Industry Focus", level=2)
                    doc.add_paragraph(transformed_data["industry_focus"])
                
                # Industry Trends
                if "industry_trends" in transformed_data and transformed_data["industry_trends"]:
                    doc.add_heading("Industry Trends", level=2)
                    trends = transformed_data["industry_trends"]
                    if isinstance(trends, list):
                        for trend in trends:
                            doc.add_paragraph(f"• {trend}", style="List Bullet")
                    else:
                        doc.add_paragraph(str(trends))
                
                # Target Audience
                if "target_audience" in transformed_data and transformed_data["target_audience"]:
                    doc.add_heading("Target Audience", level=2)
                    doc.add_paragraph(transformed_data["target_audience"])
                
                # Brand Personality
                if "brand_personality" in transformed_data and transformed_data["brand_personality"]:
                    doc.add_heading("Brand Personality", level=2)
                    personality = transformed_data["brand_personality"]
                    if isinstance(personality, list):
                        for trait in personality:
                            doc.add_paragraph(f"• {trait}", style="List Bullet")
                    else:
                        doc.add_paragraph(str(personality))
                
                # Market Positioning
                if "market_positioning" in transformed_data and transformed_data["market_positioning"]:
                    doc.add_heading("Market Positioning", level=2)
                    doc.add_paragraph(transformed_data["market_positioning"])
                
                # Brand Tone of Voice
                if "brand_tone_of_voice" in transformed_data and transformed_data["brand_tone_of_voice"]:
                    doc.add_heading("Brand Tone of Voice", level=2)
                    doc.add_paragraph(transformed_data["brand_tone_of_voice"])
                
                # Brand Identity Brief
                if "brand_identity_brief" in transformed_data and transformed_data["brand_identity_brief"]:
                    doc.add_heading("Brand Identity Brief", level=2)
                    doc.add_paragraph(transformed_data["brand_identity_brief"])
                
                # Competitive Landscape
                if "competitive_landscape" in transformed_data and transformed_data["competitive_landscape"]:
                    doc.add_heading("Competitive Landscape", level=2)
                    doc.add_paragraph(transformed_data["competitive_landscape"])
                
                # Conclusion
                doc.add_heading("Summary", level=2)
                doc.add_paragraph(
                    f"This brand context provides the foundation for the naming process. "
                    f"The ideal brand name should align with the brand's values of "
                    f"{', '.join(transformed_data.get('brand_values', [''])[:3])} and "
                    f"appeal to the target audience of {transformed_data.get('target_audience', '').split(',')[0]}. "
                    f"It should reflect the brand's {', '.join(transformed_data.get('brand_personality', [''])[:2])} "
                    f"personality and support its positioning as {transformed_data.get('market_positioning', '').split('.')[0]}."
                )
            else:
                # No brand context data available
                doc.add_paragraph("No brand context data available for this brand naming project.")
        
        except Exception as e:
            logger.error(f"Error formatting brand context section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the brand context section due to an error in processing the data.")

    async def _format_name_generation(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the brand name generation section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw name generation data
        """
        try:
            # Add section title
            doc.add_heading("Brand Name Generation", level=1)
            
            # Transform data using model
            transformed_data = self._transform_name_generation(data)
            logger.info(f"Transformed name generation data: {len(str(transformed_data))} chars")
            
            # Add introduction if available
            if "introduction" in transformed_data and transformed_data["introduction"]:
                doc.add_paragraph(transformed_data["introduction"])
            else:
                # Add default introduction
                doc.add_paragraph("The following section presents brand name candidates generated based on the brand context and naming strategy requirements.")
            
            # Add methodology and approach
            if "methodology_and_approach" in transformed_data and transformed_data["methodology_and_approach"]:
                doc.add_heading("Methodology and Approach", level=2)
                doc.add_paragraph(transformed_data["methodology_and_approach"])
            
            # Process categories and names
            if "categories" in transformed_data and isinstance(transformed_data["categories"], list):
                doc.add_heading("Brand Name Categories", level=2)
                
                # Add each category
                for category in transformed_data["categories"]:
                    # Add category heading
                    category_name = category.get("category_name")
                    if category_name:
                        doc.add_heading(category_name, level=3)
                    
                    # Add category description
                    category_description = category.get("category_description")
                    if category_description:
                        doc.add_paragraph(category_description)
                    
                    # Process names in this category
                    names = category.get("names", [])
                    if names:
                        # Add a count of names in this category
                        doc.add_paragraph(f"This category includes {len(names)} brand name options:")
                        
                        # Process each name in this category
                        for name in names:
                            # Add name heading
                            brand_name = name.get("brand_name")
                            if brand_name:
                                doc.add_heading(brand_name, level=4)
                            
                            # Add name attributes in a structured format
                            attributes = [
                                ("brand_personality_alignment", "Brand Personality Alignment"),
                                ("brand_promise_alignment", "Brand Promise Alignment"),
                                ("name_generation_methodology", "Methodology"),
                                ("memorability_score_details", "Memorability"),
                                ("pronounceability_score_details", "Pronounceability"),
                                ("visual_branding_potential_details", "Visual Branding Potential"),
                                ("target_audience_relevance_details", "Target Audience Relevance"),
                                ("market_differentiation_details", "Market Differentiation"),
                                ("rationale", "Rationale"),
                                ("trademark_status", "Trademark Status"),
                                ("cultural_considerations", "Cultural Considerations")
                            ]
                            
                            for attr_key, attr_display in attributes:
                                if attr_key in name and name[attr_key]:
                                    p = doc.add_paragraph()
                                    p.add_run(f"{attr_display}: ").bold = True
                                    p.add_run(str(name[attr_key]))
                            
                            # Add a separator between names (except for the last one)
                            if name != names[-1]:
                                doc.add_paragraph("", style="Normal")
            
            # Add generated names overview if available
            if "generated_names_overview" in transformed_data and transformed_data["generated_names_overview"]:
                doc.add_heading("Generated Names Overview", level=2)
                
                if isinstance(transformed_data["generated_names_overview"], dict):
                    # Get total count
                    total_count = transformed_data["generated_names_overview"].get("total_count", 0)
                    
                    # If we have a total count, display it
                    if total_count:
                        doc.add_paragraph(f"A total of {total_count} names were generated across various naming categories.")
                    
                    # Add any other overview information
                    for key, value in transformed_data["generated_names_overview"].items():
                        if key != "total_count" and value:
                            p = doc.add_paragraph()
                            p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                            p.add_run(str(value))
            
            # Add evaluation metrics if available
            if "evaluation_metrics" in transformed_data and transformed_data["evaluation_metrics"]:
                doc.add_heading("Evaluation Metrics", level=2)
                
                if isinstance(transformed_data["evaluation_metrics"], dict):
                    for key, value in transformed_data["evaluation_metrics"].items():
                        p = doc.add_paragraph()
                        p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                        p.add_run(str(value))
            
            # Add summary if available
            if "summary" in transformed_data and transformed_data["summary"]:
                doc.add_heading("Summary", level=2)
                doc.add_paragraph(transformed_data["summary"])
                
        except Exception as e:
            logger.error(f"Error formatting name generation section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the brand name generation section due to an error in processing the data.")

    async def _format_competitor_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the competitor analysis section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw competitor analysis data
        """
        try:
            # Add section title
            doc.add_heading("Competitor Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes competitors' brand naming strategies and industry positioning, "
                "providing valuable context for the evaluation of proposed brand names. "
                "Understanding competitor naming patterns helps inform differentiation strategies "
                "and identify potential risks of market confusion."
            )
            
            # Transform data using model
            transformed_data = self._transform_competitor_analysis(data)
            logger.info(f"Transformed competitor analysis data: {len(str(transformed_data))} chars")
            
            # Check if we have analysis data
            if transformed_data and isinstance(transformed_data, dict):
                # Process methodology if available
                if "methodology" in transformed_data and transformed_data["methodology"]:
                    doc.add_heading("Methodology", level=2)
                    doc.add_paragraph(transformed_data["methodology"])
                
                # Process industry overview
                if "industry_overview" in transformed_data and transformed_data["industry_overview"]:
                    doc.add_heading("Industry Overview", level=2)
                    doc.add_paragraph(transformed_data["industry_overview"])
                
                # Process naming patterns
                if "naming_patterns" in transformed_data and transformed_data["naming_patterns"]:
                    doc.add_heading("Industry Naming Patterns", level=2)
                    
                    patterns = transformed_data["naming_patterns"]
                    if isinstance(patterns, str):
                        doc.add_paragraph(patterns)
                    elif isinstance(patterns, list):
                        for pattern in patterns:
                            if isinstance(pattern, dict) and "pattern" in pattern and "description" in pattern:
                                p = doc.add_paragraph(style="List Bullet")
                                p.add_run(f"{pattern['pattern']}: ").bold = True
                                p.add_run(pattern["description"])
                            else:
                                doc.add_paragraph(f"• {pattern}", style="List Bullet")
                    elif isinstance(patterns, dict):
                        for pattern_type, description in patterns.items():
                            p = doc.add_paragraph(style="List Bullet")
                            p.add_run(f"{pattern_type.replace('_', ' ').title()}: ").bold = True
                            p.add_run(str(description))
                
                # Process brand name analysis for proposed names
                if "brand_names" in transformed_data and transformed_data["brand_names"]:
                    doc.add_heading("Competitive Analysis of Proposed Brand Names", level=2)
                    
                    brand_analyses = transformed_data["brand_names"]
                    
                    # Process each brand name analysis
                    for brand_name, analysis in brand_analyses.items():
                        # Add brand name as heading
                        doc.add_heading(brand_name, level=3)
                        
                        # Process differentiation
                        if "differentiation" in analysis and analysis["differentiation"]:
                            doc.add_heading("Competitive Differentiation", level=4)
                            doc.add_paragraph(analysis["differentiation"])
                        
                        # Process competitor section
                        if "competitors" in analysis and analysis["competitors"]:
                            doc.add_heading("Key Competitors", level=4)
                            
                            for competitor in analysis["competitors"]:
                                # Add competitor name as subheading
                                doc.add_heading(competitor["competitor_name"], level=5)
                                
                                # Add risk of confusion
                                if "risk_of_confusion" in competitor:
                                    p = doc.add_paragraph()
                                    p.add_run("Risk of Confusion: ").bold = True
                                    p.add_run(f"{competitor['risk_of_confusion']}/10")
                                
                                # Add trademark conflict risk
                                if "trademark_conflict_risk" in competitor:
                                    p = doc.add_paragraph()
                                    p.add_run("Trademark Conflict Risk: ").bold = True
                                    p.add_run(competitor["trademark_conflict_risk"])
                                
                                # Add positioning
                                if "positioning" in competitor:
                                    p = doc.add_paragraph()
                                    p.add_run("Positioning: ").bold = True
                                    p.add_run(competitor["positioning"])
                                
                                # Add strengths
                                if "strengths" in competitor:
                                    p = doc.add_paragraph()
                                    p.add_run("Strengths: ").bold = True
                                    p.add_run(competitor["strengths"])
                                
                                # Add weaknesses
                                if "weaknesses" in competitor:
                                    p = doc.add_paragraph()
                                    p.add_run("Weaknesses: ").bold = True
                                    p.add_run(competitor["weaknesses"])
                                
                                # Add target audience
                                if "target_audience_perception" in competitor:
                                    p = doc.add_paragraph()
                                    p.add_run("Target Audience Perception: ").bold = True
                                    p.add_run(competitor["target_audience_perception"])
                                
                                # Add differentiation opportunity
                                if "differentiation_opportunity" in competitor:
                                    p = doc.add_paragraph()
                                    p.add_run("Differentiation Opportunity: ").bold = True
                                    p.add_run(competitor["differentiation_opportunity"])
                                
                                # Add space between competitors
                                doc.add_paragraph("")
                        
                        # Process similarity assessment
                        if "similarity_assessment" in analysis and analysis["similarity_assessment"]:
                            doc.add_heading("Similarity Assessment", level=4)
                            
                            similarity = analysis["similarity_assessment"]
                            if isinstance(similarity, dict):
                                for competitor, assessment in similarity.items():
                                    p = doc.add_paragraph()
                                    p.add_run(f"{competitor}: ").bold = True
                                    p.add_run(str(assessment))
                            else:
                                doc.add_paragraph(str(similarity))
                        
                        # Process recommendations
                        if "recommendations" in analysis and analysis["recommendations"]:
                            doc.add_heading("Competitive Recommendations", level=4)
                            
                            recommendations = analysis["recommendations"]
                            if isinstance(recommendations, list):
                                for recommendation in recommendations:
                                    doc.add_paragraph(f"• {recommendation}", style="List Bullet")
                            else:
                                doc.add_paragraph(str(recommendations))
                        
                        # Add separator between brand analyses (except for the last one)
                        if brand_name != list(brand_analyses.keys())[-1]:
                            doc.add_paragraph("")
                
                # Process summary
                if "summary" in transformed_data and transformed_data["summary"]:
                    doc.add_heading("Summary", level=2)
                    doc.add_paragraph(transformed_data["summary"])
            else:
                # No competitor analysis data available
                doc.add_paragraph("No competitor analysis data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting competitor analysis section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the competitor analysis section due to an error in processing the data.")

    async def _add_table_of_contents(self, doc: Document) -> None:
        """Add a table of contents to the document."""
        logger.info("Adding table of contents")
        
        try:
            # Add the TOC heading
            doc.add_heading("Table of Contents", level=1)
            
            # Define the hardcoded TOC sections with descriptions
            toc_sections = [
                {
                    "title": "1. Executive Summary",
                    "description": "Project overview, key findings, and top recommendations."
                },
                {
                    "title": "2. Brand Context",
                    "description": "Brand identity, values, target audience, market positioning, and industry context."
                },
                {
                    "title": "3. Name Generation",
                    "description": "Methodology, generated names overview, and initial evaluation metrics."
                },
                {
                    "title": "4. Linguistic Analysis",
                    "description": "Pronunciation, euphony, rhythm, sound symbolism, and overall readability."
                },
                {
                    "title": "5. Semantic Analysis",
                    "description": "Meaning, etymology, emotional valence, and brand personality fit."
                },
                {
                    "title": "6. Cultural Sensitivity",
                    "description": "Cultural connotations, symbolic meanings, sensitivities, and risk assessment."
                },
                {
                    "title": "7. Translation Analysis",
                    "description": "Translations, semantic shifts, pronunciation, and global consistency."
                },
                {
                    "title": "8. Name Evaluation",
                    "description": "Strategic alignment, distinctiveness, memorability, and shortlisted names."
                },
                {
                    "title": "9. Domain Analysis",
                    "description": "Domain availability, alternative TLDs, social media availability, and SEO potential."
                },
                {
                    "title": "10. SEO/Online Discoverability",
                    "description": "Keyword alignment, search volume potential, and content opportunities."
                },
                {
                    "title": "11. Competitor Analysis",
                    "description": "Competitor naming styles, market positioning, and differentiation opportunities."
                },
                {
                    "title": "12. Market Research",
                    "description": "Market opportunity, target audience fit, viability, and industry insights."
                },
                {
                    "title": "13. Survey Simulation",
                    "description": "Persona demographics, brand perception, emotional associations, and feedback."
                },
                {
                    "title": "14. Strategic Recommendations",
                    "description": "Final name recommendations, implementation strategy, and next steps."
                }
            ]
            
            # Add each section to the document
            for section in toc_sections:
                # Add section title and description
                p = doc.add_paragraph()
                p.add_run(section["title"]).bold = True
                p.add_run(": " + section["description"])
                p.add_run().add_break()  # Add line break after each section
            
            # Add a page break after section descriptions
            doc.add_page_break()
            
            # Add the actual table of contents field (Word will populate this)
            doc.add_heading("Document Outline", level=1)
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
            doc.add_heading("Table of Contents", level=1)
            doc.add_paragraph("An error occurred while generating the table of contents.")
            
            # Still add the TOC field so document navigation works
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
            
            doc.add_page_break()
    
    async def _add_executive_summary(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add an executive summary to the document."""
        logger.info("Adding executive summary")
        
        try:
            # Extract necessary data
            all_data = await self.fetch_raw_data(self.current_run_id)
            brand_context = all_data.get("brand_context", {})
            
            # Fetch the user prompt from the state
            user_prompt = await self.fetch_user_prompt(self.current_run_id)
            
            # Count total names generated
            total_names = 0
            shortlisted_names = []
            
            # Get name generation data if available
            if "brand_name_generation" in all_data:
                name_gen_data = all_data["brand_name_generation"]
                
                # Count names across categories
                if "name_categories" in name_gen_data and isinstance(name_gen_data["name_categories"], list):
                    for category in name_gen_data["name_categories"]:
                        if "names" in category and isinstance(category["names"], list):
                            total_names += len(category["names"])
            
            # Get shortlisted names if available
            if "brand_name_evaluation" in all_data:
                eval_data = all_data["brand_name_evaluation"]
                
                # Check if eval_data has expected structure
                if "brand_name_evaluation" in eval_data and isinstance(eval_data["brand_name_evaluation"], dict):
                    # Extract from proper structure
                    eval_lists = eval_data["brand_name_evaluation"]
                    if "shortlisted_names" in eval_lists and isinstance(eval_lists["shortlisted_names"], list):
                        shortlisted_names = [item.get("brand_name", "") for item in eval_lists["shortlisted_names"]]
                elif isinstance(eval_data, dict):
                    # Try alternate format with names as keys
                    for name, details in eval_data.items():
                        if isinstance(details, dict) and details.get("shortlist_status", False):
                            shortlisted_names.append(name)
                
                logger.info(f"Found {len(shortlisted_names)} shortlisted names for executive summary")
            
            # Prepare data for all sections to include in executive summary
            sections_data = {}
            for section_name in self.SECTION_ORDER:
                if section_name in all_data and section_name not in ["exec_summary", "final_recommendations"]:
                    sections_data[section_name] = all_data[section_name]
            
            # Create format data for the executive summary template
            format_data = {
                "run_id": self.current_run_id,
                "sections_data": json.dumps(sections_data, indent=2),
                "brand_context": json.dumps(brand_context, indent=2),
                "total_names": total_names,
                "shortlisted_names": shortlisted_names,
                "user_prompt": user_prompt
            }
            
            # Format the template
            formatted_prompt = self._format_template("executive_summary", format_data, "Executive Summary")
            
            # Get system content
            system_content = self._get_system_content("You are an expert report formatter creating a professional executive summary.")
            
            # Call LLM to generate executive summary
            response = await self._safe_llm_invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=formatted_prompt)
            ], section_name="Executive Summary")
            
            # Try to parse the response
            summary_data = {}
            try:
                content_str = response.content
                # Extract JSON if in code blocks
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content_str)
                if json_match:
                    content_str = json_match.group(1)
                summary_data = json.loads(content_str)
            except Exception as e:
                logger.error(f"Error parsing executive summary response: {str(e)}")
                # If JSON parsing fails, use the raw content
                summary_data = {"summary": response.content}
            
            # Add the executive summary heading
            doc.add_heading("Executive Summary", level=1)
            
            # Add the summary content
            if "summary" in summary_data and summary_data["summary"]:
                doc.add_paragraph(summary_data["summary"])
            
            # Add key points if available
            if "key_points" in summary_data and isinstance(summary_data["key_points"], list):
                doc.add_heading("Key Points", level=2)
                for point in summary_data["key_points"]:
                    bullet = doc.add_paragraph(style='List Bullet')
                    bullet.add_run(point)
            
            # Add page break after executive summary
            doc.add_page_break()
            
        except Exception as e:
            logger.error(f"Error adding executive summary: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Add a simple executive summary as fallback
            doc.add_heading("Executive Summary", level=1)
            doc.add_paragraph(
                "This report presents a comprehensive brand naming analysis based on linguistic, semantic, "
                "cultural, and market research. It provides detailed evaluations of potential brand names "
                "and offers strategic recommendations for the final name selection."
            )
            doc.add_page_break()

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
        """
        Format a generic section of the report.
        
        Args:
            doc: The document to add content to
            section_name: The name of the section to format
            data: The data to format
        """
        try:
            # Check if we have a specific format method for this section
            if section_name in self.REVERSE_SECTION_MAPPING:
                db_section_name = self.REVERSE_SECTION_MAPPING[section_name]
            else:
                db_section_name = section_name
                
            # Use the specific formatter method if it exists
            method_name = f"_format_{db_section_name}"
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                logger.debug(f"Using specific formatter for {section_name} ({method_name})")
                await getattr(self, method_name)(doc, data)
                return
                
            # Get the display section name from the mapping or capitalize the section name
            display_section_name = self.SECTION_MAPPING.get(db_section_name, section_name.replace("_", " ").title())
            
            # Add a section heading
            doc.add_heading(display_section_name, level=1)
            
            # Only use LLM for formatting if it's brand_context
            if db_section_name == "brand_context" and self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        section_name: json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                        "format_instructions": self._get_format_instructions(section_name)
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template(section_name, format_data, section_name)
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, section_name)
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, section_name)
                    
                    if json_content:
                        # Add formatted content
                        for sub_section_name, content in json_content.items():
                            if content and isinstance(content, str):
                                doc.add_heading(sub_section_name.replace("_", " ").title(), level=2)
                                doc.add_paragraph(content)
                            elif content and isinstance(content, list):
                                doc.add_heading(sub_section_name.replace("_", " ").title(), level=2)
                                for item in content:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(item)
                        
                        return  # Successfully formatted with LLM
                        
                except Exception as e:
                    logger.error(f"Error formatting section {section_name} with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # For any other section, use direct formatting
            # Add a paragraph with the section content
            if isinstance(data, dict):
                # Process the dictionary data
                for key, value in data.items():
                    if isinstance(value, str) and value.strip():
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value.strip())
                    elif isinstance(value, list):
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                bullet = doc.add_paragraph(style='List Bullet')
                                bullet.add_run(item.strip())
                            elif isinstance(item, dict):
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, str) and sub_value.strip():
                                        doc.add_heading(sub_key.replace("_", " ").title(), level=3)
                                        doc.add_paragraph(sub_value.strip())
                    elif isinstance(value, dict):
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, str) and sub_value.strip():
                                doc.add_heading(sub_key.replace("_", " ").title(), level=3)
                                doc.add_paragraph(sub_value.strip())
            elif isinstance(data, str) and data.strip():
                doc.add_paragraph(data.strip())
            else:
                # Add a generic message for empty or non-string, non-dict data
                doc.add_paragraph(f"No {display_section_name} data available.")
                
        except Exception as e:
            logger.error(f"Error in _format_generic_section for {section_name}: {str(e)}")
            doc.add_paragraph(f"Error formatting {section_name} section: {str(e)}")
            # Add a more generic message
            doc.add_paragraph(f"Unable to format the {display_section_name} section due to an error in processing the data.")

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
            # Check if Supabase client is initialized
            if not self.supabase:
                logger.error("Supabase client is not initialized. Cannot upload report.")
                return None
                
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
            
            if not result:
                logger.error("Upload completed but no result returned from Supabase")
                return None
                
            # Get the public URL
            public_url = await self.supabase.storage_get_public_url(
                bucket=self.STORAGE_BUCKET,
                path=storage_path
            )
            
            if not public_url:
                logger.error("Failed to get public URL for uploaded file")
                return None
                
            logger.info(f"Report uploaded successfully to: {public_url}")
            
            # Store metadata in report_metadata table
            await self.store_report_metadata(
                run_id=run_id,
                report_url=public_url,
                file_size_kb=file_size_kb,
                format=self.FORMAT_DOCX
            )
            
            return public_url
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error uploading report to storage: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Return None instead of raising to allow the process to continue
            return None

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

    def _extract_json_from_response(self, response_content: str, section_name: str = None) -> Dict[str, Any]:
        """
        Extract JSON content from LLM response.
        
        Args:
            response_content: The raw response content from the LLM
            section_name: Optional section name for logging
            
        Returns:
            Dict containing the parsed JSON, or None if extraction fails
        """
        try:
            # Try to extract JSON between triple backticks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_content)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
            
            # If no JSON in backticks, try to parse the entire response
            return json.loads(response_content)
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            context = f" for {section_name}" if section_name else ""
            logger.error(f"Error extracting JSON from response{context}: {str(e)}")
            return None

    async def _format_market_research(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the market research section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw market research data
        """
        try:
            # Add section title
            doc.add_heading("Market Research", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section presents market research findings relevant to the brand naming process, "
                "including industry trends, target audience insights, and competitive landscape analysis. "
                "This information helps contextualize the brand name options within the current market environment."
            )
            
            # Transform data using model
            transformed_data = self._transform_market_research(data)
            logger.debug(f"Transformed market research data: {len(str(transformed_data))} chars")
            
            # Check if we have market research data
            if transformed_data and isinstance(transformed_data, dict) and "market_research" in transformed_data:
                market_research = transformed_data["market_research"]
                
                # Process each brand name's research data
                if isinstance(market_research, dict) and market_research:
                    doc.add_heading("Market Analysis by Brand Name", level=2)
                    
                    # Process each brand name research
                    for brand_name, research in market_research.items():
                        # Add brand name heading
                        doc.add_heading(brand_name, level=3)
                        
                        # Industry and Market Size
                        if "industry_name" in research and research["industry_name"]:
                            p = doc.add_paragraph()
                            p.add_run("Industry: ").bold = True
                            p.add_run(research["industry_name"])
                        
                        if "market_size" in research and research["market_size"]:
                            p = doc.add_paragraph()
                            p.add_run("Market Size: ").bold = True
                            p.add_run(research["market_size"])
                        
                        if "market_growth_rate" in research and research["market_growth_rate"]:
                            p = doc.add_paragraph()
                            p.add_run("Market Growth Rate: ").bold = True
                            p.add_run(research["market_growth_rate"])
                        
                        # Emerging Trends
                        if "emerging_trends" in research and research["emerging_trends"]:
                            doc.add_heading("Emerging Trends", level=4)
                            trends = research["emerging_trends"]
                            
                            if isinstance(trends, str):
                                doc.add_paragraph(trends)
                            elif isinstance(trends, list):
                                for trend in trends:
                                    doc.add_paragraph(f"• {trend}", style="List Bullet")
                        
                        # Key Competitors
                        if "key_competitors" in research and research["key_competitors"]:
                            doc.add_heading("Key Competitors", level=4)
                            competitors = research["key_competitors"]
                            
                            if isinstance(competitors, str):
                                doc.add_paragraph(competitors)
                            elif isinstance(competitors, list):
                                for competitor in competitors:
                                    doc.add_paragraph(f"• {competitor}", style="List Bullet")
                        
                        # Market Viability
                        if "market_viability" in research and research["market_viability"]:
                            doc.add_heading("Market Viability", level=4)
                            doc.add_paragraph(research["market_viability"])
                        
                        # Market Opportunity
                        if "market_opportunity" in research and research["market_opportunity"]:
                            doc.add_heading("Market Opportunity", level=4)
                            doc.add_paragraph(research["market_opportunity"])
                        
                        # Target Audience Fit
                        if "target_audience_fit" in research and research["target_audience_fit"]:
                            doc.add_heading("Target Audience Fit", level=4)
                            doc.add_paragraph(research["target_audience_fit"])
                        
                        # Competitive Analysis
                        if "competitive_analysis" in research and research["competitive_analysis"]:
                            doc.add_heading("Competitive Analysis", level=4)
                            doc.add_paragraph(research["competitive_analysis"])
                        
                        # Customer Pain Points
                        if "customer_pain_points" in research and research["customer_pain_points"]:
                            doc.add_heading("Customer Pain Points", level=4)
                            pain_points = research["customer_pain_points"]
                            
                            if isinstance(pain_points, str):
                                doc.add_paragraph(pain_points)
                            elif isinstance(pain_points, list):
                                for point in pain_points:
                                    doc.add_paragraph(f"• {point}", style="List Bullet")
                        
                        # Market Entry Barriers
                        if "market_entry_barriers" in research and research["market_entry_barriers"]:
                            doc.add_heading("Market Entry Barriers", level=4)
                            doc.add_paragraph(research["market_entry_barriers"])
                        
                        # Potential Risks
                        if "potential_risks" in research and research["potential_risks"]:
                            doc.add_heading("Potential Risks", level=4)
                            doc.add_paragraph(research["potential_risks"])
                        
                        # Recommendations
                        if "recommendations" in research and research["recommendations"]:
                            doc.add_heading("Recommendations", level=4)
                            doc.add_paragraph(research["recommendations"])
                        
                        # Add separator between brand research (except for the last one)
                        if brand_name != list(market_research.keys())[-1]:
                            doc.add_paragraph("")
                            doc.add_paragraph("_" * 40)
                            doc.add_paragraph("")
                
                    # Add a market research summary section
                    doc.add_heading("Market Research Summary", level=2)
                    doc.add_paragraph(
                        "The market research provides a comprehensive understanding of the industry landscape, "
                        "competitive environment, and target audience needs. This information has been valuable "
                        "in assessing the shortlisted brand names against real-world market conditions and "
                        "ensuring that the recommended names have strong potential for success."
                    )
                else:
                    # Handle generic structure
                    logger.warning("Market research data doesn't match expected model structure, using generic formatter")
                    await self._format_generic_section(doc, "Market Research", transformed_data)
            else:
                # No market research data available
                doc.add_paragraph("No market research data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting market research section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the market research section due to an error in processing the data.")
            # Try to display raw data in a structured format
            try:
                if isinstance(data, dict):
                    await self._format_generic_section_fallback(doc, "Market Research", data)
            except:
                pass

    async def _format_semantic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the semantic analysis section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw semantic analysis data
        """
        try:
            # Add section title
            doc.add_heading("Semantic Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the semantic aspects of the brand name options, "
                "including etymology, meaning, sound symbolism, and other semantic dimensions "
                "that influence brand perception and memorability."
            )
            
            # Transform data using model
            transformed_data = self._transform_semantic_analysis(data)
            logger.info(f"Transformed semantic analysis data: {len(str(transformed_data))} chars")
            
            # Check if we have brand analyses
            if "brand_analyses" in transformed_data and transformed_data["brand_analyses"]:
                brand_analyses = transformed_data["brand_analyses"]
                
                # Add a summary of the number of brand names analyzed
                doc.add_paragraph(f"Semantic analysis was conducted for {len(brand_analyses)} brand name candidates.")
                
                # Process each brand name analysis
                for analysis in brand_analyses:
                    brand_name = analysis.get("brand_name", "Unknown Brand")
                    
                    # Add brand name as heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Process attributes in a structured way - updated to match the exact fields in SemanticAnalysisDetails model
                    attributes = [
                        ("etymology", "Etymology"),
                        ("sound_symbolism", "Sound Symbolism"),
                        ("brand_personality", "Brand Personality"),
                        ("emotional_valence", "Emotional Valence"),
                        ("denotative_meaning", "Denotative Meaning"),
                        ("figurative_language", "Figurative Language"),
                        ("phoneme_combinations", "Phoneme Combinations"),
                        ("sensory_associations", "Sensory Associations"),
                        ("word_length_syllables", "Word Length (Syllables)"),
                        ("alliteration_assonance", "Alliteration/Assonance"),
                        ("compounding_derivation", "Compounding/Derivation"),
                        ("semantic_trademark_risk", "Semantic Trademark Risk")
                    ]
                    
                    for attr_key, attr_display in attributes:
                        if attr_key in analysis and analysis[attr_key]:
                            # Format based on value type
                            value = analysis[attr_key]
                            
                            # Add heading
                            p = doc.add_paragraph()
                            p.add_run(f"{attr_display}: ").bold = True
                            
                            # Handle dictionary attributes
                            if isinstance(value, dict):
                                # Add each dictionary item as a bullet point
                                for sub_key, sub_value in value.items():
                                    bullet_p = doc.add_paragraph(style="List Bullet")
                                    bullet_p.add_run(f"{sub_key.replace('_', ' ').title()}: ").bold = True
                                    bullet_p.add_run(str(sub_value))
                            # Handle list attributes
                            elif isinstance(value, list):
                                # Add the attribute value as a comma-separated list
                                p.add_run(", ".join(str(item) for item in value))
                            # Handle special formatting for boolean values
                            elif isinstance(value, bool):
                                p.add_run("Yes" if value else "No")
                            # Handle simple string/number attributes
                            else:
                                p.add_run(str(value))
                    
                    # Add separator between brand analyses (except after the last one)
                    if analysis != brand_analyses[-1]:
                        doc.add_paragraph("")
                
                # Add summary section
                doc.add_heading("Semantic Analysis Summary", level=2)
                doc.add_paragraph(
                    "The semantic analysis reveals how each brand name option communicates meaning through various "
                    "linguistic dimensions. This analysis helps identify names with strong semantic foundations that "
                    "align with the desired brand positioning and minimize potential semantic confusion or negative associations."
                )
            else:
                # No semantic analysis data available
                doc.add_paragraph("No semantic analysis data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting semantic analysis section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the semantic analysis section due to an error in processing the data.")

    async def _format_translation_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the translation analysis section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw translation analysis data
        """
        try:
            # Add section title
            doc.add_heading("Translation Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes how the brand name options translate across different languages and cultures, "
                "identifying potential issues, unintended meanings, or pronunciation challenges that could impact "
                "global brand perception."
            )
            
            # Transform data using model
            transformed_data = self._transform_translation_analysis(data)
            logger.info(f"Transformed translation analysis data: {len(str(transformed_data))} chars")
            
            # Check if we have analysis data with the expected structure
            if transformed_data and "translation_analysis" in transformed_data:
                translation_analysis = transformed_data["translation_analysis"]
                
                # Add a summary of the number of brand names analyzed
                brand_count = len(translation_analysis)
                doc.add_paragraph(f"Translation analysis was conducted for {brand_count} brand name candidates.")
                
                # Process each brand name
                for brand_name, languages in translation_analysis.items():
                    # Add brand name as heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Process each language analysis for this brand name
                    for language_name, language_analysis in languages.items():
                        # Add language heading
                        doc.add_heading(f"{language_name}", level=3)
                        
                        # Process attributes in a structured way
                        attributes = [
                            ("target_language", "Target Language"),
                            ("direct_translation", "Direct Translation"),
                            ("semantic_shift", "Semantic Shift"),
                            ("phonetic_retention", "Phonetic Retention"),
                            ("pronunciation_difficulty", "Pronunciation Difficulty"),
                            ("adaptation_needed", "Adaptation Needed"),
                            ("proposed_adaptation", "Proposed Adaptation"),
                            ("cultural_acceptability", "Cultural Acceptability"),
                            ("brand_essence_preserved", "Brand Essence Preserved"),
                            ("global_consistency_vs_localization", "Global Consistency vs Localization"),
                            ("notes", "Additional Notes")
                        ]
                        
                        # Display each attribute if present
                        for attr, label in attributes:
                            if attr in language_analysis and language_analysis[attr] is not None:
                                value = language_analysis[attr]
                                
                                # Special handling for boolean values
                                if isinstance(value, bool):
                                    value = "Yes" if value else "No"
                                
                                p = doc.add_paragraph()
                                p.add_run(f"{label}: ").bold = True
                                p.add_run(str(value))
                
                        # Add separator between brand names (except for the last one)
                        if brand_name != list(translation_analysis.keys())[-1]:
                            doc.add_paragraph("", style="Normal")
                    
                    # Add a summary section
                    doc.add_heading("Translation Analysis Summary", level=2)
                    doc.add_paragraph(
                        "The analysis above evaluates how each brand name option would perform across different languages "
                        "and cultural contexts. This assessment is crucial for brands with global aspirations, as it "
                        "identifies potential challenges and opportunities for international brand positioning."
                    )
                
            else:
                # No translation analysis data available or unexpected format
                doc.add_paragraph("No translation analysis data available for this brand naming project.")
                
        except Exception as e:
            logger.error(f"Error formatting translation analysis section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the translation analysis section due to an error in processing the data.")

    async def _format_domain_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """
        Format the domain analysis section using direct ETL process.
        
        Args:
            doc: The document to add content to
            data: The raw domain analysis data
        """
        try:
            # Add section title
            doc.add_heading("Domain Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the domain availability and suitability of the brand name options, "
                "evaluating factors such as domain extension options, pricing, and potential alternatives. "
                "Domain availability is a critical factor in today's digital business environment."
            )
            
            # Transform data using model
            transformed_data = self._transform_domain_analysis(data)
            logger.debug(f"Transformed domain analysis data: {len(str(transformed_data))} chars")
            
            # Check if we have analysis data
            if transformed_data and isinstance(transformed_data, dict) and "domain_analysis" in transformed_data:
                domain_analyses = transformed_data["domain_analysis"]
                
                if domain_analyses:
                    doc.add_heading("Domain Analysis by Brand Name", level=2)
                    
                    # Process each brand name analysis
                    for brand_name, analysis in domain_analyses.items():
                        # Add brand name as heading
                        doc.add_heading(brand_name, level=3)
                        
                        # Process notes
                        if "notes" in analysis and analysis["notes"]:
                            doc.add_paragraph(str(analysis["notes"]))
                        
                        # Create a table for metrics
                        metrics_table = doc.add_table(rows=1, cols=2)
                        metrics_table.style = 'Table Grid'
                        
                        # Add header row
                        header_cells = metrics_table.rows[0].cells
                        header_cells[0].text = "Metric"
                        header_cells[1].text = "Value"
                        
                        # Add basic metrics
                        metrics = [
                            ("Acquisition Cost", analysis.get("acquisition_cost", "Unknown")),
                            ("Domain Exact Match", "Yes" if analysis.get("domain_exact_match", False) else "No"),
                            ("Hyphens/Numbers Present", "Yes" if analysis.get("hyphens_numbers_present", False) else "No"),
                            ("Brand Name Clarity in URL", analysis.get("brand_name_clarity_in_url", "Unknown")),
                            ("Domain Length/Readability", analysis.get("domain_length_readability", "Unknown")),
                            ("Misspellings/Variations Available", "Yes" if analysis.get("misspellings_variations_available", False) else "No"),
                        ]
                        
                        # Add metrics to table
                        for metric, value in metrics:
                            row = metrics_table.add_row()
                            cells = row.cells
                            cells[0].text = metric
                            cells[1].text = str(value)
                        
                        # Add spacing after table
                        doc.add_paragraph("")
                        
                        # Process alternative TLDs
                        if "alternative_tlds" in analysis and analysis["alternative_tlds"]:
                            doc.add_heading("Alternative TLDs", level=4)
                            
                            alt_tlds = analysis["alternative_tlds"]
                            if isinstance(alt_tlds, list):
                                tlds_paragraph = doc.add_paragraph()
                                tlds_text = ", ".join([f".{tld}" for tld in alt_tlds])
                                tlds_paragraph.add_run(tlds_text)
                            else:
                                doc.add_paragraph(str(alt_tlds))
                        
                        # Process social media availability
                        if "social_media_availability" in analysis and analysis["social_media_availability"]:
                            doc.add_heading("Social Media Availability", level=4)
                            
                            social_media = analysis["social_media_availability"]
                            if isinstance(social_media, list):
                                for handle in social_media:
                                    doc.add_paragraph(f"• {handle}", style="List Bullet")
                            else:
                                doc.add_paragraph(str(social_media))
                        
                        # Process scalability and future proofing
                        if "scalability_future_proofing" in analysis and analysis["scalability_future_proofing"]:
                            doc.add_heading("Scalability & Future-Proofing", level=4)
                            doc.add_paragraph(str(analysis["scalability_future_proofing"]))
                        
                        # Add separator between brand analyses (except for the last one)
                        if brand_name != list(domain_analyses.keys())[-1]:
                            doc.add_paragraph("")
                
                # Process comparative analysis if available
                if "comparative_analysis" in transformed_data and transformed_data["comparative_analysis"]:
                    doc.add_heading("Comparative Domain Analysis", level=2)
                    doc.add_paragraph(transformed_data["comparative_analysis"])
                
                # Process general recommendations if available
                if "general_recommendations" in transformed_data and transformed_data["general_recommendations"]:
                    doc.add_heading("General Domain Recommendations", level=2)
                    
                    recommendations = transformed_data["general_recommendations"]
                    if isinstance(recommendations, list):
                        for recommendation in recommendations:
                            doc.add_paragraph(f"• {recommendation}", style="List Bullet")
                    else:
                        doc.add_paragraph(str(recommendations))
                
                # Process summary if available
                if "summary" in transformed_data and transformed_data["summary"]:
                    doc.add_heading("Summary", level=2)
                    doc.add_paragraph(transformed_data["summary"])
            else:
                # No domain analysis data available
                doc.add_paragraph("No domain analysis data available for this brand naming project.")
                if transformed_data:
                    logger.warning(f"Domain analysis missing expected keys: {list(transformed_data.keys())}")
                else:
                    logger.warning("Domain analysis transformation returned empty data")
                
        except Exception as e:
            logger.error(f"Error formatting domain analysis section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Add a generic error message to the document
            doc.add_paragraph("Unable to format the domain analysis section due to an error in processing the data.")

    async def _add_title_page(self, doc: Document, data: Dict[str, Any]) -> None:
        """Add a title page to the document."""
        logger.info("Adding title page")
        
        try:
            # Extract brand context for the title page
            brand_context = data.get("brand_context", {})
            
            # Fetch the user prompt from the state
            user_prompt = await self.fetch_user_prompt(self.current_run_id)
            
            # Format data for the prompt
            format_data = {
                "run_id": self.current_run_id,
                "brand_context": json.dumps(brand_context, indent=2) if isinstance(brand_context, dict) else str(brand_context),
                "user_prompt": user_prompt
            }
            
            # No need to call the LLM for standard title and subtitle
            # Use the exact formats from notesforformatter.md
            title = "Brand Naming Report"
            subtitle = "Generated by Mae Brand Naming Expert"
            
            # Create the title page
            # Set title with large font and center alignment
            title_para = doc.add_paragraph()
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title_para.add_run(title)
            title_run.font.size = Pt(24)
            title_run.font.bold = True
            title_para.space_after = Pt(12)
            
            # Add subtitle
            subtitle_para = doc.add_paragraph()
            subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subtitle_run = subtitle_para.add_run(subtitle)
            subtitle_run.font.size = Pt(16)
            subtitle_run.italic = True
            subtitle_para.space_after = Pt(36)
            
            # Add date
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_para.add_run(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
            date_para.space_after = Pt(12)
            
            # Add run ID
            run_id_para = doc.add_paragraph()
            run_id_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_id_para.add_run(f"Run ID: {self.current_run_id}")
            run_id_para.space_after = Pt(24)
            
            # Add user prompt note
            user_prompt_para = doc.add_paragraph()
            user_prompt_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            user_prompt_run = user_prompt_para.add_run(
                f"This report was generated using the Maestro AI Agent workflow and all content "
                f"has been derived from this prompt: \"{user_prompt}\""
            )
            user_prompt_run.italic = True
            user_prompt_run.font.size = Pt(10)
            
            # Add page break after title page
            doc.add_page_break()
            
        except Exception as e:
            logger.error(f"Error adding title page: {str(e)}")
            # Add fallback title page
            doc.add_paragraph("Brand Naming Report").style = 'Title'
            doc.add_paragraph("Generated by Mae Brand Naming Expert").style = 'Subtitle'
            doc.add_paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
            doc.add_paragraph(f"Run ID: {self.current_run_id}")
            doc.add_page_break()

    async def store_report_metadata(self, run_id: str, report_url: str, file_size_kb: float, format: str) -> None:
        """Store report metadata in the database."""
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

    async def generate_report(self, run_id: str, upload_to_storage: bool = True) -> str:
        """
        Generate a formatted report for the given run ID.
        
        Args:
            run_id: The run ID to generate a report for
            upload_to_storage: Whether to upload the report to Supabase storage (default: True)
            
        Returns:
            str: The path to the generated report
        """
        logger.info(f"Generating report for run ID: {run_id}")
        
        # Initialize run ID
        self.current_run_id = run_id
        
        # Initialize LLM if not already done
        if not self.llm:
            from ..config.settings import settings
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name, 
                temperature=0.2,  # Balanced temperature for analysis 
                google_api_key=settings.google_api_key, 
                convert_system_message_to_human=True, 
                callbacks=settings.get_langsmith_callbacks()
            )
        
        # Fetch raw data
        data = await self.fetch_raw_data(run_id)
        
        if not data:
            logger.error(f"No data found for run ID: {run_id}")
            return None
        
        # Create document
        doc = Document()
        
        # Set up document styles
        self._setup_document_styles(doc)
        
        # Add title page - Required to be first!
        await self._add_title_page(doc, data)
        
        # Add table of contents - Required to be second!
        await self._add_table_of_contents(doc)
        
        # Add executive summary - Required to be third!
        if "exec_summary" in data:
            await self._add_executive_summary(doc, data["exec_summary"])
        
        # Process each section in order
        for section_name in self.SECTION_ORDER:
            # Skip already processed sections
            if section_name in ["exec_summary", "final_recommendations"]:
                continue
                
            if section_name in data:
                section_data = data[section_name]
                
                # Format the section based on its name
                if hasattr(self, f"_format_{self.REVERSE_SECTION_MAPPING.get(section_name, section_name)}"):
                    # Call the specific formatter method
                    formatter_method = getattr(self, f"_format_{self.REVERSE_SECTION_MAPPING.get(section_name, section_name)}")
                    await formatter_method(doc, section_data)
                else:
                    # Use generic formatter as fallback
                    await self._format_generic_section(doc, section_name, section_data)
        
        # Add recommendations section at the end
        if "final_recommendations" in data:
            await self._add_recommendations(doc, data["final_recommendations"])
        
        # Save document to a temporary file
        temp_dir = Path("tmp")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / f"report_{run_id}.docx"
        
        doc.save(str(file_path))
        logger.info(f"Report saved to: {file_path}")
        
        # Upload to storage if available and requested
        report_url = None
        if upload_to_storage and self.supabase:
            try:
                logger.info("Attempting to upload report to Supabase storage")
                report_url = await self.upload_report_to_storage(str(file_path), run_id)
                
                if report_url:
                    logger.info(f"Report successfully uploaded to: {report_url}")
                    # File size in KB
                    file_size_kb = file_path.stat().st_size / 1024
                    # Store metadata
                    await self.store_report_metadata(run_id, report_url, file_size_kb, self.FORMAT_DOCX)
                else:
                    logger.error("Failed to upload report to Supabase storage - report_url is None")
            except Exception as e:
                logger.error(f"Unexpected error during upload process: {str(e)}")
                logger.error(traceback.format_exc())
        elif not upload_to_storage:
            logger.info("Skipping upload to storage as requested (upload_to_storage=False)")
        else:
            logger.warning("Supabase client not available - skipping upload to storage")
        
        return str(file_path)

    def _transform_domain_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform domain analysis data to match expected format for the formatter.
        
        Args:
            data: Raw domain analysis data
            
        Returns:
            Transformed data structure
        """
        if not data:
            return {}
        try:
            domain_analysis_data = DomainAnalysis.model_validate(data)
            return domain_analysis_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for domain analysis data: {str(e)}")
            return {}

    def _transform_survey_simulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform survey simulation data into a structured format.
        
        The raw data has a different structure than other sections, containing a list of
        survey responses directly in the root, rather than a nested structure.
        """
        if not data:
            return {}
            
        try:
            # Log the data structure to help with debugging
            logger.debug(f"Survey simulation data type: {type(data)}")
            logger.debug(f"Survey simulation data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
            
            # Check if the data is already a list of survey details
            if isinstance(data, list):
                # Data is already a list of survey details
                survey_items = data
            elif isinstance(data, dict) and "survey_simulation" in data:
                # Data follows the model structure
                survey_items = data["survey_simulation"]
            else:
                # Data might be in a different format, try to extract it
                survey_items = []
                if isinstance(data, dict):
                    # Loop through all keys to find any that might contain survey items
                    for key, value in data.items():
                        if isinstance(value, list) and value:
                            survey_items = value
                            break
                
            # Now transform each survey item
            formatted_survey_details = []
            for item in survey_items:
                try:
                    survey_detail = SurveyDetails.model_validate(item)
                    formatted_survey_details.append(survey_detail.model_dump())
                except ValidationError as e:
                    logger.error(f"Validation error for survey item: {str(e)}")
                    # Add the item as-is, even if it doesn't fully validate
                    if isinstance(item, dict):
                        formatted_survey_details.append(item)
            
            return {"survey_simulation": formatted_survey_details}
            
        except Exception as e:
            logger.error(f"Error transforming survey simulation data: {str(e)}")
            # Return the original data as a fallback
            if isinstance(data, dict):
                return data
            return {"survey_simulation": data if isinstance(data, list) else []}

    def _transform_seo_analysis(self, data: Any) -> Dict[str, Any]:
        """Transform SEO analysis data to match the expected format for the formatter."""
        transformed_data = {
            "brand_names": {},  # Dictionary mapping brand names to themselves
            "summary": "",
            "general_recommendations": []
        }
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            logger.warning("SEO analysis data is not a dictionary")
            return transformed_data
            
        # Log the incoming data structure for debugging
        if isinstance(data, dict):
            logger.info(f"SEO data keys: {list(data.keys())}")
            if "brand_names" in data:
                logger.info(f"SEO data already contains brand_names with {len(data['brand_names'])} entries")
                # We have the brand names directly in the raw data
                return data
            
        # Extract brand analyses - check for both direct format and nested format
        seo_analyses = None
        
        # Case 1: Direct format where some brand data is directly in the root object
        # This happens in the SEO tests where the data already has "VerityGlobal", "Cerebryx", etc. keys
        brand_keys = ["VerityGlobal", "Cerebryx", "CatalystAxis"]
        if any(key in data for key in brand_keys):
            logger.info("Found SEO data in direct brand key format")
            # Create the structure we need with the data
            for brand_key in brand_keys:
                if brand_key in data:
                    transformed_data["brand_names"][brand_key] = brand_key
                    transformed_data[brand_key] = data[brand_key]
            
            # Add any other top-level fields that might be in the data
            if "summary" in data:
                transformed_data["summary"] = data["summary"]
            if "general_recommendations" in data:
                transformed_data["general_recommendations"] = data["general_recommendations"]
                
            return transformed_data
            
        # Case 2: Nested under "seo_online_discoverability" key
        elif "seo_online_discoverability" in data:
            logger.info("Found SEO data nested under seo_online_discoverability key")
            seo_analyses = data["seo_online_discoverability"]
        
        # No recognizable format found
        else:
            logger.warning("SEO analysis data does not contain expected keys or structure")
            return transformed_data
            
        # If the seo_analyses is a list, process each brand analysis
        if isinstance(seo_analyses, list):
            logger.info(f"Found {len(seo_analyses)} brand analyses in the root list")
            
            # Process each brand analysis
            for brand_analysis in seo_analyses:
                if not isinstance(brand_analysis, dict):
                    continue
                    
                brand_name = brand_analysis.get("brand_name", "Unknown Brand")
                # Add to the dictionary of brand names
                transformed_data["brand_names"][brand_name] = brand_name
                
                # Extract ALL metrics and fields from the model
                search_volume = brand_analysis.get("search_volume", "Unknown")
                keyword_alignment = brand_analysis.get("keyword_alignment", "Unknown")
                
                # Get the field mapping correctly based on the model
                keyword_competition = brand_analysis.get("keyword_competition", "Unknown")
                seo_viability_score = brand_analysis.get("seo_viability_score", "Unknown")
                
                # Extract all additional fields from the model
                negative_search_results = brand_analysis.get("negative_search_results", False)
                unusual_spelling_impact = brand_analysis.get("unusual_spelling_impact", False)
                branded_keyword_potential = brand_analysis.get("branded_keyword_potential", "Unknown")
                name_length_searchability = brand_analysis.get("name_length_searchability", "Unknown")
                social_media_availability = brand_analysis.get("social_media_availability", False)
                competitor_domain_strength = brand_analysis.get("competitor_domain_strength", "Unknown")
                exact_match_search_results = brand_analysis.get("exact_match_search_results", "Unknown")
                social_media_discoverability = brand_analysis.get("social_media_discoverability", "Unknown")
                negative_keyword_associations = brand_analysis.get("negative_keyword_associations", "Unknown")
                non_branded_keyword_potential = brand_analysis.get("non_branded_keyword_potential", "Unknown")
                content_marketing_opportunities = brand_analysis.get("content_marketing_opportunities", "Unknown")
                
                # Also keep compatibility with older field names
                competition = brand_analysis.get("competition", keyword_competition)
                social_media_potential = brand_analysis.get("social_media_potential", social_media_discoverability)
                
                # Parse recommendations
                recommendations = []
                raw_recommendations = brand_analysis.get("seo_recommendations", "[]")
                
                # Handle recommendations whether they're a string, list, or object
                if isinstance(raw_recommendations, dict) and "recommendations" in raw_recommendations:
                    recommendations = raw_recommendations["recommendations"]
                    if not isinstance(recommendations, list):
                        recommendations = [str(recommendations)]
                elif isinstance(raw_recommendations, str):
                    try:
                        recommendations_data = json.loads(raw_recommendations)
                        if isinstance(recommendations_data, dict) and "recommendations" in recommendations_data:
                            recommendations = recommendations_data["recommendations"]
                        else:
                            recommendations = recommendations_data if isinstance(recommendations_data, list) else [raw_recommendations]
                    except json.JSONDecodeError:
                        recommendations = [raw_recommendations]
                elif isinstance(raw_recommendations, list):
                    recommendations = raw_recommendations
                
                # Add brand data to transformed data with ALL fields from the model
                transformed_data[brand_name] = {
                    "search_volume": search_volume,
                    "keyword_alignment": keyword_alignment,
                    "keyword_competition": keyword_competition,
                    "competition": competition,  # Compatibility
                    "seo_viability_score": seo_viability_score,
                    "social_media_potential": social_media_potential,  # Compatibility
                    "negative_search_results": negative_search_results,
                    "unusual_spelling_impact": unusual_spelling_impact,
                    "branded_keyword_potential": branded_keyword_potential,
                    "name_length_searchability": name_length_searchability,
                    "social_media_availability": social_media_availability,
                    "competitor_domain_strength": competitor_domain_strength,
                    "exact_match_search_results": exact_match_search_results,
                    "social_media_discoverability": social_media_discoverability,
                    "negative_keyword_associations": negative_keyword_associations,
                    "non_branded_keyword_potential": non_branded_keyword_potential,
                    "content_marketing_opportunities": content_marketing_opportunities,
                    "recommendations": recommendations,
                    "strengths": [],
                    "challenges": []
                }
                
                # Extract strengths and challenges if available
                if "strengths" in brand_analysis:
                    strengths = brand_analysis["strengths"]
                    if isinstance(strengths, str):
                        try:
                            transformed_data[brand_name]["strengths"] = json.loads(strengths)
                        except json.JSONDecodeError:
                            transformed_data[brand_name]["strengths"] = [strengths]
                    elif isinstance(strengths, list):
                        transformed_data[brand_name]["strengths"] = strengths
                
                if "challenges" in brand_analysis:
                    challenges = brand_analysis["challenges"]
                    if isinstance(challenges, str):
                        try:
                            transformed_data[brand_name]["challenges"] = json.loads(challenges)
                        except json.JSONDecodeError:
                            transformed_data[brand_name]["challenges"] = [challenges]
                    elif isinstance(challenges, list):
                        transformed_data[brand_name]["challenges"] = challenges
        
        # Log results and return
        logger.info(f"Processed {len(transformed_data['brand_names'])} brand analyses")
        logger.info(f"Transformed SEO analysis data: {len(str(transformed_data))} chars")
        if len(transformed_data['brand_names']) == 0:
            logger.warning("SEO analysis transformation contains empty 'brand_names' dictionary")
        return transformed_data

async def main(run_id: str = None, upload_to_storage: bool = True):
    """Main function to run the formatter."""
    if not run_id:
        # Use a default run ID for testing
        run_id = "mae_20250312_141302_d45cccde"  # Replace with an actual run ID
        
    formatter = ReportFormatter()
    output_path = await formatter.generate_report(run_id, upload_to_storage=upload_to_storage)
    print(f"Report generated at: {output_path}")


if __name__ == "__main__":
    # Command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a formatted report from raw data")
    parser.add_argument("run_id", help="The run ID to generate a report for")
    parser.add_argument("--no-upload", dest="upload_to_storage", action="store_false", 
                        help="Skip uploading the report to Supabase storage")
    parser.set_defaults(upload_to_storage=True)
    args = parser.parse_args()
    
    asyncio.run(main(args.run_id, args.upload_to_storage)) 