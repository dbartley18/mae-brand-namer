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
    NameCategory,
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
        """Initialize the LLM for formatting."""
        try:
            # Import settings
            from ..config.settings import settings
            
            if settings.google_api_key:
                self.llm = ChatGoogleGenerativeAI(
                    google_api_key=settings.google_api_key,
                    model="gemini-2.0-flash",
                    temperature=0.5,
                    top_k=40,
                    top_p=0.8,
                )
            else:
                logger.warning("No Google API key found, using default LLM")
                self.llm = None
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self.llm = None
            
    def _get_format_instructions(self, section_name: str) -> str:
        """Get formatting instructions for a specific section."""
        # Common instructions for all sections
        common_instructions = (
            "Structure your response as valid JSON with clear hierarchical organization. "
            "Use professional language appropriate for a business document. "
            "Avoid ALL marketing language, superlatives, and unnecessary adjectives."
        )
        
        # Specific instructions by section
        if section_name.lower() == "title page":
            return (
                "Create a professional title page that reflects the branding project's focus. "
                "The title should be concise but descriptive of the specific project. "
                "The subtitle should provide additional context about the purpose of this report."
            )
        elif section_name.lower() == "table of contents":
            return (
                "Create a clear and descriptive table of contents that outlines the major sections of the report. "
                "Each section should have a brief (1-2 sentence) description that explains what information "
                "the reader will find in that section. The descriptions should be factual and focused on "
                "explaining the content and purpose of each section. "
                "Follow the TableOfContentsSection model structure with a sections array and optional introduction."
            )
        elif section_name.lower() == "executive summary":
            return (
                "Provide a concise summary of the key findings and insights from the brand naming analysis. "
                "The summary should be factual and focused on conveying the most important information "
                "to the reader. Avoid any marketing language or superlatives. "
                "Use the exact values provided in the JSON for numeric fields and boolean status flags."
            )
        elif section_name.lower() == "recommendations":
            return (
                "Provide strategic recommendations based on the findings from the brand naming analysis. "
                "The recommendations should be actionable and focused on providing clear and concise advice "
                "to the reader. Avoid any marketing language or superlatives. "
                "Use the exact values provided in the JSON for numeric fields and boolean status flags."
            )
        elif section_name.lower() == "seo analysis":
            return (
                "Provide insights into the SEO potential of the brand name options. "
                "The analysis should include search volume, keyword potential, and social media considerations "
                "that influence a brand's online visibility and findability. "
                "Use the exact values provided in the JSON for numeric fields and boolean status flags."
            )
        elif section_name.lower() == "brand context":
            return (
                "Provide a detailed description of the brand's context, including its brand promise, brand purpose, "
                "brand values, and any other relevant information. Use the exact text from string fields like "
                "brand_promise, brand_purpose, etc. Avoid any marketing language or superlatives. "
                "Maintain the organization of all 13 brand context components in a logical structure."
            )
        elif section_name.lower() == "brand name generation":
            return (
                "Organize content by naming categories and individual brand names. "
                "Present each brand name with its full assessment data (brand_personality_ alignment, brand_promise_alignment, etc.). "
                "Include the introduction, methodology, and evaluation metrics as separate sections. "
                "Ensure that each name's rationale is clearly presented when available."
            )
        elif section_name.lower() == "semantic analysis":
            return (
                "Organize data hierarchically by brand name with detailed semantic analysis for each. "
                "Include all fields from the SemanticAnalysis model: etymology, sound_symbolism, brand_personality, etc. "
                "Present technical linguistic concepts (etymology, phoneme_combinations) in accessible language. "
                "Preserve boolean values (alliteration_assonance) and numeric values (word_length_syllables) with clear explanations. "
                "Highlight semantic trademark risks and their implications for each brand name. "
                "Include comparative analysis of semantic characteristics across all brand names. "
                "Provide a summary of key semantic insights that impact brand perception and memorability."
            )
        elif section_name.lower() == "linguistic analysis":
            return (
                "Organize data by brand name with comprehensive linguistic details for each. "
                "Present technical linguistic aspects (pronunciation, rhythm, morphology) in accessible language. "
                "Include all fields: word_class, sound_symbolism, pronunciation_ease, euphony_vs_cacophony, etc. "
                "Compare phonetic and morphological characteristics across different brand names. "
                "Highlight how linguistic features might impact marketing effectiveness and memorability."
            )
        elif section_name.lower() == "cultural sensitivity analysis":
            return (
                "Organize data by brand name with detailed analysis for each name. "
                "Present the overall_risk_rating prominently for each brand name with clear reasoning. "
                "Include all cultural dimensions: symbolic, historical, religious, social-political, and age-related. "
                "Highlight regional variations in cultural perception across different markets. "
                "Provide clear recommendations for mitigating any identified cultural sensitivity risks."
            )
        elif section_name.lower() == "translation analysis":
            return (
                "Organize the data hierarchically by brand name, then by language. "
                "Include all fields from the LanguageAnalysis model for each language: direct_translation, semantic_shift, adaptation_needed, etc. "
                "Present boolean fields (adaptation_needed) with clear Yes/No statements and accompanying explanations. "
                "Highlight proposed adaptations and their rationale when adaptation_needed is true. "
                "Compare phonetic_retention, brand_essence_preserved, and cultural_acceptability across different languages. "
                "Include analysis of global_consistency_vs_localization for each brand name across markets. "
                "Provide clear recommendations for internationalization strategy based on translation findings."
            )
        elif section_name.lower() == "brand name evaluation":
            return (
                "Extract overall_score as a numeric value, shortlist_status as a boolean, "
                "and analysis details from evaluation_comments. Rank the brand names by overall_score in descending order. "
                "Clearly indicate which names are shortlisted based on the shortlist_status field."
            )
        elif section_name.lower() == "domain analysis":
            return (
                "Organize data by brand name with detailed domain availability information. "
                "Present both boolean fields (domain_exact_match, hyphens_numbers_present, misspellings_variations_available) with explanations. "
                "Include acquisition cost information and assessments of brand name clarity and domain readability. "
                "List available alternative TLDs and social media handles for each brand name. "
                "Provide future-proofing considerations and specific domain strategy recommendations."
            )
        elif section_name.lower() == "seo online discoverability":
            return (
                "Organize data hierarchically by brand name with detailed SEO analysis for each. "
                "Include all fields from the SEOOnlineDiscoverabilityDetails model: search_volume, keyword_alignment, etc. "
                "Present numeric data (search_volume, seo_viability_score) with context and implications. "
                "Format boolean values (negative_search_results, unusual_spelling_impact, social_media_availability) as clear statements. "
                "Present seo_recommendations as a prioritized, bulleted list of actionable recommendations. "
                "Include a comparative analysis showing relative SEO strengths and weaknesses across brand names. "
                "Provide a summary of key SEO insights and their implications for brand naming strategy."
            )
        elif section_name.lower() == "competitor analysis":
            return (
                "Organize data hierarchically by brand name and then by competitor name. "
                "Present the risk_of_confusion as a numeric value (1-10) with clear interpretation. "
                "Highlight key differentiation opportunities and risks for each competitor relationship. "
                "Ensure competitor strengths, weaknesses, and positioning are presented in detail. "
                "Include analysis of trademark conflict risk and target audience perception for each competitor."
            )
        elif section_name.lower() == "market research":
            return (
                "Organize data hierarchically by brand name with detailed market analysis for each. "
                "Extract and summarize industry-level insights (industry_name, market_size, growth trends) across all brands. "
                "For each brand name, include all fields from the MarketResearchDetails model: market_size, industry_name, emerging_trends, etc. "
                "Present list fields (key_competitors, customer_pain_points) as clearly formatted bullet points. "
                "Include comparative analysis across all brands highlighting market positioning differences. "
                "Provide a summary of key market insights and implications for brand naming strategy."
            )
        elif section_name.lower() == "survey simulation":
            return (
                "Organize data hierarchically by brand name with detailed persona information for each. "
                "Include all fields from the SurveySimulationDetails model: industry, job_title, seniority, etc. "
                "Present numeric scores (brand_promise_perception_score, personality_fit_score, competitive_differentiation_score, simulated_market_adoption_score) with context and interpretation. "
                "Include verbatim quotes from raw_qualitative_feedback including all seven feedback categories. "
                "Present the content from the nested models (RawQualitativeFeedback and CurrentBrandRelationships) clearly labeled. "
                "Provide comparative analysis across brand names highlighting differences in perception and market potential. "
                "Include specific final_survey_recommendation and qualitative_feedback_summary for each brand name."
            )
        
        # Default instructions if no specific section match
        return common_instructions

    def _format_template(self, template_name: str, format_data: Dict[str, Any], section_name: str = None) -> str:
        """Format a prompt template with provided data."""
        try:
            # Get template name based on section name if provided
            if section_name is not None:
                section_file = self.REVERSE_SECTION_MAPPING.get(section_name, section_name.lower())
                actual_template_name = section_file
                logger.debug(f"Using section file: {section_file} for section: {section_name}")
            else:
                actual_template_name = template_name
                logger.debug(f"Using direct template name: {template_name}")
                
            # Get format instructions if section_name is provided and not already present
            if section_name and 'format_instructions' not in format_data:
                format_data['format_instructions'] = self._get_format_instructions(section_name)
                logger.debug(f"Added format_instructions for {section_name}")
            
            # Try using the utility functions from __init__.py based on template name
            try:
                # Map template names to utility functions
                template_map = {
                    "title_page": get_title_page_prompt,
                    "table_of_contents": get_toc_prompt,
                    "executive_summary": get_executive_summary_prompt,
                    "recommendations": get_recommendations_prompt,
                    "seo_analysis": get_seo_analysis_prompt,
                    "brand_context": get_brand_context_prompt,
                    "brand_name_generation": get_brand_name_generation_prompt,
                    "semantic_analysis": get_semantic_analysis_prompt,
                    "linguistic_analysis": get_linguistic_analysis_prompt,
                    "cultural_sensitivity": get_cultural_sensitivity_prompt,
                    "translation_analysis": get_translation_analysis_prompt,
                    "market_research": get_market_research_prompt,
                    "competitor_analysis": get_competitor_analysis_prompt,
                    "name_evaluation": get_name_evaluation_prompt,
                    "domain_analysis": get_domain_analysis_prompt,
                    "survey_simulation": get_survey_simulation_prompt,
                    "system": get_system_prompt,
                    "shortlisted_names_summary": get_shortlisted_names_summary_prompt,
                    "format_section": get_format_section_prompt
                }
                
                if actual_template_name in template_map:
                    logger.debug(f"Using utility function for {actual_template_name}")
                    
                    # Make sure all required variables are in format_data
                    if actual_template_name == "brand_name_generation" and "brand_name_generation" in format_data:
                        # Additional handling for brand_name_generation data
                        # Ensure the data is properly processed
                        logger.debug("Special handling for brand_name_generation template")
                        if isinstance(format_data["brand_name_generation"], str):
                            try:
                                # Try to parse as JSON if it's not already
                                brand_name_data = json.loads(format_data["brand_name_generation"])
                                format_data["brand_name_generation"] = json.dumps(brand_name_data, indent=2)
                                logger.debug("Successfully parsed and reformatted brand_name_generation data")
                            except json.JSONDecodeError:
                                # Already a string, leave as is
                                pass
                    
                    prompt_data = template_map[actual_template_name](**format_data)
                    if "template" in prompt_data:
                        template_content = prompt_data.get("template", "")
                        logger.debug(f"Got template from utility function: {template_content[:50]}...")
                        
                        # Direct variable substitution for brand_name_generation template
                        if actual_template_name == "brand_name_generation":
                            for key, value in format_data.items():
                                placeholder = "{{" + key + "}}"
                                if placeholder in template_content:
                                    if isinstance(value, (dict, list)):
                                        value = json.dumps(value, indent=2)
                                    template_content = template_content.replace(placeholder, str(value))
                                    logger.debug(f"Directly replaced {placeholder} in template")
                        
                        return template_content
                    else:
                        logger.warning(f"Utility function for {actual_template_name} didn't return a template key")
            except Exception as e:
                logger.warning(f"Error using utility function for {actual_template_name}: {str(e)}")
                # Continue to fallback method
            
            # Fallback to using the pre-loaded prompt templates
            logger.debug(f"Falling back to cached prompts for {actual_template_name}")
            
            # Use the pre-loaded prompt template from self.prompts
            if actual_template_name in self.prompts:
                prompt_template = self.prompts[actual_template_name]
                logger.debug(f"Found template for {actual_template_name} in self.prompts")
                
                # Format the template with data
                logger.debug(f"Formatting template with variables: {list(format_data.keys())}")
                
                # For brand_name_generation, use direct replacement instead of .format()
                if actual_template_name == "brand_name_generation":
                    template_content = prompt_template.template
                    for key, value in format_data.items():
                        placeholder = "{{" + key + "}}"
                        if placeholder in template_content:
                            if isinstance(value, (dict, list)):
                                value = json.dumps(value, indent=2)
                            template_content = template_content.replace(placeholder, str(value))
                            logger.debug(f"Directly replaced {placeholder} in template")
                    formatted_template = template_content
                else:
                    # Use normal format for other templates
                    formatted_template = prompt_template.format(**format_data)
                
                # Log a preview of the formatted template (first 100 chars)
                logger.debug(f"Formatted template preview: {formatted_template[:100]}...")
                return formatted_template
            else:
                logger.warning(f"No template found for {actual_template_name}, trying generic template")
                # Debug info about available templates
                logger.debug(f"Available templates: {list(self.prompts.keys())}")
                
                if "format_section" in self.prompts:
                    prompt_template = self.prompts["format_section"]
                    logger.debug("Using format_section fallback template")
                    
                    # Format the template with data
                    formatted_template = prompt_template.format(**format_data)
                    logger.debug(f"Formatted template preview: {formatted_template[:100]}...")
                    return formatted_template
                else:
                    # Last resort fallback
                    logger.error(f"No template found for {actual_template_name} and no format_section fallback")
                    raise ValueError(f"No template found for {actual_template_name} and no format_section fallback")
            
        except Exception as e:
            logger.error(f"Error formatting template {template_name}: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Return a simple fallback template
            if section_name:
                return f"Please format the {section_name} section data into a professional report section."
            return f"Please format the provided data for {template_name} into a professional report section."
    
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
        if not data:
            return {}
        try:
            cultural_sensitivity_analysis_data = CulturalSensitivityAnalysis.model_validate(data)
            return cultural_sensitivity_analysis_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for cultural sensitivity analysis data: {str(e)}")
            return {}

    def _transform_name_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            name_evaluation_data = BrandNameEvaluation.model_validate(data)
            return name_evaluation_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for name evaluation data: {str(e)}")
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
        if not data:
            return {}
        try:
            market_research_data = MarketResearch.model_validate(data)
            return market_research_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for market research data: {str(e)}")
            return {}

    def _transform_competitor_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            competitor_analysis_data = CompetitorAnalysis.model_validate(data)
            return competitor_analysis_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for competitor analysis data: {str(e)}")
            return {}

    def _transform_domain_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            domain_analysis_data = DomainAnalysis.model_validate(data)
            return domain_analysis_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for domain analysis data: {str(e)}")
            return {}

    def _transform_survey_simulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            survey_simulation_data = SurveySimulation.model_validate(data)
            return survey_simulation_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for survey simulation data: {str(e)}")
            return {}

    def _transform_seo_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            seo_analysis_data = SEOOnlineDiscoverability.model_validate(data)
            return seo_analysis_data.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for SEO analysis data: {str(e)}")
            return {}

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
        """Format the survey simulation section."""
        try:
            # Add section title
            doc.add_heading("Survey Simulation Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section presents simulated market research findings based on "
                "survey responses from target audience personas. The analysis includes detailed persona profiles, "
                "qualitative feedback, perception scores, and adoption potential for each brand name option."
            )
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        "survey_simulation": json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                        "format_instructions": self._get_format_instructions("survey_simulation")
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template("survey_simulation", format_data, "survey_simulation")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "survey_simulation")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "survey_simulation")
                    
                    if json_content:
                        # Add methodology if provided
                        if "methodology" in json_content and json_content["methodology"]:
                            doc.add_heading("Survey Methodology", level=2)
                            doc.add_paragraph(json_content["methodology"])
                        
                        # Add demographics if provided
                        if "demographics" in json_content and json_content["demographics"]:
                            doc.add_heading("Participant Demographics", level=2)
                            doc.add_paragraph(json_content["demographics"])
                        
                        # Add overview if provided
                        if "overview" in json_content and json_content["overview"]:
                            doc.add_heading("Survey Overview", level=2)
                            doc.add_paragraph(json_content["overview"])
                        
                        # Format each brand analysis
                        if "brand_analyses" in json_content and isinstance(json_content["brand_analyses"], list):
                            for analysis in json_content["brand_analyses"]:
                                if "brand_name" in analysis:
                                    # Add brand name heading
                                    doc.add_heading(analysis["brand_name"], level=2)
                                    
                                    # Add persona profiles if available
                                    if "persona_profiles" in analysis and isinstance(analysis["persona_profiles"], list):
                                        doc.add_heading("Persona Profiles", level=3)
                                        for i, profile in enumerate(analysis["persona_profiles"], 1):
                                            doc.add_heading(f"Persona {i}", level=4)
                                            
                                            # Professional info
                                            if "professional_info" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Professional Background: ").bold = True
                                                p.add_run(profile["professional_info"])
                                            
                                            # Company context
                                            if "company_context" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Company Context: ").bold = True
                                                p.add_run(profile["company_context"])
                                            
                                            # Decision making
                                            if "decision_making" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Decision-Making Authority: ").bold = True
                                                p.add_run(profile["decision_making"])
                                            
                                            # Pain points
                                            if "pain_points_challenges" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Pain Points & Challenges: ").bold = True
                                                p.add_run(profile["pain_points_challenges"])
                                            
                                            # Goals and priorities
                                            if "goals_priorities" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Goals & Priorities: ").bold = True
                                                p.add_run(profile["goals_priorities"])
                                            
                                            # Information consumption
                                            if "information_consumption" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Information Consumption: ").bold = True
                                                p.add_run(profile["information_consumption"])
                                            
                                            # Current relationships
                                            if "current_relationships" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Current Brand Relationships: ").bold = True
                                                p.add_run(profile["current_relationships"])
                                            
                                            # Buying behavior
                                            if "buying_behavior" in profile:
                                                p = doc.add_paragraph()
                                                p.add_run("Buying Behavior: ").bold = True
                                                p.add_run(profile["buying_behavior"])
                                    
                                    # Add brand promise alignment
                                    if "brand_promise_alignment" in analysis and analysis["brand_promise_alignment"]:
                                        doc.add_heading("Brand Promise Alignment", level=3)
                                        doc.add_paragraph(analysis["brand_promise_alignment"])
                                    
                                    # Add personality fit
                                    if "personality_fit" in analysis and analysis["personality_fit"]:
                                        doc.add_heading("Personality Fit", level=3)
                                        doc.add_paragraph(analysis["personality_fit"])
                                    
                                    # Add emotional impact
                                    if "emotional_impact" in analysis and analysis["emotional_impact"]:
                                        doc.add_heading("Emotional Impact", level=3)
                                        doc.add_paragraph(analysis["emotional_impact"])
                                    
                                    # Add competitive positioning
                                    if "competitive_positioning" in analysis and analysis["competitive_positioning"]:
                                        doc.add_heading("Competitive Positioning", level=3)
                                        doc.add_paragraph(analysis["competitive_positioning"])
                                    
                                    # Add market receptivity
                                    if "market_receptivity" in analysis and analysis["market_receptivity"]:
                                        doc.add_heading("Market Adoption Potential", level=3)
                                        doc.add_paragraph(analysis["market_receptivity"])
                                    
                                    # Add raw qualitative feedback
                                    if "raw_qualitative_feedback" in analysis and isinstance(analysis["raw_qualitative_feedback"], dict):
                                        doc.add_heading("Qualitative Feedback", level=3)
                                        feedback = analysis["raw_qualitative_feedback"]
                                        
                                        # Relevance
                                        if "relevance" in feedback:
                                            p = doc.add_paragraph()
                                            p.add_run("Relevance: ").bold = True
                                            p.add_run(feedback["relevance"])
                                        
                                        # Memorability
                                        if "memorability" in feedback:
                                            p = doc.add_paragraph()
                                            p.add_run("Memorability: ").bold = True
                                            p.add_run(feedback["memorability"])
                                        
                                        # Pronunciation
                                        if "pronunciation" in feedback:
                                            p = doc.add_paragraph()
                                            p.add_run("Pronunciation: ").bold = True
                                            p.add_run(feedback["pronunciation"])
                                        
                                        # Visual imagery
                                        if "visual_imagery" in feedback:
                                            p = doc.add_paragraph()
                                            p.add_run("Visual Imagery: ").bold = True
                                            p.add_run(feedback["visual_imagery"])
                                        
                                        # Differentiation
                                        if "differentiation" in feedback:
                                            p = doc.add_paragraph()
                                            p.add_run("Differentiation: ").bold = True
                                            p.add_run(feedback["differentiation"])
                                        
                                        # Emotional impact
                                        if "emotional_impact" in feedback:
                                            p = doc.add_paragraph()
                                            p.add_run("Emotional Impact: ").bold = True
                                            p.add_run(feedback["emotional_impact"])
                                        
                                        # First impression
                                        if "first_impression" in feedback:
                                            p = doc.add_paragraph()
                                            p.add_run("First Impression: ").bold = True
                                            p.add_run(feedback["first_impression"])
                                    
                                    # Add final recommendations
                                    if "final_recommendations" in analysis and analysis["final_recommendations"]:
                                        doc.add_heading("Final Recommendations", level=3)
                                        doc.add_paragraph(analysis["final_recommendations"])
                        
                        # Add comparative analysis
                        if "comparative_analysis" in json_content and json_content["comparative_analysis"]:
                            doc.add_heading("Comparative Analysis", level=2)
                            doc.add_paragraph(json_content["comparative_analysis"])
                        
                        # Add strategic implications
                        if "strategic_implications" in json_content and json_content["strategic_implications"]:
                            doc.add_heading("Strategic Implications", level=2)
                            doc.add_paragraph(json_content["strategic_implications"])
                        
                        return  # Successfully formatted with LLM
                except Exception as e:
                    logger.error(f"Error formatting survey simulation with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Check if data is in the SurveySimulation model format
            if "survey_simulation" in data and isinstance(data["survey_simulation"], dict):
                section_data = data["survey_simulation"]
                
                for brand_name, details in section_data.items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Create a table for key metrics
                    table = doc.add_table(rows=5, cols=2)
                    table.style = 'TableGrid'
                    
                    # Set header row
                    header_cells = table.rows[0].cells
                    header_cells[0].text = "Metric"
                    header_cells[1].text = "Value"
                    
                    # Add brand promise perception score
                    if "brand_promise_perception_score" in details:
                        row = table.rows[1].cells
                        row[0].text = "Brand Promise Perception Score"
                        row[1].text = str(details["brand_promise_perception_score"])
                    
                    # Add personality fit score
                    if "personality_fit_score" in details:
                        row = table.rows[2].cells
                        row[0].text = "Personality Fit Score"
                        row[1].text = str(details["personality_fit_score"])
                    
                    # Add competitive differentiation score
                    if "competitive_differentiation_score" in details:
                        row = table.rows[3].cells
                        row[0].text = "Competitive Differentiation Score"
                        row[1].text = str(details["competitive_differentiation_score"])
                    
                    # Add simulated market adoption score
                    if "simulated_market_adoption_score" in details:
                        row = table.rows[4].cells
                        row[0].text = "Simulated Market Adoption Score"
                        row[1].text = str(details["simulated_market_adoption_score"])
                    
                    # Add spacing
                    doc.add_paragraph()
                    
                    # Add persona details
                    doc.add_heading("Persona Details", level=3)
                    
                    # Professional background
                    if any(field in details for field in ["industry", "job_title", "seniority", "years_of_experience"]):
                        doc.add_heading("Professional Background", level=4)
                        
                        if "industry" in details:
                            p = doc.add_paragraph()
                            p.add_run("Industry: ").bold = True
                            p.add_run(details["industry"])
                        
                        if "job_title" in details:
                            p = doc.add_paragraph()
                            p.add_run("Job Title: ").bold = True
                            p.add_run(details["job_title"])
                        
                        if "seniority" in details:
                            p = doc.add_paragraph()
                            p.add_run("Seniority: ").bold = True
                            p.add_run(details["seniority"])
                        
                        if "years_of_experience" in details:
                            p = doc.add_paragraph()
                            p.add_run("Years of Experience: ").bold = True
                            p.add_run(str(details["years_of_experience"]))
                    
                    # Company context
                    if any(field in details for field in ["company_name", "company_size_employees", "company_revenue"]):
                        doc.add_heading("Company Context", level=4)
                        
                        if "company_name" in details:
                            p = doc.add_paragraph()
                            p.add_run("Company Name: ").bold = True
                            p.add_run(details["company_name"])
                        
                        if "company_size_employees" in details:
                            p = doc.add_paragraph()
                            p.add_run("Company Size (Employees): ").bold = True
                            p.add_run(details["company_size_employees"])
                        
                        if "company_revenue" in details:
                            p = doc.add_paragraph()
                            p.add_run("Company Revenue: ").bold = True
                            p.add_run(f"${details['company_revenue']:,}")
                    
                    # Decision making
                    if any(field in details for field in ["decision_making_style", "budget_authority", "influence_within_company"]):
                        doc.add_heading("Decision Making", level=4)
                        
                        if "decision_making_style" in details:
                            p = doc.add_paragraph()
                            p.add_run("Decision Making Style: ").bold = True
                            p.add_run(details["decision_making_style"])
                        
                        if "budget_authority" in details:
                            p = doc.add_paragraph()
                            p.add_run("Budget Authority: ").bold = True
                            p.add_run(details["budget_authority"])
                        
                        if "influence_within_company" in details:
                            p = doc.add_paragraph()
                            p.add_run("Influence Within Company: ").bold = True
                            p.add_run(details["influence_within_company"])
                    
                    # Raw qualitative feedback
                    if "raw_qualitative_feedback" in details and isinstance(details["raw_qualitative_feedback"], dict):
                        doc.add_heading("Qualitative Feedback", level=3)
                        feedback = details["raw_qualitative_feedback"]
                        
                        if "relevance" in feedback:
                            p = doc.add_paragraph()
                            p.add_run("Relevance: ").bold = True
                            p.add_run(feedback["relevance"])
                        
                        if "memorability" in feedback:
                            p = doc.add_paragraph()
                            p.add_run("Memorability: ").bold = True
                            p.add_run(feedback["memorability"])
                        
                        if "pronunciation" in feedback:
                            p = doc.add_paragraph()
                            p.add_run("Pronunciation: ").bold = True
                            p.add_run(feedback["pronunciation"])
                        
                        if "visual_imagery" in feedback:
                            p = doc.add_paragraph()
                            p.add_run("Visual Imagery: ").bold = True
                            p.add_run(feedback["visual_imagery"])
                        
                        if "differentiation" in feedback:
                            p = doc.add_paragraph()
                            p.add_run("Differentiation: ").bold = True
                            p.add_run(feedback["differentiation"])
                        
                        if "emotional_impact" in feedback:
                            p = doc.add_paragraph()
                            p.add_run("Emotional Impact: ").bold = True
                            p.add_run(feedback["emotional_impact"])
                        
                        if "first_impression" in feedback:
                            p = doc.add_paragraph()
                            p.add_run("First Impression: ").bold = True
                            p.add_run(feedback["first_impression"])
                    
                    # Current brand relationships
                    if "current_brand_relationships" in details and isinstance(details["current_brand_relationships"], dict):
                        doc.add_heading("Current Brand Relationships", level=3)
                        relationships = details["current_brand_relationships"]
                        
                        if "sap" in relationships:
                            p = doc.add_paragraph()
                            p.add_run("SAP: ").bold = True
                            p.add_run(relationships["sap"])
                        
                        if "oracle" in relationships:
                            p = doc.add_paragraph()
                            p.add_run("Oracle: ").bold = True
                            p.add_run(relationships["oracle"])
                        
                        if "microsoft" in relationships:
                            p = doc.add_paragraph()
                            p.add_run("Microsoft: ").bold = True
                            p.add_run(relationships["microsoft"])
                        
                        if "salesforce" in relationships:
                            p = doc.add_paragraph()
                            p.add_run("Salesforce: ").bold = True
                            p.add_run(relationships["salesforce"])
                        
                        if "aws" in relationships:
                            p = doc.add_paragraph()
                            p.add_run("AWS: ").bold = True
                            p.add_run(relationships["aws"])
                        
                        if "gcp" in relationships:
                            p = doc.add_paragraph()
                            p.add_run("GCP: ").bold = True
                            p.add_run(relationships["gcp"])
                    
                    # Final survey recommendation
                    if "final_survey_recommendation" in details:
                        doc.add_heading("Final Survey Recommendation", level=3)
                        doc.add_paragraph(details["final_survey_recommendation"])
                
                # Add summary section
                doc.add_heading("Survey Simulation Summary", level=2)
                doc.add_paragraph(
                    "This survey simulation analysis provides valuable insights into how potential customers "
                    "might perceive and respond to each brand name option. The findings highlight key strengths "
                    "and weaknesses of each name from the perspective of the target audience."
                )
            else:
                # Fallback for unstructured data
                doc.add_paragraph("Survey simulation data could not be properly formatted.")
                doc.add_paragraph(str(data))
                
        except Exception as e:
            logger.error(f"Error formatting survey simulation: {str(e)}")
            doc.add_paragraph(f"Error formatting survey simulation section: {str(e)}", style='Intense Quote')

    async def _format_linguistic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the linguistic analysis section."""
        try:
            # Add section title
            doc.add_heading("Linguistic Analysis", level=1)
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Check if data is already in the expected structure
                    linguistic_data = data
                    if "linguistic_analysis" in data and isinstance(data["linguistic_analysis"], dict):
                        # Data is already in the correct Pydantic model structure
                        linguistic_data = data
                    else:
                        # Convert legacy structure to match the Pydantic model
                        linguistic_data = {"linguistic_analysis": {}}
                        
                        # Check if we have linguistic_analyses list structure (old format)
                        if "linguistic_analyses" in data and isinstance(data["linguistic_analyses"], list):
                            analyses = data["linguistic_analyses"]
                            
                            for analysis in analyses:
                                if "brand_name" in analysis:
                                    brand_name = analysis["brand_name"]
                                    features = analysis.get("features", [])
                                    
                                    # Extract linguistic features into a structured format
                                    linguistic_data["linguistic_analysis"][brand_name] = {
                                        "notes": "Linguistic analysis notes for " + brand_name,
                                        "word_class": "Unknown",
                                        "sound_symbolism": "Unknown",
                                        "rhythm_and_meter": "Unknown",
                                        "pronunciation_ease": "Unknown",
                                        "euphony_vs_cacophony": "Unknown",
                                        "inflectional_properties": "Unknown",
                                        "neologism_appropriateness": "Unknown",
                                        "overall_readability_score": "Unknown",
                                        "morphological_transparency": "Unknown",
                                        "naturalness_in_collocations": "Unknown",
                                        "ease_of_marketing_integration": "Unknown",
                                        "phoneme_frequency_distribution": "Unknown",
                                        "semantic_distance_from_competitors": "Unknown"
                                    }
                                    
                                    # Try to extract actual features from the features list
                                    for feature in features:
                                        if "pronunciation" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["pronunciation_ease"] = feature
                                        elif "sound" in feature.lower() or "symbolic" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["sound_symbolism"] = feature
                                        elif "rhythm" in feature.lower() or "meter" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["rhythm_and_meter"] = feature
                                        elif "euphony" in feature.lower() or "cacophony" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["euphony_vs_cacophony"] = feature
                                        elif "word class" in feature.lower() or "part of speech" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["word_class"] = feature
                                        elif "readability" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["overall_readability_score"] = feature
                                        elif "marketing" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["ease_of_marketing_integration"] = feature
                                        elif "competitor" in feature.lower() or "similar" in feature.lower():
                                            linguistic_data["linguistic_analysis"][brand_name]["semantic_distance_from_competitors"] = feature
                                        # Add other mappings as needed
                    
                    # Create a prompt with the format data
                    format_data = {
                        "run_id": self.run_id,
                        "linguistic_analysis": json.dumps(linguistic_data),
                        "format_instructions": self._get_format_instructions("linguistic_analysis")
                    }
                    
                    # Format template
                    content = await self._format_template("linguistic_analysis", format_data)
                    
                    # Extract JSON content from LLM response
                    json_content = self._extract_json_from_response(content)
                    
                    if json_content:
                        # Add introduction
                        introduction = json_content.get("introduction", "This section presents linguistic analysis findings based on the analysis of brand names and their associated linguistic features.")
                        doc.add_paragraph(introduction)
                        
                        # Add brand analyses
                        brand_analyses = json_content.get("brand_analyses", [])
                        for brand_analysis in brand_analyses:
                            if isinstance(brand_analysis, dict) and "brand_name" in brand_analysis:
                                # Add brand name heading
                                brand_name = brand_analysis.get("brand_name")
                                doc.add_heading(brand_name, level=2)
                                
                                # Add pronunciation analysis
                                pronunciation = brand_analysis.get("pronunciation_ease")
                                if pronunciation:
                                    doc.add_heading("Pronunciation Ease", level=3)
                                    doc.add_paragraph(pronunciation)
                                
                                # Add sound quality/symbolism
                                sound_symbolism = brand_analysis.get("sound_symbolism")
                                if sound_symbolism:
                                    doc.add_heading("Sound Symbolism", level=3)
                                    doc.add_paragraph(sound_symbolism)
                                
                                # Add rhythmic analysis
                                rhythm = brand_analysis.get("rhythm_and_meter")
                                if rhythm:
                                    doc.add_heading("Rhythm and Meter", level=3)
                                    doc.add_paragraph(rhythm)
                                
                                # Add phoneme distribution
                                phoneme = brand_analysis.get("phoneme_frequency_distribution")
                                if phoneme:
                                    doc.add_heading("Phoneme Frequency Distribution", level=3)
                                    doc.add_paragraph(phoneme)
                                
                                # Add euphony vs cacophony
                                euphony = brand_analysis.get("euphony_vs_cacophony")
                                if euphony:
                                    doc.add_heading("Euphony vs Cacophony", level=3)
                                    doc.add_paragraph(euphony)
                                
                                # Add word class
                                word_class = brand_analysis.get("word_class")
                                if word_class:
                                    doc.add_heading("Word Class", level=3)
                                    doc.add_paragraph(word_class)
                                
                                # Add morphological analysis
                                morphological = brand_analysis.get("morphological_transparency")
                                if morphological:
                                    doc.add_heading("Morphological Transparency", level=3)
                                    doc.add_paragraph(morphological)
                                
                                # Add inflectional properties
                                inflectional = brand_analysis.get("inflectional_properties")
                                if inflectional:
                                    doc.add_heading("Inflectional Properties", level=3)
                                    doc.add_paragraph(inflectional)
                                
                                # Add marketing integration
                                marketing = brand_analysis.get("ease_of_marketing_integration")
                                if marketing:
                                    doc.add_heading("Marketing Integration", level=3)
                                    doc.add_paragraph(marketing)
                                
                                # Add linguistic naturalness
                                naturalness = brand_analysis.get("naturalness_in_collocations")
                                if naturalness:
                                    doc.add_heading("Naturalness in Collocations", level=3)
                                    doc.add_paragraph(naturalness)
                                
                                # Add competitive distinction
                                competitive = brand_analysis.get("semantic_distance_from_competitors")
                                if competitive:
                                    doc.add_heading("Semantic Distance from Competitors", level=3)
                                    doc.add_paragraph(competitive)
                                
                                # Add neologism assessment
                                neologism = brand_analysis.get("neologism_appropriateness")
                                if neologism:
                                    doc.add_heading("Neologism Appropriateness", level=3)
                                    doc.add_paragraph(neologism)
                                
                                # Add readability score
                                readability = brand_analysis.get("overall_readability_score")
                                if readability:
                                    doc.add_heading("Overall Readability Score", level=3)
                                    doc.add_paragraph(readability)
                                
                                # Add notes
                                notes = brand_analysis.get("notes")
                                if notes:
                                    doc.add_heading("Notes", level=3)
                                    doc.add_paragraph(notes)
                        
                        # Add comparative insights
                        comparative = json_content.get("comparative_insights")
                        if comparative:
                            doc.add_heading("Comparative Linguistic Insights", level=2)
                            doc.add_paragraph(comparative)
                        
                        # Add summary
                        summary = json_content.get("summary")
                        if summary:
                            doc.add_heading("Summary", level=2)
                            doc.add_paragraph(summary)
                        
                        return
                except Exception as e:
                    logger.error(f"Error formatting linguistic analysis with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Add introduction
            doc.add_paragraph(
                "This section presents linguistic analysis findings based on "
                "the analysis of brand names and their associated linguistic features."
            )
            
            # Check if we have the linguistic_analysis structure from Pydantic model
            if "linguistic_analysis" in data and isinstance(data["linguistic_analysis"], dict):
                section_data = data["linguistic_analysis"]
                
                for brand_name, details in section_data.items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Add linguistic features
                    doc.add_heading("Linguistic Features", level=3)
                    
                    # Process each linguistic aspect
                    for key, value in details.items():
                        if key != "notes" and isinstance(value, str) and value:
                            p = doc.add_paragraph(style='List Bullet')
                            p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                            p.add_run(value)
                    
                    # Add notes at the end if available
                    if "notes" in details and details["notes"]:
                        doc.add_heading("Additional Notes", level=3)
                        doc.add_paragraph(details["notes"])
            # Process linguistic analyses (old format)
            elif "linguistic_analyses" in data and isinstance(data["linguistic_analyses"], list):
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
            # Add section title
            doc.add_heading("Cultural Sensitivity Analysis", level=1)
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        "cultural_sensitivity_analysis": json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                        "format_instructions": self._get_format_instructions("cultural_sensitivity_analysis")
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template("cultural_sensitivity_analysis", format_data, "cultural_sensitivity_analysis")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "cultural_sensitivity_analysis")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "cultural_sensitivity_analysis")
                    
                    if json_content:
                        # Format the document using the enhanced LLM output
                        # Add introduction
                        if "introduction" in json_content and json_content["introduction"]:
                            doc.add_paragraph(json_content["introduction"])
                        
                        # Process each brand analysis
                        if "brand_analyses" in json_content and isinstance(json_content["brand_analyses"], list):
                            for brand_analysis in json_content["brand_analyses"]:
                                if "brand_name" in brand_analysis:
                                    # Add brand name heading
                                    doc.add_heading(brand_analysis["brand_name"], level=2)
                                    
                                    # Add cultural connotations
                                    if "cultural_connotations" in brand_analysis:
                                        doc.add_heading("Cultural Connotations", level=3)
                                        doc.add_paragraph(brand_analysis["cultural_connotations"])
                                    
                                    # Add symbolic meanings
                                    if "symbolic_meanings" in brand_analysis:
                                        doc.add_heading("Symbolic Meanings", level=3)
                                        doc.add_paragraph(brand_analysis["symbolic_meanings"])
                                    
                                    # Add alignment with cultural values
                                    if "alignment_with_cultural_values" in brand_analysis:
                                        doc.add_heading("Alignment with Cultural Values", level=3)
                                        doc.add_paragraph(brand_analysis["alignment_with_cultural_values"])
                                    
                                    # Add religious sensitivities
                                    if "religious_sensitivities" in brand_analysis:
                                        doc.add_heading("Religious Sensitivities", level=3)
                                        doc.add_paragraph(brand_analysis["religious_sensitivities"])
                                    
                                    # Add social and political taboos
                                    if "social_political_taboos" in brand_analysis:
                                        doc.add_heading("Social and Political Taboos", level=3)
                                        doc.add_paragraph(brand_analysis["social_political_taboos"])
                                    
                                    # Add age-related connotations
                                    if "age_related_connotations" in brand_analysis:
                                        doc.add_heading("Age-Related Connotations", level=3)
                                        doc.add_paragraph(brand_analysis["age_related_connotations"])
                                    
                                    # Add regional variations
                                    if "regional_variations" in brand_analysis:
                                        doc.add_heading("Regional Variations", level=3)
                                        doc.add_paragraph(brand_analysis["regional_variations"])
                                    
                                    # Add historical meaning
                                    if "historical_meaning" in brand_analysis:
                                        doc.add_heading("Historical Meaning", level=3)
                                        doc.add_paragraph(brand_analysis["historical_meaning"])
                                    
                                    # Add current event relevance
                                    if "current_event_relevance" in brand_analysis:
                                        doc.add_heading("Current Event Relevance", level=3)
                                        doc.add_paragraph(brand_analysis["current_event_relevance"])
                                    
                                    # Add overall risk rating
                                    if "overall_risk_rating" in brand_analysis:
                                        doc.add_heading("Risk Assessment", level=3)
                                        p = doc.add_paragraph()
                                        p.add_run("Overall Risk Rating: ").bold = True
                                        p.add_run(brand_analysis["overall_risk_rating"])
                                    
                                    # Add notes
                                    if "notes" in brand_analysis:
                                        doc.add_heading("Additional Notes", level=3)
                                        doc.add_paragraph(brand_analysis["notes"])
                        
                        # Add risk mitigation strategies
                        if "risk_mitigation_strategies" in json_content:
                            doc.add_heading("Risk Mitigation Strategies", level=2)
                            doc.add_paragraph(json_content["risk_mitigation_strategies"])
                        
                        # Add summary
                        if "summary" in json_content:
                            doc.add_heading("Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                        
                        return  # Successfully formatted with LLM
                except Exception as e:
                    logger.error(f"Error during LLM formatting for cultural sensitivity analysis: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the cultural sensitivity of each brand name option, "
                "assessing potential cultural, religious, and social implications across global markets."
            )
            
            # Check if data is in the CulturalSensitivityAnalysis model format
            if "cultural_sensitivity_analysis" in data and isinstance(data["cultural_sensitivity_analysis"], dict):
                section_data = data["cultural_sensitivity_analysis"]
                
                for brand_name, details in section_data.items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Add risk assessment
                    if "overall_risk_rating" in details:
                        doc.add_heading("Risk Assessment", level=3)
                        p = doc.add_paragraph()
                        p.add_run("Overall Risk Rating: ").bold = True
                        p.add_run(details["overall_risk_rating"])
                    
                    # Add cultural connotations
                    if "cultural_connotations" in details:
                        doc.add_heading("Cultural Connotations", level=3)
                        doc.add_paragraph(details["cultural_connotations"])
                    
                    # Add symbolic meanings
                    if "symbolic_meanings" in details:
                        doc.add_heading("Symbolic Meanings", level=3)
                        doc.add_paragraph(details["symbolic_meanings"])
                    
                    # Add historical meaning
                    if "historical_meaning" in details:
                        doc.add_heading("Historical Meaning", level=3)
                        doc.add_paragraph(details["historical_meaning"])
                    
                    # Add regional variations
                    if "regional_variations" in details:
                        doc.add_heading("Regional Variations", level=3)
                        doc.add_paragraph(details["regional_variations"])
                    
                    # Add religious sensitivities
                    if "religious_sensitivities" in details:
                        doc.add_heading("Religious Sensitivities", level=3)
                        doc.add_paragraph(details["religious_sensitivities"])
                    
                    # Add social and political taboos
                    if "social_political_taboos" in details:
                        doc.add_heading("Social and Political Taboos", level=3)
                        doc.add_paragraph(details["social_political_taboos"])
                    
                    # Add age-related connotations
                    if "age_related_connotations" in details:
                        doc.add_heading("Age-Related Connotations", level=3)
                        doc.add_paragraph(details["age_related_connotations"])
                    
                    # Add alignment with cultural values
                    if "alignment_with_cultural_values" in details:
                        doc.add_heading("Alignment with Cultural Values", level=3)
                        doc.add_paragraph(details["alignment_with_cultural_values"])
                    
                    # Add current event relevance
                    if "current_event_relevance" in details:
                        doc.add_heading("Current Event Relevance", level=3)
                        doc.add_paragraph(details["current_event_relevance"])
                    
                    # Add notes
                    if "notes" in details:
                        doc.add_heading("Additional Notes", level=3)
                        doc.add_paragraph(details["notes"])
            
            # Process legacy format
            elif "cultural_analyses" in data and isinstance(data["cultural_analyses"], list):
                analyses = data["cultural_analyses"]
                
                for analysis in analyses:
                    if "brand_name" in analysis:
                        doc.add_heading(analysis["brand_name"], level=2)
                        
                        # Map old attributes to new ones if they exist
                        attribute_mapping = {
                            "risk_assessment": "overall_risk_rating",
                            "cultural_values_alignment": "alignment_with_cultural_values",
                            "social_political_considerations": "social_political_taboos",
                            "age_related_factors": "age_related_connotations",
                            "regional_analysis": "regional_variations",
                            "historical_context": "historical_meaning",
                            "current_relevance": "current_event_relevance",
                            "religious_considerations": "religious_sensitivities",
                            "additional_insights": "notes"
                        }
                        
                        # Process attributes based on mapping
                        for old_attr, new_attr in attribute_mapping.items():
                            if old_attr in analysis:
                                heading_title = new_attr.replace("_", " ").title()
                                doc.add_heading(heading_title, level=3)
                                
                                if old_attr == "risk_assessment":
                                    p = doc.add_paragraph()
                                    p.add_run("Overall Risk Rating: ").bold = True
                                    p.add_run(analysis[old_attr])
                                else:
                                    doc.add_paragraph(analysis[old_attr])
                        
                        # Process remaining attributes
                        for key, value in analysis.items():
                            if key != "brand_name" and key not in attribute_mapping.keys() and value:
                                heading_title = key.replace("_", " ").title()
                                doc.add_heading(heading_title, level=3)
                                doc.add_paragraph(value)
            else:
                # Fallback for unstructured data
                doc.add_paragraph("Cultural sensitivity analysis data could not be properly formatted.")
                doc.add_paragraph(str(data))
                
        except Exception as e:
            logger.error(f"Error formatting cultural sensitivity section: {str(e)}")
            doc.add_paragraph(f"Error occurred while formatting cultural sensitivity section: {str(e)}", style='Intense Quote')

    async def _format_name_evaluation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name evaluation section."""
        try:
            # Add section title
            doc.add_heading("Brand Name Evaluation", level=1)
            
            # Format with LLM if available
            if self.llm:
                # Format data for the prompt
                format_data = {
                    "run_id": self.run_id,
                    "brand_name_evaluation": json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                    "format_instructions": self._get_format_instructions("brand_name_evaluation")
                }
                
                # Create prompt
                prompt_content = self._format_template("brand_name_evaluation", format_data, "brand_name_evaluation")
                
                # Create messages
                system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=prompt_content)
                ]
                
                # Invoke LLM
                response = await self._safe_llm_invoke(messages, "brand_name_evaluation")
                
                # Extract JSON content
                json_content = self._extract_json_from_response(response.content, "brand_name_evaluation")
                
                # Format the document with JSON content
                if json_content:
                    # Parse evaluation data
                    if isinstance(json_content, dict):
                        # Add introduction
                        if "introduction" in json_content:
                            doc.add_paragraph(json_content["introduction"])
                        
                        # Add shortlisted summary
                        if "shortlisted_summary" in json_content:
                            doc.add_heading("Shortlisted Names Summary", level=2)
                            doc.add_paragraph(json_content["shortlisted_summary"])
                        
                        # Add individual evaluations
                        if "brand_evaluations" in json_content and isinstance(json_content["brand_evaluations"], list):
                            for brand_analysis in json_content["brand_evaluations"]:
                                if not isinstance(brand_analysis, dict):
                                    continue
                                
                                # Add name
                                brand_name = brand_analysis.get("brand_name", "Unnamed")
                                doc.add_heading(brand_name, level=2)
                                
                                # Add overview table
                                table = doc.add_table(rows=3, cols=2)
                                table.style = 'TableGrid'
                                
                                # Set cell values
                                table.cell(0, 0).text = "Overall Score"
                                table.cell(0, 1).text = str(brand_analysis.get("overall_score", ""))
                                
                                table.cell(1, 0).text = "Shortlist Status"
                                table.cell(1, 1).text = "Shortlisted" if brand_analysis.get("shortlist_status") else "Not Shortlisted"
                                
                                table.cell(2, 0).text = "Recommendation"
                                table.cell(2, 1).text = brand_analysis.get("recommendation", "")
                                
                                # Add evaluation details
                                if "evaluation_details" in brand_analysis:
                                    doc.add_heading("Evaluation Details", level=3)
                                    doc.add_paragraph(brand_analysis["evaluation_details"])
                                
                                # Add strengths and weaknesses
                                if "key_strengths" in brand_analysis:
                                    doc.add_heading("Key Strengths", level=3)
                                    doc.add_paragraph(brand_analysis["key_strengths"])
                                
                                if "potential_weaknesses" in brand_analysis:
                                    doc.add_heading("Potential Weaknesses", level=3)
                                    doc.add_paragraph(brand_analysis["potential_weaknesses"])
                        
                        # Add comparative analysis
                        if "comparative_analysis" in json_content:
                            doc.add_heading("Comparative Analysis", level=2)
                            doc.add_paragraph(json_content["comparative_analysis"])
                        
                        # Add summary
                        if "summary" in json_content:
                            doc.add_heading("Evaluation Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                    else:
                        # Add response as is
                        doc.add_paragraph(response.content)
                else:
                    # Add raw response
                    doc.add_paragraph("The following is the raw evaluation data:")
                    doc.add_paragraph(response.content)
            else:
                # Check if data is in the BrandNameEvaluation model format
                if "brand_name_evaluation" in data and isinstance(data["brand_name_evaluation"], dict):
                    section_data = data["brand_name_evaluation"]
                    
                    # Add introduction
                    doc.add_paragraph(
                        "This section presents the evaluation results for each brand name option, "
                        "including overall scores, shortlist status, and detailed analysis."
                    )
                    
                    # Create a summary of shortlisted names
                    shortlisted_names = []
                    for brand_name, details in section_data.items():
                        if details.get("shortlist_status", False):
                            shortlisted_names.append(brand_name)
                    
                    # Add shortlisted summary
                    if shortlisted_names:
                        doc.add_heading("Shortlisted Names", level=2)
                        shortlist_text = "The following names have been shortlisted: " + ", ".join(shortlisted_names)
                        doc.add_paragraph(shortlist_text)
                    
                    # Process each brand name
                    for brand_name, details in section_data.items():
                        doc.add_heading(brand_name, level=2)
                        
                        # Create evaluation table
                        table = doc.add_table(rows=2, cols=2)
                        table.style = 'TableGrid'
                        
                        # Set cell values
                        table.cell(0, 0).text = "Overall Score"
                        table.cell(0, 1).text = str(details.get("overall_score", "N/A"))
                        
                        table.cell(1, 0).text = "Shortlist Status"
                        table.cell(1, 1).text = "Shortlisted" if details.get("shortlist_status", False) else "Not Shortlisted"
                        
                        # Add evaluation comments
                        if "evaluation_comments" in details:
                            doc.add_heading("Evaluation Comments", level=3)
                            doc.add_paragraph(details["evaluation_comments"])
                else:
                    # Fallback to simple formatting
                    self._format_generic_section_fallback(doc, "brand_name_evaluation", data)
        except Exception as e:
            logger.error(f"Error formatting name evaluation section: {str(e)}")
            doc.add_paragraph(f"Error formatting name evaluation section: {str(e)}")
            # Fallback to simple formatting
            self._format_generic_section_fallback(doc, "brand_name_evaluation", data)

    async def _format_seo_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format SEO analysis section."""
        try:
            # Add section title
            doc.add_heading("SEO and Online Discoverability Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the SEO and online discoverability aspects of the brand name options, "
                "including search volume, keyword potential, and social media considerations that "
                "influence a brand's online visibility and findability."
            )
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        "seo_online_discoverability": json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                        "format_instructions": self._get_format_instructions("seo_analysis")
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template("seo_analysis", format_data, "seo_analysis")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "seo_analysis")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "seo_analysis")
                    
                    if json_content:
                        # Add introduction if provided
                        if "introduction" in json_content and json_content["introduction"]:
                            doc.add_paragraph(json_content["introduction"])
                        
                        # Format each brand analysis
                        if "brand_analyses" in json_content and isinstance(json_content["brand_analyses"], list):
                            for analysis in json_content["brand_analyses"]:
                                if "brand_name" in analysis:
                                    # Add brand name heading
                                    doc.add_heading(analysis["brand_name"], level=2)
                                    
                                    # Add SEO viability score
                                    if "seo_viability_score" in analysis and analysis["seo_viability_score"]:
                                        p = doc.add_paragraph()
                                        p.add_run("SEO Viability Score: ").bold = True
                                        p.add_run(analysis["seo_viability_score"])
                                    
                                    # Add search volume
                                    if "search_volume" in analysis and analysis["search_volume"]:
                                        p = doc.add_paragraph()
                                        p.add_run("Search Volume: ").bold = True
                                        p.add_run(analysis["search_volume"])
                                    
                                    # Add keyword alignment
                                    if "keyword_alignment" in analysis and analysis["keyword_alignment"]:
                                        p = doc.add_paragraph()
                                        p.add_run("Keyword Alignment: ").bold = True
                                        p.add_run(analysis["keyword_alignment"])
                                    
                                    # Add keyword competition
                                    if "keyword_competition" in analysis and analysis["keyword_competition"]:
                                        p = doc.add_paragraph()
                                        p.add_run("Keyword Competition: ").bold = True
                                        p.add_run(analysis["keyword_competition"])
                                    
                                    # Add branded keyword potential
                                    if "branded_keyword_potential" in analysis and analysis["branded_keyword_potential"]:
                                        doc.add_heading("Branded Keyword Potential", level=3)
                                        doc.add_paragraph(analysis["branded_keyword_potential"])
                                    
                                    # Add non-branded keyword potential
                                    if "non_branded_keyword_potential" in analysis and analysis["non_branded_keyword_potential"]:
                                        doc.add_heading("Non-Branded Keyword Potential", level=3)
                                        doc.add_paragraph(analysis["non_branded_keyword_potential"])
                                    
                                    # Add name length searchability
                                    if "name_length_searchability" in analysis and analysis["name_length_searchability"]:
                                        doc.add_heading("Name Length Searchability", level=3)
                                        doc.add_paragraph(analysis["name_length_searchability"])
                                    
                                    # Add unusual spelling impact
                                    if "unusual_spelling_impact" in analysis and analysis["unusual_spelling_impact"]:
                                        doc.add_heading("Unusual Spelling Impact", level=3)
                                        doc.add_paragraph(analysis["unusual_spelling_impact"])
                                    
                                    # Add negative search results
                                    if "negative_search_results" in analysis and analysis["negative_search_results"]:
                                        doc.add_heading("Negative Search Results", level=3)
                                        doc.add_paragraph(analysis["negative_search_results"])
                                    
                                    # Add social media availability
                                    if "social_media_availability" in analysis and analysis["social_media_availability"]:
                                        doc.add_heading("Social Media Availability", level=3)
                                        doc.add_paragraph(analysis["social_media_availability"])
                                    
                                    # Add social media discoverability
                                    if "social_media_discoverability" in analysis and analysis["social_media_discoverability"]:
                                        doc.add_heading("Social Media Discoverability", level=3)
                                        doc.add_paragraph(analysis["social_media_discoverability"])
                                    
                                    # Add competitor domain strength
                                    if "competitor_domain_strength" in analysis and analysis["competitor_domain_strength"]:
                                        doc.add_heading("Competitor Domain Strength", level=3)
                                        doc.add_paragraph(analysis["competitor_domain_strength"])
                                    
                                    # Add exact match search results
                                    if "exact_match_search_results" in analysis and analysis["exact_match_search_results"]:
                                        doc.add_heading("Exact Match Search Results", level=3)
                                        doc.add_paragraph(analysis["exact_match_search_results"])
                                    
                                    # Add negative keyword associations
                                    if "negative_keyword_associations" in analysis and analysis["negative_keyword_associations"]:
                                        doc.add_heading("Negative Keyword Associations", level=3)
                                        doc.add_paragraph(analysis["negative_keyword_associations"])
                                    
                                    # Add content marketing opportunities
                                    if "content_marketing_opportunities" in analysis and analysis["content_marketing_opportunities"]:
                                        doc.add_heading("Content Marketing Opportunities", level=3)
                                        doc.add_paragraph(analysis["content_marketing_opportunities"])
                                    
                                    # Add SEO recommendations
                                    if "seo_recommendations" in analysis and isinstance(analysis["seo_recommendations"], list) and analysis["seo_recommendations"]:
                                        doc.add_heading("SEO Recommendations", level=3)
                                        for recommendation in analysis["seo_recommendations"]:
                                            bullet = doc.add_paragraph(style='List Bullet')
                                            bullet.add_run(recommendation)
                        
                        # Add comparative analysis
                        if "comparative_analysis" in json_content and json_content["comparative_analysis"]:
                            doc.add_heading("Comparative SEO Analysis", level=2)
                            doc.add_paragraph(json_content["comparative_analysis"])
                        
                        # Add summary
                        if "summary" in json_content and json_content["summary"]:
                            doc.add_heading("SEO Analysis Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                        
                        return  # Successfully formatted with LLM
                except Exception as e:
                    logger.error(f"Error formatting SEO analysis with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Check if data is in the SEOOnlineDiscoverability model format
            if "seo_online_discoverability" in data and isinstance(data["seo_online_discoverability"], dict):
                section_data = data["seo_online_discoverability"]
                
                for brand_name, details in section_data.items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Create a table for key metrics
                    table = doc.add_table(rows=4, cols=2)
                    table.style = 'TableGrid'
                    
                    # Set header row
                    header_cells = table.rows[0].cells
                    header_cells[0].text = "Metric"
                    header_cells[1].text = "Value"
                    
                    # Add search volume
                    if "search_volume" in details:
                        row = table.rows[1].cells
                        row[0].text = "Search Volume"
                        row[1].text = str(details["search_volume"])
                    
                    # Add SEO viability score
                    if "seo_viability_score" in details:
                        row = table.rows[2].cells
                        row[0].text = "SEO Viability Score"
                        row[1].text = f"{details['seo_viability_score']}/10"
                    
                    # Add keyword competition
                    if "keyword_competition" in details:
                        row = table.rows[3].cells
                        row[0].text = "Keyword Competition"
                        row[1].text = details["keyword_competition"]
                    
                    # Add spacing
                    doc.add_paragraph()
                    
                    # Add keyword alignment
                    if "keyword_alignment" in details:
                        p = doc.add_paragraph()
                        p.add_run("Keyword Alignment: ").bold = True
                        p.add_run(details["keyword_alignment"])
                    
                    # Add branded keyword potential
                    if "branded_keyword_potential" in details:
                        p = doc.add_paragraph()
                        p.add_run("Branded Keyword Potential: ").bold = True
                        p.add_run(details["branded_keyword_potential"])
                    
                    # Add non-branded keyword potential
                    if "non_branded_keyword_potential" in details:
                        p = doc.add_paragraph()
                        p.add_run("Non-Branded Keyword Potential: ").bold = True
                        p.add_run(details["non_branded_keyword_potential"])
                    
                    # Add name length searchability
                    if "name_length_searchability" in details:
                        p = doc.add_paragraph()
                        p.add_run("Name Length Searchability: ").bold = True
                        p.add_run(details["name_length_searchability"])
                    
                    # Add social media section
                    doc.add_heading("Social Media Factors", level=3)
                    
                    # Add social media availability
                    if "social_media_availability" in details:
                        p = doc.add_paragraph()
                        p.add_run("Social Media Handles Available: ").bold = True
                        p.add_run("Yes" if details["social_media_availability"] else "No")
                    
                    # Add social media discoverability
                    if "social_media_discoverability" in details:
                        p = doc.add_paragraph()
                        p.add_run("Social Media Discoverability: ").bold = True
                        p.add_run(details["social_media_discoverability"])
                    
                    # Add risk factors section
                    doc.add_heading("Risk Factors", level=3)
                    
                    # Add negative search results
                    if "negative_search_results" in details:
                        p = doc.add_paragraph()
                        p.add_run("Negative Search Results Present: ").bold = True
                        p.add_run("Yes" if details["negative_search_results"] else "No")
                    
                    # Add unusual spelling impact
                    if "unusual_spelling_impact" in details:
                        p = doc.add_paragraph()
                        p.add_run("Unusual Spelling Impacts Discoverability: ").bold = True
                        p.add_run("Yes" if details["unusual_spelling_impact"] else "No")
                    
                    # Add negative keyword associations
                    if "negative_keyword_associations" in details:
                        p = doc.add_paragraph()
                        p.add_run("Negative Keyword Associations: ").bold = True
                        p.add_run(details["negative_keyword_associations"])
                    
                    # Add competitor domain strength
                    if "competitor_domain_strength" in details:
                        p = doc.add_paragraph()
                        p.add_run("Competitor Domain Strength: ").bold = True
                        p.add_run(details["competitor_domain_strength"])
                    
                    # Add exact match search results
                    if "exact_match_search_results" in details:
                        p = doc.add_paragraph()
                        p.add_run("Exact Match Search Results: ").bold = True
                        p.add_run(details["exact_match_search_results"])
                    
                    # Add content marketing opportunities
                    if "content_marketing_opportunities" in details:
                        doc.add_heading("Content Marketing Opportunities", level=3)
                        doc.add_paragraph(details["content_marketing_opportunities"])
                    
                    # Add SEO recommendations
                    if "seo_recommendations" in details and isinstance(details["seo_recommendations"], dict) and "recommendations" in details["seo_recommendations"]:
                        doc.add_heading("SEO Recommendations", level=3)
                        for recommendation in details["seo_recommendations"]["recommendations"]:
                            bullet = doc.add_paragraph(style='List Bullet')
                            bullet.add_run(recommendation)
                
                # Add summary section
                doc.add_heading("SEO Analysis Summary", level=2)
                doc.add_paragraph(
                    "This SEO analysis highlights the online visibility potential for each brand name option. "
                    "The findings should inform the final brand name selection, particularly for brands with "
                    "significant digital marketing needs."
                )
            else:
                # Fallback for unstructured data
                doc.add_paragraph("SEO analysis data could not be properly formatted.")
                doc.add_paragraph(str(data))
                
        except Exception as e:
            logger.error(f"Error formatting SEO analysis section: {str(e)}")
            doc.add_paragraph(f"Error formatting SEO analysis section: {str(e)}", style='Intense Quote')

    async def _format_brand_context(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the brand context section."""
        try:
            # Add section title
            doc.add_heading("Brand Context", level=1)
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Extract brand context data if it's nested under 'brand_context' key
                    brand_context_data = data.get('brand_context', data)
                    
                    # Format data for the prompt - use the extracted brand context data
                    format_data = {
                        "run_id": self.current_run_id,
                        "brand_context": json.dumps(brand_context_data, indent=2) if isinstance(brand_context_data, dict) else str(brand_context_data),
                        "format_instructions": self._get_format_instructions("brand_context")
                    }
                    
                    # Log the data being sent to the template
                    logger.debug(f"Brand context data for template: {format_data['brand_context'][:200]}...")
                    
                    # Create prompt
                    prompt_content = self._format_template("brand_context", format_data, "brand_context")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "brand_context")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "brand_context")
                    
                    # Format the document with JSON content
                    if json_content:
                        # Parse brand context data
                        if isinstance(json_content, dict):
                            # Brand Promise
                            if "brand_promise" in json_content:
                                doc.add_heading("Brand Promise", level=2)
                                doc.add_paragraph(json_content["brand_promise"])
                            
                            # Brand Personality
                            if "brand_personality" in json_content:
                                doc.add_heading("Brand Personality", level=2)
                                doc.add_paragraph(json_content["brand_personality"])
                            
                            # Brand Tone of Voice
                            if "brand_tone_of_voice" in json_content:
                                doc.add_heading("Brand Tone of Voice", level=2)
                                doc.add_paragraph(json_content["brand_tone_of_voice"])
                            
                            # Brand Values
                            if "brand_values" in json_content:
                                doc.add_heading("Brand Values", level=2)
                                doc.add_paragraph(json_content["brand_values"])
                            
                            # Brand Purpose
                            if "brand_purpose" in json_content:
                                doc.add_heading("Brand Purpose", level=2)
                                doc.add_paragraph(json_content["brand_purpose"])
                            
                            # Brand Mission
                            if "brand_mission" in json_content:
                                doc.add_heading("Brand Mission", level=2)
                                doc.add_paragraph(json_content["brand_mission"])
                            
                            # Target Audience
                            if "target_audience" in json_content:
                                doc.add_heading("Target Audience", level=2)
                                doc.add_paragraph(json_content["target_audience"])
                            
                            # Customer Needs
                            if "customer_needs" in json_content:
                                doc.add_heading("Customer Needs", level=2)
                                doc.add_paragraph(json_content["customer_needs"])
                            
                            # Market Positioning
                            if "market_positioning" in json_content:
                                doc.add_heading("Market Positioning", level=2)
                                doc.add_paragraph(json_content["market_positioning"])
                            
                            # Competitive Landscape
                            if "competitive_landscape" in json_content:
                                doc.add_heading("Competitive Landscape", level=2)
                                doc.add_paragraph(json_content["competitive_landscape"])
                            
                            # Industry Focus
                            if "industry_focus" in json_content:
                                doc.add_heading("Industry Focus", level=2)
                                doc.add_paragraph(json_content["industry_focus"])
                            
                            # Industry Trends
                            if "industry_trends" in json_content:
                                doc.add_heading("Industry Trends", level=2)
                                doc.add_paragraph(json_content["industry_trends"])
                            
                            # Brand Identity Brief
                            if "brand_identity_brief" in json_content:
                                doc.add_heading("Brand Identity Brief", level=2)
                                doc.add_paragraph(json_content["brand_identity_brief"])
                        else:
                            # Add response as is
                            doc.add_paragraph(response.content)
                    else:
                        # Add raw response
                        doc.add_paragraph("The following is the raw brand context data:")
                        doc.add_paragraph(response.content)
                except Exception as e:
                    logger.error(f"Error during LLM formatting for brand context: {str(e)}")
                    # Fallback to simple formatting
                    self._format_generic_section_fallback(doc, "brand_context", data)
            else:
                # Fallback to simple formatting
                self._format_generic_section_fallback(doc, "brand_context", data)
        except Exception as e:
            logger.error(f"Error formatting brand context section: {str(e)}")
            doc.add_paragraph(f"Error formatting brand context section: {str(e)}")
            # Fallback to simple formatting
            self._format_generic_section_fallback(doc, "brand_context", data)

    async def _format_name_generation(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the name generation section."""
        try:
            # Add section title
            doc.add_heading("Brand Name Generation", level=1)
            
            # Transform the data for the template
            transformed_data = self._transform_name_generation(data)
            logger.debug(f"Transformed name generation data: {len(str(transformed_data))} chars, keys: {list(transformed_data.keys()) if isinstance(transformed_data, dict) else 'not a dict'}")
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Make sure transformed data is in the format expected by the LLM template
                    brand_names = []
                    
                    # Extract brand names for easier reference in the prompt
                    if isinstance(transformed_data, dict) and "categories" in transformed_data:
                        for category in transformed_data.get("categories", []):
                            if isinstance(category, dict) and "names" in category:
                                for name in category.get("names", []):
                                    if isinstance(name, dict) and "brand_name" in name:
                                        brand_names.append(name["brand_name"])
                    
                    # Format data for the prompt
                    # Convert transformed_data to string with proper indentation for template
                    json_transformed_data = json.dumps(transformed_data, indent=2, ensure_ascii=False)
                    
                    # Prepare format data for the template
                    format_data = {
                        "run_id": self.current_run_id,
                        "brand_name_generation": json_transformed_data,
                        "format_instructions": self._get_format_instructions("brand_name_generation"),
                        "brand_names": ", ".join(brand_names[:10]) + (", ..." if len(brand_names) > 10 else "")
                    }
                    
                    # Log the data being sent to the template
                    logger.debug(f"Format data for name generation template has keys: {list(format_data.keys())}")
                    logger.debug(f"Format data includes {len(format_data['brand_name_generation'])} chars of brand name data")
                    logger.debug(f"First 200 chars of brand_name_generation data: {format_data['brand_name_generation'][:200]}")
                    logger.debug(f"Format instructions: {format_data['format_instructions'][:200]}")
                    
                    # Create prompt using our specialized template handling
                    prompt_content = self._format_template("brand_name_generation", format_data, "brand_name_generation")
                    
                    # Verify that variables were correctly substituted
                    placeholder_check = "{{" in prompt_content or "}}" in prompt_content
                    if placeholder_check:
                        logger.warning(f"Template variables may not have been properly substituted in the prompt")
                        # Try direct replacement as a fallback
                        for key, value in format_data.items():
                            placeholder = "{{" + key + "}}"
                            if placeholder in prompt_content:
                                prompt_content = prompt_content.replace(placeholder, str(value))
                                logger.debug(f"Direct fallback replacement for {placeholder}")
                    
                    # Log the prompt content (first 200 chars)
                    logger.debug(f"Prompt content: {prompt_content[:200]}...")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    
                    # Add brand names directly to the system message for context
                    if brand_names:
                        system_content += f"\nBrand names to organize and analyze: {', '.join(brand_names[:10])}" + ("..." if len(brand_names) > 10 else "")
                    
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "brand_name_generation")
                    
                    # Log the first 200 chars of the response
                    logger.debug(f"LLM response: {response.content[:200]}...")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "brand_name_generation")
                    
                    if json_content:
                        logger.debug(f"Extracted JSON content with keys: {list(json_content.keys()) if isinstance(json_content, dict) else 'not a dict'}")
                        
                        # Format the document using the enhanced LLM output
                        # Add introduction
                        if "introduction" in json_content and json_content["introduction"]:
                            doc.add_paragraph(json_content["introduction"])
                        
                        # Add methodology and approach
                        if "methodology_and_approach" in json_content and json_content["methodology_and_approach"]:
                            doc.add_heading("Methodology and Approach", level=2)
                            doc.add_paragraph(json_content["methodology_and_approach"])
                        
                        # Process each category in the specified order
                        if "categories" in json_content and isinstance(json_content["categories"], list):
                            for category in json_content["categories"]:
                                if "category_name" in category:
                                    # Add category heading and description
                                    doc.add_heading(category["category_name"], level=2)
                                    
                                    if "category_description" in category and category["category_description"]:
                                        doc.add_paragraph(category["category_description"])
                                    
                                    # Process each name in this category
                                    if "names" in category and isinstance(category["names"], list):
                                        for name in category["names"]:
                                            if "brand_name" in name:
                                                # Add name as heading
                                                doc.add_heading(name["brand_name"], level=3)
                                                
                                                # Process each name attribute
                                                name_attributes = [
                                                    ("brand_personality_alignment", "Brand Personality Alignment"),
                                                    ("brand_promise_alignment", "Brand Promise Alignment"),
                                                    ("name_generation_methodology", "Methodology"),
                                                    ("memorability_score_details", "Memorability"),
                                                    ("pronounceability_score_details", "Pronounceability"),
                                                    ("visual_branding_potential_details", "Visual Branding Potential"),
                                                    ("target_audience_relevance_details", "Target Audience Relevance"),
                                                    ("market_differentiation_details", "Market Differentiation"),
                                                    ("rationale", "Rationale")  # Added rationale which was missing
                                                ]
                                                
                                                for attr_key, attr_display in name_attributes:
                                                    if attr_key in name and name[attr_key]:
                                                        p = doc.add_paragraph()
                                                        p.add_run(f"{attr_display}: ").bold = True
                                                        p.add_run(str(name[attr_key]))
                        
                        # Add generated names overview
                        if "generated_names_overview" in json_content:
                            doc.add_heading("Generated Names Overview", level=2)
                            
                            # Handle both string and dictionary formats
                            if isinstance(json_content["generated_names_overview"], dict):
                                for key, value in json_content["generated_names_overview"].items():
                                    p = doc.add_paragraph()
                                    p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                                    p.add_run(str(value))
                            else:
                                doc.add_paragraph(str(json_content["generated_names_overview"]))
                        
                        # Add evaluation metrics
                        if "evaluation_metrics" in json_content:
                            doc.add_heading("Initial Evaluation Metrics", level=2)
                            
                            # Handle both string and dictionary formats
                            if isinstance(json_content["evaluation_metrics"], dict):
                                for key, value in json_content["evaluation_metrics"].items():
                                    p = doc.add_paragraph()
                                    p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                                    p.add_run(str(value))
                            else:
                                doc.add_paragraph(str(json_content["evaluation_metrics"]))
                        
                        # Add summary if available
                        if "summary" in json_content and json_content["summary"]:
                            doc.add_heading("Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                        
                        return  # Successfully formatted with LLM
                    else:
                        logger.error("Failed to extract JSON content from LLM response")
                        
                except Exception as e:
                    logger.error(f"Error formatting name generation section with LLM: {str(e)}")
                    logger.debug(f"Error details: {traceback.format_exc()}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            try:
                # Try to parse as a NameGenerationSection
                name_generation_data = None
                
                # Check if the transformed data has the expected structure
                if isinstance(transformed_data, dict):
                    if "categories" in transformed_data and isinstance(transformed_data["categories"], list):
                        # This appears to be a properly structured NameGenerationSection
                        name_generation_data = transformed_data
                
                if name_generation_data:
                    # Add introduction
                    if "introduction" in name_generation_data and name_generation_data["introduction"]:
                        doc.add_paragraph(name_generation_data["introduction"])
                    
                    # Add methodology and approach
                    if "methodology_and_approach" in name_generation_data and name_generation_data["methodology_and_approach"]:
                        doc.add_heading("Methodology and Approach", level=2)
                        doc.add_paragraph(name_generation_data["methodology_and_approach"])
                    
                    # Add generated names overview
                    if "generated_names_overview" in name_generation_data and name_generation_data["generated_names_overview"]:
                        doc.add_heading("Generated Names Overview", level=2)
                        total_count = name_generation_data["generated_names_overview"].get("total_count", 0)
                        doc.add_paragraph(f"A total of {total_count} names were generated across various naming categories.")
                        
                        # Add any other overview information
                        for key, value in name_generation_data["generated_names_overview"].items():
                            if key != "total_count" and value:
                                p = doc.add_paragraph()
                                p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                                p.add_run(str(value))
                    
                    # Add evaluation metrics
                    if "evaluation_metrics" in name_generation_data and name_generation_data["evaluation_metrics"]:
                        doc.add_heading("Initial Evaluation Metrics", level=2)
                        for key, value in name_generation_data["evaluation_metrics"].items():
                            p = doc.add_paragraph()
                            p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                            p.add_run(str(value))
                    
                    # Process each category and its names
                    if "categories" in name_generation_data and isinstance(name_generation_data["categories"], list):
                        for category in name_generation_data["categories"]:
                            # Add category heading and description
                            if "category_name" in category:
                                doc.add_heading(category["category_name"], level=2)
                                
                                if "category_description" in category and category["category_description"]:
                                    doc.add_paragraph(category["category_description"])
                                
                                # Process each name in this category
                                if "names" in category and isinstance(category["names"], list):
                                    for name in category["names"]:
                                        if "brand_name" in name:
                                            # Add name as heading
                                            doc.add_heading(name["brand_name"], level=3)
                                            
                                            # Add all properties in a structured format based on the required sections
                                            sections = [
                                                ("Brand Personality Alignment", name.get("brand_personality_alignment", "")),
                                                ("Brand Promise Alignment", name.get("brand_promise_alignment", "")),
                                                ("Methodology", name.get("name_generation_methodology", "")),
                                                ("Memorability", name.get("memorability_score_details", "")),
                                                ("Pronounceability", name.get("pronounceability_score_details", "")),
                                                ("Visual Branding Potential", name.get("visual_branding_potential_details", "")),
                                                ("Target Audience Relevance", name.get("target_audience_relevance_details", "")),
                                                ("Market Differentiation", name.get("market_differentiation_details", "")),
                                                ("Rationale", name.get("rationale", ""))  # Added rationale
                                            ]
                                            
                                            for section_name, section_content in sections:
                                                if section_content:
                                                    p = doc.add_paragraph()
                                                    p.add_run(f"{section_name}: ").bold = True
                                                    p.add_run(str(section_content))
                    
                    # Add summary if available
                    if "summary" in name_generation_data and name_generation_data["summary"]:
                        doc.add_heading("Summary", level=2)
                        doc.add_paragraph(name_generation_data["summary"])
                else:
                    # Fallback for when we don't have a structured NameGenerationSection
                    doc.add_paragraph("The brand name generation section includes various name options categorized by naming approach.")
                    
                    # Try to extract name data from less structured format
                    if isinstance(transformed_data, dict):
                        for category_name, names in transformed_data.items():
                            if isinstance(names, list):
                                doc.add_heading(category_name, level=2)
                                
                                for name_data in names:
                                    if isinstance(name_data, dict) and "brand_name" in name_data:
                                        doc.add_heading(name_data["brand_name"], level=3)
                                        
                                        # Loop through key data points
                                        for key, value in name_data.items():
                                            if key != "brand_name" and value:
                                                p = doc.add_paragraph()
                                                p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                                                p.add_run(str(value))
            except Exception as e:
                logger.error(f"Error in standard formatting for name generation: {str(e)}")
                logger.debug(f"Error details: {traceback.format_exc()}")
                # Final fallback to generic formatting
                self._format_generic_section_fallback(doc, "brand_name_generation", data)
        except Exception as e:
            logger.error(f"Error formatting name generation section: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            # Fallback to generic formatting as a last resort
            self._format_generic_section_fallback(doc, "brand_name_generation", data)

    async def _format_competitor_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the competitor analysis section."""
        try:
            # Add section title
            doc.add_heading("Competitor Analysis", level=1)
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        "competitor_analysis": json.dumps(data, indent=2),
                        "format_instructions": self._get_format_instructions("competitor_analysis")
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template("competitor_analysis", format_data, "competitor_analysis")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "competitor_analysis")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "competitor_analysis")
                    
                    if json_content:
                        # Format the document using the enhanced LLM output
                        # Add introduction
                        if "introduction" in json_content and json_content["introduction"]:
                            doc.add_paragraph(json_content["introduction"])
                        
                        # Add competitive landscape overview
                        if "competitive_landscape_overview" in json_content and json_content["competitive_landscape_overview"]:
                            doc.add_heading("Competitive Landscape Overview", level=2)
                            doc.add_paragraph(json_content["competitive_landscape_overview"])
                        
                        # Process each brand analysis
                        if "brand_analyses" in json_content and isinstance(json_content["brand_analyses"], list):
                            for brand_analysis in json_content["brand_analyses"]:
                                if "brand_name" in brand_analysis:
                                    # Add brand name heading
                                    doc.add_heading(brand_analysis["brand_name"], level=2)
                                    
                                    # Add competitive context summary
                                    if "competitive_context_summary" in brand_analysis:
                                        doc.add_paragraph(brand_analysis["competitive_context_summary"])
                                    
                                    # Process competitor analyses for this brand
                                    if "competitor_analyses" in brand_analysis and isinstance(brand_analysis["competitor_analyses"], list):
                                        for competitor in brand_analysis["competitor_analyses"]:
                                            if "competitor_name" in competitor:
                                                # Add competitor name heading
                                                doc.add_heading(competitor["competitor_name"], level=3)
                                                
                                                # Add competitor positioning
                                                if "competitor_positioning" in competitor:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Positioning: ").bold = True
                                                    p.add_run(competitor["competitor_positioning"])
                                                
                                                # Add competitor strengths
                                                if "competitor_strengths" in competitor:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Strengths: ").bold = True
                                                    p.add_run(competitor["competitor_strengths"])
                                                
                                                # Add competitor weaknesses
                                                if "competitor_weaknesses" in competitor:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Weaknesses: ").bold = True
                                                    p.add_run(competitor["competitor_weaknesses"])
                                                
                                                # Add risk of confusion
                                                if "risk_of_confusion" in competitor:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Risk of Confusion: ").bold = True
                                                    p.add_run(str(competitor["risk_of_confusion"]))
                                                
                                                # Add target audience perception
                                                if "target_audience_perception" in competitor:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Target Audience Perception: ").bold = True
                                                    p.add_run(competitor["target_audience_perception"])
                                                
                                                # Add trademark conflict risk
                                                if "trademark_conflict_risk" in competitor:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Trademark Conflict Risk: ").bold = True
                                                    p.add_run(competitor["trademark_conflict_risk"])
                                                
                                                # Add competitor differentiation opportunity
                                                if "competitor_differentiation_opportunity" in competitor:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Differentiation Opportunities: ").bold = True
                                                    p.add_run(competitor["competitor_differentiation_opportunity"])
                                    
                                    # Add overall competitive position
                                    if "overall_competitive_position" in brand_analysis:
                                        doc.add_heading("Overall Competitive Position", level=3)
                                        doc.add_paragraph(brand_analysis["overall_competitive_position"])
                                    
                                    # Add differentiation strategy
                                    if "differentiation_strategy" in brand_analysis:
                                        doc.add_heading("Differentiation Strategy", level=3)
                                        doc.add_paragraph(brand_analysis["differentiation_strategy"])
                        
                        # Add comparative competitive analysis
                        if "comparative_competitive_analysis" in json_content and json_content["comparative_competitive_analysis"]:
                            doc.add_heading("Comparative Competitive Analysis", level=2)
                            doc.add_paragraph(json_content["comparative_competitive_analysis"])
                        
                        # Add summary
                        if "summary" in json_content and json_content["summary"]:
                            doc.add_heading("Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                        
                        return  # Successfully formatted with LLM
                except Exception as e:
                    logger.error(f"Error during LLM formatting for competitor analysis: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Add introduction
            doc.add_paragraph(
                "This section analyzes competitors' brand names to provide context and differentiation "
                "strategies for the proposed brand name options."
            )
            
            # Check if data is in the CompetitorAnalysis model format
            if "competitor_analysis" in data and isinstance(data["competitor_analysis"], dict):
                # CompetitorAnalysis model - structured as: brand_name -> competitor_name -> CompetitorDetails
                for brand_name, competitors in data["competitor_analysis"].items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Process each competitor for this brand
                    for competitor_name, details in competitors.items():
                        # Add competitor name heading
                        doc.add_heading(competitor_name, level=3)
                        
                        # Add risk of confusion (integer value)
                        if "risk_of_confusion" in details:
                            p = doc.add_paragraph()
                            p.add_run("Risk of Confusion: ").bold = True
                            risk_value = details["risk_of_confusion"]
                            # Format as a score out of 10
                            risk_text = f"{risk_value}/10"
                            p.add_run(risk_text)
                        
                        # Add competitor positioning
                        if "competitor_positioning" in details:
                            p = doc.add_paragraph()
                            p.add_run("Positioning: ").bold = True
                            p.add_run(details["competitor_positioning"])
                        
                        # Add competitor strengths
                        if "competitor_strengths" in details:
                            doc.add_heading("Strengths", level=4)
                            doc.add_paragraph(details["competitor_strengths"])
                        
                        # Add competitor weaknesses
                        if "competitor_weaknesses" in details:
                            doc.add_heading("Weaknesses", level=4)
                            doc.add_paragraph(details["competitor_weaknesses"])
                        
                        # Add trademark conflict risk
                        if "trademark_conflict_risk" in details:
                            p = doc.add_paragraph()
                            p.add_run("Trademark Conflict Risk: ").bold = True
                            p.add_run(details["trademark_conflict_risk"])
                        
                        # Add target audience perception
                        if "target_audience_perception" in details:
                            p = doc.add_paragraph()
                            p.add_run("Target Audience Perception: ").bold = True
                            p.add_run(details["target_audience_perception"])
                        
                        # Add differentiation opportunities
                        if "competitor_differentiation_opportunity" in details:
                            doc.add_heading("Differentiation Opportunities", level=4)
                            doc.add_paragraph(details["competitor_differentiation_opportunity"])
            
            # Process legacy format 
            elif "competitor_analyses" in data and isinstance(data["competitor_analyses"], list):
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
                        
                        # Map model field names to legacy field names
                        field_mapping = {
                            "competitor_positioning": "positioning",
                            "competitor_strengths": "strengths",
                            "competitor_weaknesses": "weaknesses",
                            "competitor_differentiation_opportunity": "differentiation_opportunities"
                        }
                        
                        # Check both model and legacy field names
                        for model_field, legacy_field in field_mapping.items():
                            value = None
                            if model_field in analysis:
                                value = analysis[model_field]
                            elif legacy_field in analysis:
                                value = analysis[legacy_field]
                            
                            if value:
                                if model_field in ["competitor_strengths", "competitor_weaknesses"]:
                                    # Add as a heading and paragraph or list
                                    field_title = model_field.replace("competitor_", "").replace("_", " ").title()
                                    doc.add_heading(field_title, level=4)
                                    
                                    if isinstance(value, list):
                                        for item in value:
                                            bullet = doc.add_paragraph(style='List Bullet')
                                            bullet.add_run(item)
                                    else:
                                        doc.add_paragraph(value)
                                else:
                                    # Add as a labeled paragraph
                                    p = doc.add_paragraph()
                                    field_title = model_field.replace("competitor_", "").replace("_", " ").title()
                                    p.add_run(f"{field_title}: ").bold = True
                                    p.add_run(str(value))
                        
                        # Add fields that aren't in the mapping
                        for field in ["risk_of_confusion", "target_audience_perception", "trademark_conflict_risk"]:
                            if field in analysis:
                                p = doc.add_paragraph()
                                field_title = field.replace("_", " ").title()
                                p.add_run(f"{field_title}: ").bold = True
                                
                                if field == "risk_of_confusion":
                                    risk_value = analysis[field]
                                    risk_text = f"{risk_value}/10"
                                    p.add_run(risk_text)
                                else:
                                    p.add_run(str(analysis[field]))
                
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
                
                # Extract shortlisted names
                for name, details in eval_data.items():
                    if isinstance(details, dict) and details.get("shortlist_status", False):
                        shortlisted_names.append(name)
            
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
        """Format the market research section."""
        try:
            # Add section title
            doc.add_heading("Market Research", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section presents market research findings related to the brand naming process, "
                "including industry analysis, target audience insights, and market trends."
            )
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        "market_research": json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                        "format_instructions": self._get_format_instructions("market_research")
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template("market_research", format_data, "market_research")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "market_research")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "market_research")
                    
                    if json_content:
                        # Add industry overview
                        if "industry_overview" in json_content and isinstance(json_content["industry_overview"], dict):
                            doc.add_heading("Industry Overview", level=2)
                            overview = json_content["industry_overview"]
                            
                            for key, value in overview.items():
                                if value:
                                    heading_title = key.replace("_", " ").title()
                                    doc.add_heading(heading_title, level=3)
                                    doc.add_paragraph(value)
                        
                        # Process each brand analysis
                        if "brand_analyses" in json_content and isinstance(json_content["brand_analyses"], list):
                            doc.add_heading("Brand-Specific Market Analysis", level=2)
                            
                            for brand_analysis in json_content["brand_analyses"]:
                                if "brand_name" in brand_analysis:
                                    # Add brand name heading
                                    doc.add_heading(brand_analysis["brand_name"], level=3)
                                    
                                    # Industry name
                                    if "industry_name" in brand_analysis and brand_analysis["industry_name"]:
                                        p = doc.add_paragraph()
                                        p.add_run("Industry: ").bold = True
                                        p.add_run(brand_analysis["industry_name"])
                                    
                                    # Market size
                                    if "market_size" in brand_analysis and brand_analysis["market_size"]:
                                        p = doc.add_paragraph()
                                        p.add_run("Market Size: ").bold = True
                                        p.add_run(brand_analysis["market_size"])
                                    
                                    # Market growth rate
                                    if "market_growth_rate" in brand_analysis and brand_analysis["market_growth_rate"]:
                                        p = doc.add_paragraph()
                                        p.add_run("Growth Rate: ").bold = True
                                        p.add_run(brand_analysis["market_growth_rate"])
                                    
                                    # Market opportunity
                                    if "market_opportunity" in brand_analysis and brand_analysis["market_opportunity"]:
                                        doc.add_heading("Market Opportunity", level=4)
                                        doc.add_paragraph(brand_analysis["market_opportunity"])
                                    
                                    # Target audience fit
                                    if "target_audience_fit" in brand_analysis and brand_analysis["target_audience_fit"]:
                                        doc.add_heading("Target Audience Fit", level=4)
                                        doc.add_paragraph(brand_analysis["target_audience_fit"])
                                    
                                    # Competitive analysis
                                    if "competitive_analysis" in brand_analysis and brand_analysis["competitive_analysis"]:
                                        doc.add_heading("Competitive Analysis", level=4)
                                        doc.add_paragraph(brand_analysis["competitive_analysis"])
                                    
                                    # Key competitors
                                    if "key_competitors" in brand_analysis and isinstance(brand_analysis["key_competitors"], list) and brand_analysis["key_competitors"]:
                                        doc.add_heading("Key Competitors", level=4)
                                        for competitor in brand_analysis["key_competitors"]:
                                            bullet = doc.add_paragraph(style='List Bullet')
                                            bullet.add_run(competitor)
                                    
                                    # Market viability
                                    if "market_viability" in brand_analysis and brand_analysis["market_viability"]:
                                        doc.add_heading("Market Viability", level=4)
                                        doc.add_paragraph(brand_analysis["market_viability"])
                                    
                                    # Potential risks
                                    if "potential_risks" in brand_analysis and brand_analysis["potential_risks"]:
                                        doc.add_heading("Potential Risks", level=4)
                                        doc.add_paragraph(brand_analysis["potential_risks"])
                                    
                                    # Customer pain points
                                    if "customer_pain_points" in brand_analysis and isinstance(brand_analysis["customer_pain_points"], list) and brand_analysis["customer_pain_points"]:
                                        doc.add_heading("Customer Pain Points", level=4)
                                        for pain_point in brand_analysis["customer_pain_points"]:
                                            bullet = doc.add_paragraph(style='List Bullet')
                                            bullet.add_run(pain_point)
                                    
                                    # Market entry barriers
                                    if "market_entry_barriers" in brand_analysis and brand_analysis["market_entry_barriers"]:
                                        doc.add_heading("Market Entry Barriers", level=4)
                                        doc.add_paragraph(brand_analysis["market_entry_barriers"])
                                    
                                    # Emerging trends
                                    if "emerging_trends" in brand_analysis and brand_analysis["emerging_trends"]:
                                        doc.add_heading("Emerging Trends", level=4)
                                        doc.add_paragraph(brand_analysis["emerging_trends"])
                                    
                                    # Recommendations
                                    if "recommendations" in brand_analysis and brand_analysis["recommendations"]:
                                        doc.add_heading("Recommendations", level=4)
                                        doc.add_paragraph(brand_analysis["recommendations"])
                        
                        # Add comparative analysis
                        if "comparative_market_analysis" in json_content and json_content["comparative_market_analysis"]:
                            doc.add_heading("Comparative Market Analysis", level=2)
                            doc.add_paragraph(json_content["comparative_market_analysis"])
                        
                        # Add summary
                        if "summary" in json_content and json_content["summary"]:
                            doc.add_heading("Market Research Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                        
                        return  # Successfully formatted with LLM
                except Exception as e:
                    logger.error(f"Error formatting market research with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting (fallback if LLM fails or is not available)
            # Check if data matches the MarketResearch model format
            if "market_research" in data and isinstance(data["market_research"], dict):
                section_data = data["market_research"]
                
                # Add industry overview section combining data from all brands
                doc.add_heading("Industry Overview", level=2)
                
                # Extract and display unique industry information
                industry_names = set()
                market_sizes = set()
                growth_rates = set()
                
                for details in section_data.values():
                    if "industry_name" in details and details["industry_name"]:
                        industry_names.add(details["industry_name"])
                    if "market_size" in details and details["market_size"]:
                        market_sizes.add(details["market_size"])
                    if "market_growth_rate" in details and details["market_growth_rate"]:
                        growth_rates.add(details["market_growth_rate"])
                
                # Display industry information
                if industry_names:
                    doc.add_heading("Industry Context", level=3)
                    doc.add_paragraph(f"Industries analyzed: {', '.join(filter(None, industry_names))}")
                
                if market_sizes:
                    doc.add_heading("Market Size", level=3)
                    for size in market_sizes:
                        doc.add_paragraph(size)
                
                if growth_rates:
                    doc.add_heading("Market Growth", level=3)
                    for rate in growth_rates:
                        doc.add_paragraph(rate)
                
                # Process each brand's market research details
                doc.add_heading("Brand-Specific Market Analysis", level=2)
                
                for brand_name, details in section_data.items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=3)
                    
                    # Add market size
                    if "market_size" in details and details["market_size"]:
                        p = doc.add_paragraph()
                        p.add_run("Market Size: ").bold = True
                        p.add_run(details["market_size"])
                    
                    # Add industry name
                    if "industry_name" in details and details["industry_name"]:
                        p = doc.add_paragraph()
                        p.add_run("Industry: ").bold = True
                        p.add_run(details["industry_name"])
                    
                    # Add market growth rate
                    if "market_growth_rate" in details and details["market_growth_rate"]:
                        p = doc.add_paragraph()
                        p.add_run("Market Growth Rate: ").bold = True
                        p.add_run(details["market_growth_rate"])
                    
                    # Add market opportunity
                    if "market_opportunity" in details and details["market_opportunity"]:
                        doc.add_heading("Market Opportunity", level=4)
                        doc.add_paragraph(details["market_opportunity"])
                    
                    # Add target audience fit
                    if "target_audience_fit" in details and details["target_audience_fit"]:
                        doc.add_heading("Target Audience Fit", level=4)
                        doc.add_paragraph(details["target_audience_fit"])
                    
                    # Add competitive analysis
                    if "competitive_analysis" in details and details["competitive_analysis"]:
                        doc.add_heading("Competitive Analysis", level=4)
                        doc.add_paragraph(details["competitive_analysis"])
                    
                    # Add key competitors as bullet points
                    if "key_competitors" in details and isinstance(details["key_competitors"], list) and details["key_competitors"]:
                        doc.add_heading("Key Competitors", level=4)
                        for competitor in details["key_competitors"]:
                            bullet = doc.add_paragraph(style='List Bullet')
                            bullet.add_run(competitor)
                    
                    # Add market viability
                    if "market_viability" in details and details["market_viability"]:
                        doc.add_heading("Market Viability", level=4)
                        doc.add_paragraph(details["market_viability"])
                    
                    # Add potential risks
                    if "potential_risks" in details and details["potential_risks"]:
                        doc.add_heading("Potential Risks", level=4)
                        doc.add_paragraph(details["potential_risks"])
                    
                    # Add customer pain points as bullet points
                    if "customer_pain_points" in details and isinstance(details["customer_pain_points"], list) and details["customer_pain_points"]:
                        doc.add_heading("Customer Pain Points", level=4)
                        for pain_point in details["customer_pain_points"]:
                            bullet = doc.add_paragraph(style='List Bullet')
                            bullet.add_run(pain_point)
                    
                    # Add market entry barriers
                    if "market_entry_barriers" in details and details["market_entry_barriers"]:
                        doc.add_heading("Market Entry Barriers", level=4)
                        doc.add_paragraph(details["market_entry_barriers"])
                    
                    # Add emerging trends
                    if "emerging_trends" in details and details["emerging_trends"]:
                        doc.add_heading("Emerging Trends", level=4)
                        doc.add_paragraph(details["emerging_trends"])
                    
                    # Add recommendations
                    if "recommendations" in details and details["recommendations"]:
                        doc.add_heading("Recommendations", level=4)
                        doc.add_paragraph(details["recommendations"])
                
                # Add a summary section
                doc.add_heading("Market Research Summary", level=2)
                doc.add_paragraph(
                    "This market research analysis highlights the opportunities and challenges for each brand name "
                    "within the relevant market context. The findings should be considered alongside linguistic, "
                    "cultural, and domain analyses to ensure optimal brand name selection."
                )
            
            # Handle legacy format (if needed)
            elif "market_researches" in data and isinstance(data["market_researches"], list):
                # Legacy format handling code can be preserved here if needed
                researches = data["market_researches"]
                
                doc.add_heading("Market Research Overview", level=2)
                for research in researches:
                    if "overview" in research and research["overview"]:
                        doc.add_paragraph(research["overview"])
                        break
                
                # Map old field names to new model field names
                field_mapping = {
                    "target_audience_insights": "target_audience_fit",
                    "market_trends": "emerging_trends",
                    "implications_for_naming": "recommendations"
                }
                
                for research in researches:
                    # Handle brand-specific analysis if available
                    if "brand_name" in research:
                        doc.add_heading(research["brand_name"], level=3)
                        
                        # Process fields using mapping
                        for old_field, new_field in field_mapping.items():
                            if old_field in research and research[old_field]:
                                heading_title = new_field.replace("_", " ").title()
                                doc.add_heading(heading_title, level=4)
                                doc.add_paragraph(research[old_field])
            else:
                # Fallback for unstructured data
                doc.add_paragraph("Market research data could not be properly formatted.")
                doc.add_paragraph(str(data))
                
        except Exception as e:
            logger.error(f"Error formatting market research: {str(e)}")
            doc.add_paragraph(f"Error formatting market research section: {str(e)}", style='Intense Quote')

    async def _format_semantic_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the semantic analysis section."""
        try:
            # Add section title
            doc.add_heading("Semantic Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section analyzes the semantic aspects of the brand name options, "
                "including etymology, meaning, sound symbolism, and other semantic dimensions "
                "that influence brand perception and memorability."
            )
            
            # Format with LLM if available
            if self.llm:
                try:
                    # Transform data for template
                    transformed_data = self._transform_semantic_analysis(data)
                    
                    # Extract brand names for the prompt
                    brand_names = []
                    if "brand_analyses" in transformed_data and isinstance(transformed_data["brand_analyses"], list):
                        brand_names = [brand["brand_name"] for brand in transformed_data["brand_analyses"] if "brand_name" in brand]
                    
                    # Log the transformed data
                    logger.debug(f"Semantic analysis data for template: {str(transformed_data)[:200]}...")
                    
                    # Get the original semantic analysis data
                    original_semantic_data = data.get("semantic_analysis", data)
                    if isinstance(original_semantic_data, dict) and "semantic_analysis" in original_semantic_data:
                        original_semantic_data = original_semantic_data["semantic_analysis"]
                    
                    # Format data for the prompt - use the original data for the template
                    format_data = {
                        "run_id": self.current_run_id,
                        "semantic_analysis": json.dumps(original_semantic_data, indent=2),
                        "format_instructions": self._get_format_instructions("semantic_analysis"),
                        "brand_names": ", ".join(brand_names) if brand_names else "",
                        "brand_names_instruction": "Please analyze the semantic properties of these brand names."
                    }
                    
                    # Create prompt with template
                    prompt_content = self._format_template("semantic_analysis", format_data, "semantic_analysis")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    
                    # Add brand names directly to the system message as a fallback in case template substitution fails
                    if brand_names:
                        system_content += f"\nAnalyze the following brand names: {', '.join(brand_names)}"
                    
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "semantic_analysis")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "semantic_analysis")
                    
                    if json_content:
                        # Add introduction if provided
                        if "introduction" in json_content and json_content["introduction"]:
                            doc.add_paragraph(json_content["introduction"])
                            
                        # Format each brand analysis
                        if "brand_analyses" in json_content and isinstance(json_content["brand_analyses"], list):
                            for analysis in json_content["brand_analyses"]:
                                if "brand_name" in analysis:
                                    # Add brand name heading
                                    doc.add_heading(analysis["brand_name"], level=2)
                                    
                                    # Add fields based on SemanticAnalysis model
                                    # Etymology
                                    if "etymology" in analysis and analysis["etymology"]:
                                        doc.add_heading("Etymology", level=3)
                                        doc.add_paragraph(analysis["etymology"])
                                    
                                    # Sound symbolism
                                    if "sound_symbolism" in analysis and analysis["sound_symbolism"]:
                                        doc.add_heading("Sound Symbolism", level=3)
                                        doc.add_paragraph(analysis["sound_symbolism"])
                                    
                                    # Brand personality
                                    if "brand_personality" in analysis and analysis["brand_personality"]:
                                        doc.add_heading("Brand Personality", level=3)
                                        doc.add_paragraph(analysis["brand_personality"])
                                    
                                    # Emotional valence
                                    if "emotional_valence" in analysis and analysis["emotional_valence"]:
                                        doc.add_heading("Emotional Valence", level=3)
                                        doc.add_paragraph(analysis["emotional_valence"])
                                    
                                    # Denotative meaning
                                    if "denotative_meaning" in analysis and analysis["denotative_meaning"]:
                                        doc.add_heading("Denotative Meaning", level=3)
                                        doc.add_paragraph(analysis["denotative_meaning"])
                                    
                                    # Figurative language
                                    if "figurative_language" in analysis and analysis["figurative_language"]:
                                        doc.add_heading("Figurative Language", level=3)
                                        doc.add_paragraph(analysis["figurative_language"])
                                    
                                    # Phoneme combinations
                                    if "phoneme_combinations" in analysis and analysis["phoneme_combinations"]:
                                        doc.add_heading("Phoneme Combinations", level=3)
                                        doc.add_paragraph(analysis["phoneme_combinations"])
                                    
                                    # Sensory associations
                                    if "sensory_associations" in analysis and analysis["sensory_associations"]:
                                        doc.add_heading("Sensory Associations", level=3)
                                        doc.add_paragraph(analysis["sensory_associations"])
                                    
                                    # Word length and syllables
                                    if "word_length_syllables" in analysis and analysis["word_length_syllables"]:
                                        doc.add_heading("Word Length and Syllables", level=3)
                                        doc.add_paragraph(analysis["word_length_syllables"])
                                    
                                    # Alliteration and assonance
                                    if "alliteration_assonance" in analysis and analysis["alliteration_assonance"]:
                                        doc.add_heading("Alliteration and Assonance", level=3)
                                        doc.add_paragraph(analysis["alliteration_assonance"])
                                    
                                    # Compounding and derivation
                                    if "compounding_derivation" in analysis and analysis["compounding_derivation"]:
                                        doc.add_heading("Compounding and Derivation", level=3)
                                        doc.add_paragraph(analysis["compounding_derivation"])
                                    
                                    # Semantic trademark risk
                                    if "semantic_trademark_risk" in analysis and analysis["semantic_trademark_risk"]:
                                        doc.add_heading("Semantic Trademark Risk", level=3)
                                        doc.add_paragraph(analysis["semantic_trademark_risk"])
                        
                        # Add comparative insights
                        if "comparative_insights" in json_content and json_content["comparative_insights"]:
                            doc.add_heading("Comparative Semantic Analysis", level=2)
                            doc.add_paragraph(json_content["comparative_insights"])
                        
                        # Add summary
                        if "summary" in json_content and json_content["summary"]:
                            doc.add_heading("Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                        
                        return  # Successfully formatted with LLM
                except Exception as e:
                    logger.error(f"Error formatting semantic analysis with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Check if data is in the SemanticAnalysis model format
            if "semantic_analysis" in data and isinstance(data["semantic_analysis"], dict):
                section_data = data["semantic_analysis"]
                
                for brand_name, details in section_data.items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Add each field from SemanticAnalysis model
                    semantic_fields = [
                        ("etymology", "Etymology"),
                        ("sound_symbolism", "Sound Symbolism"), 
                        ("brand_personality", "Brand Personality"),
                        ("emotional_valence", "Emotional Valence"),
                        ("denotative_meaning", "Denotative Meaning"),
                        ("figurative_language", "Figurative Language"),
                        ("phoneme_combinations", "Phoneme Combinations"),
                        ("sensory_associations", "Sensory Associations"),
                        ("word_length_syllables", "Word Length and Syllables"),
                        ("alliteration_assonance", "Alliteration and Assonance"),
                        ("compounding_derivation", "Compounding and Derivation"),
                        ("semantic_trademark_risk", "Semantic Trademark Risk")
                    ]
                    
                    for field_name, display_name in semantic_fields:
                        if field_name in details and details[field_name]:
                            doc.add_heading(display_name, level=3)
                            
                            # Handle different data types appropriately
                            if isinstance(details[field_name], bool):
                                # For boolean values
                                value = "Yes" if details[field_name] else "No"
                                doc.add_paragraph(f"The name does{'' if details[field_name] else ' not'} use alliteration or assonance.")
                            elif isinstance(details[field_name], int):
                                # For numeric values
                                doc.add_paragraph(f"The name has {details[field_name]} syllables.")
                            elif isinstance(details[field_name], list):
                                # For list values
                                for item in details[field_name]:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(str(item))
                            else:
                                # For string values
                                doc.add_paragraph(str(details[field_name]))
            
            # Process legacy format
            elif "semantic_analyses" in data and isinstance(data["semantic_analyses"], list):
                analyses = data["semantic_analyses"]
                
                # Map old field names to new model field names
                field_mapping = {
                    "meaning": "denotative_meaning",
                    "connotations": "emotional_valence",
                    "semantic_fields": "figurative_language",
                    "phonetic_analysis": "phoneme_combinations",
                    "syllable_count": "word_length_syllables",
                    "trademark_implications": "semantic_trademark_risk"
                }
                
                for analysis in analyses:
                    if "brand_name" in analysis:
                        doc.add_heading(analysis["brand_name"], level=2)
                        
                        # Process each field with mapping
                        for old_field, new_field in field_mapping.items():
                            if old_field in analysis and analysis[old_field]:
                                display_name = new_field.replace("_", " ").title()
                                doc.add_heading(display_name, level=3)
                                
                                if isinstance(analysis[old_field], list):
                                    for item in analysis[old_field]:
                                        bullet = doc.add_paragraph(style='List Bullet')
                                        bullet.add_run(str(item))
                                else:
                                    doc.add_paragraph(str(analysis[old_field]))
                        
                        # Process other fields
                        for key, value in analysis.items():
                            if key not in ["brand_name"] + list(field_mapping.keys()) and value:
                                doc.add_heading(key.replace("_", " ").title(), level=3)
                                
                                if isinstance(value, list):
                                    for item in value:
                                        bullet = doc.add_paragraph(style='List Bullet')
                                        bullet.add_run(str(item))
                                else:
                                    doc.add_paragraph(str(value))
            else:
                # Fallback for unstructured data
                doc.add_paragraph("Semantic analysis data could not be properly formatted.")
                doc.add_paragraph(str(data))
                
        except Exception as e:
            logger.error(f"Error formatting semantic analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting semantic analysis section: {str(e)}", style='Intense Quote')

    async def _format_translation_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the translation analysis section."""
        try:
            # Add section title
            doc.add_heading("Translation Analysis", level=1)
            
            # Add introduction
            doc.add_paragraph(
                "This section examines how the brand name options translate across different languages, "
                "identifying potential issues or advantages in global contexts."
            )
            
            # Use LLM if available
            if self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        "translation_data": json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                        "format_instructions": self._get_format_instructions("translation_analysis")
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template("translation_analysis", format_data, "translation_analysis")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "translation_analysis")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "translation_analysis")
                    
                    if json_content:
                        # Parse and add formatted content from LLM
                        # Add introduction if provided
                        if "introduction" in json_content:
                            doc.add_paragraph(json_content["introduction"])
                            
                        # Format each brand analysis
                        if "per_name_analysis" in json_content and isinstance(json_content["per_name_analysis"], list):
                            for analysis in json_content["per_name_analysis"]:
                                if "brand_name" in analysis:
                                    doc.add_heading(analysis["brand_name"], level=2)
                                    
                                    # Format each language analysis
                                    if "language_analyses" in analysis and isinstance(analysis["language_analyses"], list):
                                        for lang_analysis in analysis["language_analyses"]:
                                            if "language" in lang_analysis:
                                                doc.add_heading(lang_analysis["language"], level=3)
                                                
                                                # Direct translation 
                                                if "direct_translation" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Direct Translation: ").bold = True
                                                    p.add_run(lang_analysis["direct_translation"])
                                                
                                                # Semantic shift
                                                if "semantic_shift" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Semantic Shift: ").bold = True
                                                    p.add_run(lang_analysis["semantic_shift"])
                                                
                                                # Adaptation needed
                                                if "adaptation_needed" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Adaptation Needed: ").bold = True
                                                    p.add_run("Yes" if lang_analysis["adaptation_needed"] else "No")
                                                
                                                # Proposed adaptation
                                                if "proposed_adaptation" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Proposed Adaptation: ").bold = True
                                                    p.add_run(lang_analysis["proposed_adaptation"])
                                                
                                                # Phonetic retention
                                                if "phonetic_retention" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Phonetic Retention: ").bold = True
                                                    p.add_run(lang_analysis["phonetic_retention"])
                                                
                                                # Cultural acceptability
                                                if "cultural_acceptability" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Cultural Acceptability: ").bold = True
                                                    p.add_run(lang_analysis["cultural_acceptability"])
                                                
                                                # Brand essence preserved
                                                if "brand_essence_preserved" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Brand Essence Preserved: ").bold = True
                                                    p.add_run(lang_analysis["brand_essence_preserved"])
                                                
                                                # Pronunciation difficulty
                                                if "pronunciation_difficulty" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Pronunciation Difficulty: ").bold = True
                                                    p.add_run(lang_analysis["pronunciation_difficulty"])
                                                
                                                # Global consistency vs localization
                                                if "global_consistency_vs_localization" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Global Consistency vs Localization: ").bold = True
                                                    p.add_run(lang_analysis["global_consistency_vs_localization"])
                                                
                                                # Notes
                                                if "notes" in lang_analysis:
                                                    p = doc.add_paragraph()
                                                    p.add_run("Notes: ").bold = True
                                                    p.add_run(lang_analysis["notes"])
                                    
                                    # Add translation summary if provided
                                    if "translation_summary" in analysis:
                                        doc.add_heading("Translation Summary", level=3)
                                        doc.add_paragraph(analysis["translation_summary"])
                        
                        # Add global recommendations if available
                        if "global_recommendations" in json_content:
                            doc.add_heading("Global Recommendations", level=2)
                            doc.add_paragraph(json_content["global_recommendations"])
                        
                        # Add internationalization strategy if available
                        if "internationalization_strategy" in json_content:
                            doc.add_heading("Internationalization Strategy", level=2)
                            doc.add_paragraph(json_content["internationalization_strategy"])
                            
                        return
                except Exception as e:
                    logger.error(f"Error formatting translation analysis with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Check for model format based on TranslationAnalysis
            if "translation_analysis" in data and isinstance(data["translation_analysis"], dict):
                translation_data = data["translation_analysis"]
                
                for brand_name, languages in translation_data.items():
                    doc.add_heading(brand_name, level=2)
                    
                    for language, details in languages.items():
                        doc.add_heading(language, level=3)
                        
                        # Create a table for key translation details
                        table = doc.add_table(rows=4, cols=2)
                        table.style = 'TableGrid'
                        
                        # Set headers
                        header_cells = table.rows[0].cells
                        header_cells[0].text = "Field"
                        header_cells[1].text = "Value"
                        
                        # Add direct translation
                        if "direct_translation" in details:
                            row = table.rows[1].cells
                            row[0].text = "Direct Translation"
                            row[1].text = details["direct_translation"]
                        
                        # Add adaptation needed
                        if "adaptation_needed" in details:
                            row = table.rows[2].cells
                            row[0].text = "Adaptation Needed"
                            row[1].text = "Yes" if details["adaptation_needed"] else "No"
                        
                        # Add proposed adaptation
                        if "proposed_adaptation" in details:
                            row = table.rows[3].cells
                            row[0].text = "Proposed Adaptation"
                            row[1].text = details["proposed_adaptation"]
                        
                        # Add spacing
                        doc.add_paragraph()
                        
                        # Add remaining details
                        for key, value in details.items():
                            if key not in ["direct_translation", "adaptation_needed", "proposed_adaptation"] and value:
                                p = doc.add_paragraph()
                                p.add_run(f"{key.replace('_', ' ').title()}: ").bold = True
                                p.add_run(str(value))
                    
                    # Add summary for this brand name
                    doc.add_heading(f"Summary for {brand_name}", level=3)
                    doc.add_paragraph(
                        f"The translation analysis for {brand_name} shows how this name performs "
                        f"across different languages and cultural contexts. Review the details above "
                        f"to understand potential adaptations needed for global markets."
                    )
            else:
                # Fallback for unstructured data
                doc.add_paragraph("Translation analysis data could not be properly formatted.")
                doc.add_paragraph(str(data))
                        
        except Exception as e:
            logger.error(f"Error formatting translation analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting translation analysis section: {str(e)}", style='Intense Quote')

    async def _format_domain_analysis(self, doc: Document, data: Dict[str, Any]) -> None:
        """Format the domain analysis section."""
        try:
            # Add introduction
            doc.add_paragraph(
                "This section evaluates domain availability and digital presence potential for each brand name option, "
                "providing insights into online viability and strategy."
            )
            
            # Use LLM if available
            if self.llm:
                try:
                    # Format data for the prompt
                    format_data = {
                        "run_id": self.current_run_id,
                        "domain_analysis": json.dumps(data, indent=2) if isinstance(data, dict) else str(data),
                        "format_instructions": self._get_format_instructions("domain_analysis")
                    }
                    
                    # Create prompt
                    prompt_content = self._format_template("domain_analysis", format_data, "domain_analysis")
                    
                    # Create messages
                    system_content = self._get_system_content("You are an expert report formatter helping to create a professional brand naming report.")
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt_content)
                    ]
                    
                    # Invoke LLM
                    response = await self._safe_llm_invoke(messages, "domain_analysis")
                    
                    # Extract JSON content
                    json_content = self._extract_json_from_response(response.content, "domain_analysis")
                    
                    if json_content:
                        # Parse and add formatted content from LLM
                        # Add introduction if provided
                        if "introduction" in json_content:
                            doc.add_paragraph(json_content["introduction"])
                        
                        # Add methodology if provided
                        if "methodology" in json_content:
                            doc.add_heading("Methodology", level=2)
                            doc.add_paragraph(json_content["methodology"])
                            
                        # Format each domain analysis
                        if "domain_analyses" in json_content and isinstance(json_content["domain_analyses"], list):
                            for brand_analysis in json_content["domain_analyses"]:
                                if "brand_name" in brand_analysis:
                                    doc.add_heading(brand_analysis["brand_name"], level=2)
                                    
                                    # Add domain availability details
                                    if "domain_availability" in brand_analysis and isinstance(brand_analysis["domain_availability"], dict):
                                        doc.add_heading("Domain Availability", level=3)
                                        availability = brand_analysis["domain_availability"]
                                        
                                        if "domain_exact_match" in availability:
                                            p = doc.add_paragraph()
                                            p.add_run("Exact Match Domain: ").bold = True
                                            p.add_run(availability["domain_exact_match"])
                                        
                                        if "hyphens_numbers_present" in availability:
                                            p = doc.add_paragraph()
                                            p.add_run("Hyphens/Numbers Present: ").bold = True
                                            p.add_run(availability["hyphens_numbers_present"])
                                        
                                        if "acquisition_cost" in availability:
                                            p = doc.add_paragraph()
                                            p.add_run("Acquisition Cost: ").bold = True
                                            p.add_run(availability["acquisition_cost"])
                                    
                                    # Add alternative TLDs
                                    if "alternative_tlds" in brand_analysis:
                                        doc.add_heading("Alternative TLDs", level=3)
                                        doc.add_paragraph(brand_analysis["alternative_tlds"])
                                    
                                    # Add brand name presentation
                                    if "brand_name_presentation" in brand_analysis and isinstance(brand_analysis["brand_name_presentation"], dict):
                                        doc.add_heading("Brand Name Presentation Online", level=3)
                                        presentation = brand_analysis["brand_name_presentation"]
                                        
                                        if "brand_name_clarity_in_url" in presentation:
                                            p = doc.add_paragraph()
                                            p.add_run("Brand Name Clarity in URL: ").bold = True
                                            p.add_run(presentation["brand_name_clarity_in_url"])
                                        
                                        if "domain_length_readability" in presentation:
                                            p = doc.add_paragraph()
                                            p.add_run("Domain Length & Readability: ").bold = True
                                            p.add_run(presentation["domain_length_readability"])
                                    
                                    # Add social media availability
                                    if "social_media_availability" in brand_analysis:
                                        doc.add_heading("Social Media Availability", level=3)
                                        doc.add_paragraph(brand_analysis["social_media_availability"])
                                    
                                    # Add future considerations
                                    if "future_considerations" in brand_analysis and isinstance(brand_analysis["future_considerations"], dict):
                                        doc.add_heading("Future Considerations", level=3)
                                        future = brand_analysis["future_considerations"]
                                        
                                        if "scalability_future_proofing" in future:
                                            p = doc.add_paragraph()
                                            p.add_run("Scalability & Future-Proofing: ").bold = True
                                            p.add_run(future["scalability_future_proofing"])
                                        
                                        if "misspellings_variations_available" in future:
                                            p = doc.add_paragraph()
                                            p.add_run("Misspellings & Variations Available: ").bold = True
                                            p.add_run(future["misspellings_variations_available"])
                                    
                                    # Add notes
                                    if "notes" in brand_analysis:
                                        doc.add_heading("Additional Notes", level=3)
                                        doc.add_paragraph(brand_analysis["notes"])
                                    
                                    # Add recommendations
                                    if "recommendations" in brand_analysis:
                                        doc.add_heading("Recommendations", level=3)
                                        doc.add_paragraph(brand_analysis["recommendations"])
                        
                        # Add comparative analysis if available
                        if "comparative_analysis" in json_content:
                            doc.add_heading("Comparative Analysis", level=2)
                            doc.add_paragraph(json_content["comparative_analysis"])
                        
                        # Add summary if available
                        if "summary" in json_content:
                            doc.add_heading("Summary", level=2)
                            doc.add_paragraph(json_content["summary"])
                            
                        return
                except Exception as e:
                    logger.error(f"Error formatting domain analysis with LLM: {str(e)}")
                    # Fall back to standard formatting
            
            # Standard formatting if LLM not available or if LLM formatting failed
            # Add a general introduction
            doc.add_paragraph(
                "This section evaluates domain availability and digital presence potential for each brand name option, "
                "providing insights into online viability and strategy."
            )
            
            # Check if data is in the DomainAnalysis model format
            if "domain_analysis" in data and isinstance(data["domain_analysis"], dict):
                section_data = data["domain_analysis"]
                
                for brand_name, details in section_data.items():
                    # Add brand name heading
                    doc.add_heading(brand_name, level=2)
                    
                    # Add domain availability section
                    doc.add_heading("Domain Availability", level=3)
                    
                    # Handle domain_exact_match (boolean)
                    if "domain_exact_match" in details:
                        p = doc.add_paragraph()
                        p.add_run("Exact Match Domain Available: ").bold = True
                        exact_match = "Yes" if details["domain_exact_match"] else "No"
                        p.add_run(exact_match)
                    
                    # Handle hyphens_numbers_present (boolean)
                    if "hyphens_numbers_present" in details:
                        p = doc.add_paragraph()
                        p.add_run("Hyphens/Numbers Present: ").bold = True
                        hyphens_numbers = "Yes" if details["hyphens_numbers_present"] else "No"
                        p.add_run(hyphens_numbers)
                    
                    # Handle acquisition_cost (string)
                    if "acquisition_cost" in details:
                        p = doc.add_paragraph()
                        p.add_run("Acquisition Cost: ").bold = True
                        p.add_run(details["acquisition_cost"])
                    
                    # Handle alternative_tlds (list of strings)
                    if "alternative_tlds" in details and details["alternative_tlds"]:
                        doc.add_heading("Alternative TLDs", level=3)
                        if isinstance(details["alternative_tlds"], list):
                            for tld in details["alternative_tlds"]:
                                bullet = doc.add_paragraph(style='List Bullet')
                                bullet.add_run(tld)
                        else:
                            doc.add_paragraph(str(details["alternative_tlds"]))
                    
                    # Handle brand name online presentation
                    doc.add_heading("Brand Name Presentation Online", level=3)
                    
                    # Handle brand_name_clarity_in_url (string)
                    if "brand_name_clarity_in_url" in details:
                        p = doc.add_paragraph()
                        p.add_run("Brand Name Clarity in URL: ").bold = True
                        p.add_run(details["brand_name_clarity_in_url"])
                    
                    # Handle domain_length_readability (string)
                    if "domain_length_readability" in details:
                        p = doc.add_paragraph()
                        p.add_run("Domain Length & Readability: ").bold = True
                        p.add_run(details["domain_length_readability"])
                    
                    # Handle social_media_availability (list of strings)
                    if "social_media_availability" in details and details["social_media_availability"]:
                        doc.add_heading("Social Media Availability", level=3)
                        if isinstance(details["social_media_availability"], list):
                            for platform in details["social_media_availability"]:
                                bullet = doc.add_paragraph(style='List Bullet')
                                bullet.add_run(platform)
                        else:
                            doc.add_paragraph(str(details["social_media_availability"]))
                    
                    # Handle future considerations
                    doc.add_heading("Future Considerations", level=3)
                    
                    # Handle scalability_future_proofing (string)
                    if "scalability_future_proofing" in details:
                        p = doc.add_paragraph()
                        p.add_run("Scalability & Future-Proofing: ").bold = True
                        p.add_run(details["scalability_future_proofing"])
                    
                    # Handle misspellings_variations_available (boolean)
                    if "misspellings_variations_available" in details:
                        p = doc.add_paragraph()
                        p.add_run("Misspellings & Variations Available: ").bold = True
                        misspellings = "Yes" if details["misspellings_variations_available"] else "No"
                        p.add_run(misspellings)
                    
                    # Handle notes (string)
                    if "notes" in details:
                        doc.add_heading("Additional Notes", level=3)
                        doc.add_paragraph(details["notes"])
            else:
                # Fallback for legacy or unstructured data
                for key, value in data.items():
                    if isinstance(value, str) and value:
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        doc.add_paragraph(value)
                    elif isinstance(value, dict):
                        doc.add_heading(key.replace("_", " ").title(), level=2)
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, str) and sub_value:
                                p = doc.add_paragraph()
                                p.add_run(f"{sub_key.replace('_', ' ').title()}: ").bold = True
                                p.add_run(sub_value)
                            elif isinstance(sub_value, (list, tuple)) and sub_value:
                                doc.add_heading(sub_key.replace("_", " ").title(), level=3)
                                for item in sub_value:
                                    bullet = doc.add_paragraph(style='List Bullet')
                                    bullet.add_run(str(item))
        except Exception as e:
            logger.error(f"Error formatting domain analysis: {str(e)}")
            doc.add_paragraph(f"Error formatting domain analysis section: {str(e)}", style='Intense Quote')

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
            self._initialize_llm()
        
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