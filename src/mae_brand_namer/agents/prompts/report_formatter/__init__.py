"""Report formatter prompt templates loader."""

import os
from typing import Dict, Any

from ....utils.template_utils import load_and_process_template, get_template_dir

# Directory where this file is located
TEMPLATE_DIR = get_template_dir(__file__)

def get_executive_summary_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the executive summary prompt template with variables replaced.
    
    Args:
        **kwargs: Variables to substitute in the template
        
    Returns:
        Processed template with variables replaced
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'executive_summary.yaml'),
        kwargs
    )

def get_recommendations_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the recommendations prompt template with variables replaced.
    
    Args:
        **kwargs: Variables to substitute in the template
        
    Returns:
        Processed template with variables replaced
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'recommendations.yaml'),
        kwargs
    )

def get_title_page_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the title page prompt template with variables replaced.
    
    Args:
        **kwargs: Variables to substitute in the template
        
    Returns:
        Processed template with variables replaced
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'title_page.yaml'),
        kwargs
    )

def get_toc_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the table of contents prompt template with variables replaced.
    
    Args:
        **kwargs: Variables to substitute in the template
        
    Returns:
        Processed template with variables replaced
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'table_of_contents.yaml'),
        kwargs
    )

def get_seo_analysis_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the SEO analysis prompt template with variables replaced.
    
    Args:
        **kwargs: Variables to substitute in the template
        
    Returns:
        Processed template with variables replaced
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'seo_analysis.yaml'),
        kwargs
    )

def get_brand_context_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the brand context prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'brand_context.yaml'),
        kwargs
    )

def get_brand_name_generation_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the brand name generation prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'brand_name_generation.yaml'),
        kwargs
    )

def get_semantic_analysis_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the semantic analysis prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'semantic_analysis.yaml'),
        kwargs
    )

def get_linguistic_analysis_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the linguistic analysis prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'linguistic_analysis.yaml'),
        kwargs
    )

def get_cultural_sensitivity_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the cultural sensitivity prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'cultural_sensitivity_analysis.yaml'),
        kwargs
    )

def get_translation_analysis_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the translation analysis prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'translation_analysis.yaml'),
        kwargs
    )

def get_market_research_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the market research prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'market_research.yaml'),
        kwargs
    )

def get_competitor_analysis_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the competitor analysis prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'competitor_analysis.yaml'),
        kwargs
    )

def get_name_evaluation_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the name evaluation prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'brand_name_evaluation.yaml'),
        kwargs
    )

def get_domain_analysis_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the domain analysis prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'domain_analysis.yaml'),
        kwargs
    )

def get_survey_simulation_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the survey simulation prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'survey_simulation.yaml'),
        kwargs
    )

def get_system_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the system prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'system.yaml'),
        kwargs
    )

def get_shortlisted_names_summary_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the shortlisted names summary prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'shortlisted_names_summary.yaml'),
        kwargs
    )

def get_format_section_prompt(**kwargs) -> Dict[str, Any]:
    """
    Get the format section prompt template with variables replaced.
    """
    return load_and_process_template(
        os.path.join(TEMPLATE_DIR, 'format_section.yaml'),
        kwargs
    )

# Export all the functions
__all__ = [
    'get_executive_summary_prompt',
    'get_recommendations_prompt',
    'get_title_page_prompt',
    'get_toc_prompt',
    'get_seo_analysis_prompt',
    'get_brand_context_prompt',
    'get_brand_name_generation_prompt',
    'get_semantic_analysis_prompt',
    'get_linguistic_analysis_prompt',
    'get_cultural_sensitivity_prompt',
    'get_translation_analysis_prompt',
    'get_market_research_prompt',
    'get_competitor_analysis_prompt',
    'get_name_evaluation_prompt',
    'get_domain_analysis_prompt',
    'get_survey_simulation_prompt',
    'get_system_prompt',
    'get_shortlisted_names_summary_prompt',
    'get_format_section_prompt'
] 