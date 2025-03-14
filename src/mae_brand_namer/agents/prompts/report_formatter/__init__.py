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

__all__ = [
    'get_executive_summary_prompt',
    'get_recommendations_prompt',
    'get_title_page_prompt',
    'get_toc_prompt',
    'get_seo_analysis_prompt'
] 