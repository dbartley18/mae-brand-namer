"""Report formatter prompt templates loader."""

import os
import yaml
from typing import Dict, Any, List

# Directory where this file is located
TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_yaml_template(filename: str) -> Dict[str, Any]:
    """Load a YAML template from file."""
    template_path = os.path.join(TEMPLATE_DIR, filename)
    with open(template_path, 'r') as f:
        return yaml.safe_load(f)

def get_executive_summary_prompt(**kwargs) -> Dict:
    """Get the executive summary prompt template."""
    return _load_yaml_template('executive_summary.yaml')

def get_recommendations_prompt(**kwargs) -> Dict:
    """Get the recommendations prompt template."""
    return _load_yaml_template('recommendations.yaml')

def get_title_page_prompt(**kwargs) -> Dict:
    """Get the title page prompt template."""
    return _load_yaml_template('title_page.yaml')

def get_toc_prompt(**kwargs) -> Dict:
    """Get the table of contents prompt template."""
    return _load_yaml_template('table_of_contents.yaml')

__all__ = [
    'get_executive_summary_prompt',
    'get_recommendations_prompt',
    'get_title_page_prompt',
    'get_toc_prompt'
] 