"""Utilities for processing YAML templates with variable substitution."""

import os
import yaml
import re
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def load_yaml_template(template_path: str) -> Dict[str, Any]:
    """
    Load a YAML template from file.
    
    Args:
        template_path: Path to the YAML template file
        
    Returns:
        Dictionary containing the loaded template
    """
    try:
        with open(template_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading template {template_path}: {str(e)}")
        return {}

def process_template(template: Union[Dict[str, Any], str], variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a template by replacing variables with their values.
    
    Args:
        template: Template dictionary or string to process
        variables: Dictionary of variable values to substitute
        
    Returns:
        Processed template with variables replaced
    """
    try:
        # Convert template to string if it's a dictionary
        if isinstance(template, dict):
            template_str = yaml.dump(template)
        else:
            template_str = str(template)
        
        # Find all variables in the template using regex
        var_pattern = r'\{\{\s*([a-zA-Z0-9_]+)\s*\}\}'
        matches = re.findall(var_pattern, template_str)
        
        # Replace each variable with its value
        for var_name in matches:
            if var_name in variables:
                # Replace {{var_name}} with the actual value
                var_pattern = r'\{\{\s*' + var_name + r'\s*\}\}'
                var_value = str(variables[var_name]).replace('\\', '\\\\').replace('"', '\\"')
                template_str = re.sub(var_pattern, var_value, template_str)
            else:
                logger.warning(f"Variable {var_name} not found in provided variables")
        
        # Convert back to dictionary
        return yaml.safe_load(template_str)
    except Exception as e:
        logger.error(f"Error processing template: {str(e)}")
        logger.debug(f"Template: {template}")
        logger.debug(f"Variables: {variables}")
        # Return original template as fallback
        return template if isinstance(template, dict) else {}

def load_and_process_template(template_path: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load a YAML template and process its variables in one step.
    
    Args:
        template_path: Path to the YAML template file
        variables: Dictionary of variable values to substitute
        
    Returns:
        Processed template with variables replaced
    """
    template = load_yaml_template(template_path)
    return process_template(template, variables)

def get_template_dir(module_file: str) -> str:
    """
    Get the template directory for a given module file.
    
    Args:
        module_file: The __file__ attribute of the module
        
    Returns:
        Absolute path to the template directory
    """
    return os.path.dirname(os.path.abspath(module_file)) 