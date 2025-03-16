"""Utilities for processing YAML templates with variable substitution."""

import os
import yaml
import re
import logging
import json
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
        
        # Log available variables for debugging
        logger.debug(f"Available variables for template substitution: {list(variables.keys())}")
        
        # Extra debugging for double curly braces
        if '{{' in template_str:
            logger.debug(f"Template contains double curly braces. Example: {template_str.split('{{')[1].split('}}')[0] if '{{' in template_str and '}}' in template_str else 'none found'}")
            
        # Find all variables in the template using regex (handles both single and double curly braces)
        # First, check for double curly braces format ({{var_name}})
        double_var_pattern = r'\{\{\s*([a-zA-Z0-9_]+)\s*\}\}'
        double_matches = re.findall(double_var_pattern, template_str)
        
        # Also check for single curly braces format ({var_name})
        single_var_pattern = r'\{([a-zA-Z0-9_]+)\}'
        single_matches = re.findall(single_var_pattern, template_str)
        
        # Combine both matches (removing duplicates)
        all_matches = list(set(double_matches + single_matches))
        
        logger.debug(f"Found template variables to substitute: {all_matches}")
        
        # Keep track of variables that failed to substitute
        failed_vars = []
        
        # Replace each variable with its value
        for var_name in all_matches:
            if var_name in variables:
                # Get the variable value
                var_value = variables[var_name]
                
                # More detailed debug for important variables
                if var_name in ['run_id', 'brand_name_generation', 'format_instructions']:
                    if isinstance(var_value, str):
                        logger.debug(f"Variable '{var_name}' value type: {type(var_value)}, length: {len(var_value)}, starts with: {var_value[:50]}...")
                    else:
                        logger.debug(f"Variable '{var_name}' value type: {type(var_value)}")
                
                # Handle different variable types appropriately
                if isinstance(var_value, (list, dict)):
                    # Use JSON serialization for lists and dicts to properly format them
                    var_value = json.dumps(var_value)
                else:
                    var_value = str(var_value)
                
                # Escape backslashes and quotes to avoid issues in the template
                var_value = var_value.replace('\\', '\\\\').replace('"', '\\"')
                
                # Replace {{var_name}} with the actual value (double braces format)
                double_var_pattern = r'\{\{\s*' + var_name + r'\s*\}\}'
                old_template_str = template_str
                template_str = re.sub(double_var_pattern, var_value, template_str)
                
                # Check if any replacements were made
                if old_template_str != template_str:
                    logger.debug(f"Substituted variable '{var_name}' in template (double braces)")
                
                # Replace {var_name} with the actual value (single braces format)
                single_var_pattern = r'\{' + var_name + r'\}'
                old_template_str = template_str
                template_str = re.sub(single_var_pattern, var_value, template_str)
                
                # Check if any replacements were made
                if old_template_str != template_str:
                    logger.debug(f"Substituted variable '{var_name}' in template (single braces)")
            else:
                failed_vars.append(var_name)
                logger.warning(f"Variable '{var_name}' not found in provided variables. Available keys: {list(variables.keys())}")
        
        # Check if any variables remain unsubstituted
        remaining_double_vars = re.findall(r'\{\{\s*([a-zA-Z0-9_]+)\s*\}\}', template_str)
        if remaining_double_vars:
            logger.warning(f"Some double-braced variables remain unsubstituted: {remaining_double_vars}")
        
        # Log variables that failed to substitute
        if failed_vars:
            logger.error(f"Variables not substituted: {failed_vars}")
        
        # Log a sample of the processed template
        logger.debug(f"Processed template (first 200 chars): {template_str[:200]}...")
        
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