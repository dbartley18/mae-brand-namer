"""JSON parsing utilities for handling LLM-generated JSON with unescaped quotes."""

import json
import re
import logging

logger = logging.getLogger(__name__)

def try_parse_json(json_str):
    """Attempt to parse JSON string directly."""
    try:
        json.loads(json_str)
        return True, json_str
    except json.JSONDecodeError:
        return False, json_str

def clean_json_string(json_str):
    """Clean up common LLM formatting issues in JSON strings."""
    json_str = json_str.strip()
    
    # Extract JSON from markdown code blocks if present
    if "```json" in json_str:
        json_str = extract_json_from_markdown(json_str)
    
    # Remove markdown code block markers
    if json_str.startswith("```") and json_str.endswith("```"):
        json_str = json_str[3:-3].strip()
        
    return json_str

def fix_with_regex_replacement(json_str):
    """First attempt to fix JSON with regex replacing unescaped quotes."""
    try:
        pattern = r'("[\w_]+"\s*:\s*"[^"]*)"([^"]*)"([^"]*)'
        fixed_json = re.sub(pattern, r'\1\'\2\'\3', json_str)
        json.loads(fixed_json)
        logger.debug("Successfully fixed JSON with first attempt")
        return True, fixed_json
    except Exception:
        return False, json_str

def fix_with_aggressive_replacement(json_str):
    """Second attempt with more aggressive quote replacement."""
    try:
        pattern = r'"([\w_]+)"\s*:\s*"(.+?)"(?=\s*,|\s*})'
        
        def replace_inner_quotes(match):
            key = match.group(1)
            value = match.group(2)
            fixed_value = value.replace('"', "'")
            return f'"{key}":"{fixed_value}"'
        
        fixed_json = re.sub(pattern, replace_inner_quotes, json_str, flags=re.DOTALL)
        json.loads(fixed_json)
        logger.debug("Successfully fixed JSON with second attempt")
        return True, fixed_json
    except Exception:
        return False, json_str

def fix_json_with_unescaped_quotes(json_str):
    """
    Fix JSON string with unescaped quotes within string values.
    
    Args:
        json_str (str): The potentially invalid JSON string
        
    Returns:
        str: Fixed JSON string, or original if no fix was possible
    """
    # First try parsing directly
    success, result = try_parse_json(json_str)
    if success:
        return result
    
    logger.debug("Initial JSON parsing failed, attempting fixes")
    
    # Clean up the JSON string
    json_str = clean_json_string(json_str)
    
    # Attempt first fix strategy
    success, result = fix_with_regex_replacement(json_str)
    if success:
        return result
    
    logger.debug("First fix attempt failed, trying more aggressive approach")
    
    # Attempt second fix strategy
    success, result = fix_with_aggressive_replacement(json_str)
    if success:
        return result
    
    logger.debug("Second fix attempt failed, trying minimal extraction")
    
    # Last resort: extract minimal valid JSON
    minimal_json = extract_minimal_json(json_str)
    if minimal_json:
        return minimal_json
    
    logger.warning("All JSON fixing attempts failed, returning original")
    return json_str

def extract_json_from_markdown(text):
    """Extract JSON from markdown code blocks."""
    if "```json" in text:
        # Extract content between ```json and ```
        match = re.search(r'```json\n([\s\S]*?)\n```', text)
        if match:
            return match.group(1).strip()
    return text

def extract_minimal_json(json_str):
    """
    Attempt to extract a minimal valid JSON object from a broken JSON string.
    
    This is a last resort when other parsing attempts fail.
    """
    try:
        # Find all key-value pairs using a simple regex
        # This won't work for all JSON but might help in simple cases
        pairs = re.findall(r'"([\w_]+)"\s*:\s*([^,}]+)', json_str)
        
        # Reconstruct a simple JSON object
        result = {}
        for key, value in pairs:
            # Try to clean up the value
            value = value.strip()
            
            # Convert to appropriate type
            if value.lower() == 'true':
                result[key] = True
            elif value.lower() == 'false':
                result[key] = False
            elif value.replace('.', '', 1).isdigit():
                result[key] = float(value) if '.' in value else int(value)
            else:
                # Remove quotes and escape any internal quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                # Replace double quotes with single quotes
                value = value.replace('"', "'")
                result[key] = value
        
        # Only return if we got at least some key-value pairs
        if result:
            return json.dumps(result)
    except Exception as e:
        logger.debug(f"Minimal JSON extraction failed: {e}")
    
    return None 