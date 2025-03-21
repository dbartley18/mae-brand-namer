"""JSON parsing utilities for handling LLM-generated JSON with unescaped quotes."""

import json
import re
import logging

logger = logging.getLogger(__name__)

def fix_json_with_unescaped_quotes(json_str):
    """
    Fix JSON string with unescaped quotes within string values.
    
    Args:
        json_str (str): The potentially invalid JSON string
        
    Returns:
        str: Fixed JSON string, or original if no fix was possible
    """
    try:
        # First try parsing directly
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parsing error: {e}")
        
        # Extract JSON from markdown code blocks if present
        if "```json" in json_str:
            json_str = extract_json_from_markdown(json_str)
        
        # Clean up common LLM formatting issues
        json_str = json_str.strip()
        if json_str.startswith("```") and json_str.endswith("```"):
            json_str = json_str[3:-3].strip()
        
        # Try a simple fix for unescaped quotes within string values
        try:
            # Regex to find unescaped double quotes inside strings
            # This regex tries to find patterns where we have:
            # 1. A field with double quotes
            # 2. Followed by colon and opening quote for string value
            # 3. With content containing unescaped double quotes
            pattern = r'("[\w_]+"\s*:\s*"[^"]*)"([^"]*)"([^"]*)'
            
            # First attempt: Replace all unescaped double quotes inside string values with single quotes
            fixed_json = re.sub(pattern, r'\1\'\2\'\3', json_str)
            
            # Try parsing the fixed version
            result = json.loads(fixed_json)
            logger.debug("Successfully fixed JSON with first attempt")
            return fixed_json
        except (json.JSONDecodeError, Exception) as e1:
            logger.debug(f"First fix attempt failed: {e1}")
            
            # Second attempt: More aggressive approach
            try:
                # Find all key-value pairs and process the values
                pattern = r'"([\w_]+)"\s*:\s*"(.+?)"(?=\s*,|\s*})'
                
                def replace_inner_quotes(match):
                    key = match.group(1)
                    value = match.group(2)
                    # Replace all double quotes with single quotes in the value
                    fixed_value = value.replace('"', "'")
                    return f'"{key}":"{fixed_value}"'
                
                fixed_json = re.sub(pattern, replace_inner_quotes, json_str, flags=re.DOTALL)
                
                # Try parsing the fixed version
                result = json.loads(fixed_json)
                logger.debug("Successfully fixed JSON with second attempt")
                return fixed_json
            except (json.JSONDecodeError, Exception) as e2:
                logger.debug(f"Second fix attempt failed: {e2}")
                
                # Last resort: Try a very aggressive approach to at least get some valid JSON
                try:
                    # Remove markdown code blocks
                    if json_str.startswith("```") and json_str.endswith("```"):
                        json_str = json_str[3:-3].strip()
                    
                    # Try to reconstruct a minimal valid JSON object
                    minimal_json = extract_minimal_json(json_str)
                    if minimal_json:
                        return minimal_json
                except Exception as e3:
                    logger.debug(f"Third fix attempt failed: {e3}")
                    
                # If all attempts failed, return the original
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