from pydantic import BaseModel, Field
from typing import List, Optional
import json
import re
import logging

# Get logger
logger = logging.getLogger(__name__)

class TOCSection(BaseModel):
    """Model for a section in the table of contents."""
    
    title: str = Field(..., description="Section title with number (e.g., '1. Executive Summary')")
    description: str = Field(..., description="Brief description of what the section contains")


class TableOfContentsSection(BaseModel):
    """Model for the table of contents section of the report."""
    
    sections: List[TOCSection] = Field(..., description="List of sections in the report")
    introduction: Optional[str] = Field(None, description="Optional introductory text for the TOC")
    
    @classmethod
    def from_json(cls, content_str: str) -> 'TableOfContentsSection':
        """Parse a TOC section from JSON string."""
        try:
            # Extract JSON if in code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content_str)
            if json_match:
                content_str = json_match.group(1)
                
            # Parse the JSON
            data = json.loads(content_str)
            
            # Create the model
            return cls.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse TableOfContentsSection: {e}")
            raise ValueError(f"Invalid TOC JSON: {e}") 