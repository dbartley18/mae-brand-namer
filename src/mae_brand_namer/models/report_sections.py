"""
Pydantic models for report sections.

These models define the expected structure for each section of the brand naming report.
They are used to validate and parse data from Supabase before formatting into the document.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import json
import re

# Get logger
logger = logging.getLogger(__name__)


class BrandName(BaseModel):
    """Model for an individual brand name in the name generation section."""
    
    # Required fields from the example
    brand_name: str = Field(..., description="The generated brand name")
    naming_category: Optional[str] = Field(None, description="Category of the name")
    brand_personality_alignment: str = Field(..., description="How the name aligns with the brand personality")
    brand_promise_alignment: str = Field(..., description="How the name aligns with the brand promise")
    name_generation_methodology: str = Field(..., description="The methodology used to generate the name")
    memorability_score_details: str = Field(..., description="Assessment of how memorable the name is")
    pronounceability_score_details: str = Field(..., description="Assessment of how pronounceable the name is")
    visual_branding_potential_details: str = Field(..., description="Assessment of the visual branding potential")
    target_audience_relevance_details: str = Field(..., description="How relevant the name is to the target audience")
    market_differentiation_details: str = Field(..., description="How the name differentiates from competitors")
    
    # Optional fields
    rationale: Optional[str] = Field(None, description="Rationale for the name")
    
    class Config:
        # Allow extra fields we don't explicitly model
        extra = "ignore"


class NameCategory(BaseModel):
    """Model for a category of brand names."""
    
    category_name: str = Field(..., description="Name of the naming category (e.g., Symbolic, Descriptive)")
    category_description: Optional[str] = Field(None, description="Description of this naming approach/category")
    names: List[BrandName] = Field(..., description="List of brand names in this category")


class NameGenerationSection(BaseModel):
    """Model for the complete name generation section of the report."""
    
    introduction: Optional[str] = Field(None, description="Overview of the brand name generation process")
    categories: List[NameCategory] = Field(..., description="Categories of generated names")
    summary: Optional[str] = Field(None, description="Optional summary of the name generation outcomes")
    methodology_and_approach: Optional[str] = Field(None, description="Description of the methodology and approach used")
    generated_names_overview: Optional[Dict[str, Any]] = Field(None, description="Overview of generated names, including total count")
    evaluation_metrics: Optional[Dict[str, Any]] = Field(None, description="Initial evaluation metrics for the names")
    
    @classmethod
    def from_raw_json(cls, data: Dict[str, Any]) -> 'NameGenerationSection':
        """
        Parse the brand_name_generation JSON structure that looks like:
        {
            "brand_name_generation": {
                "Symbolic Names": [list of names],
                "Geographic Names": [list of names],
                ...
            }
        }
        """
        # Get the brand_name_generation data
        if "brand_name_generation" not in data:
            raise ValueError("Missing 'brand_name_generation' key in data")
            
        raw_data = data["brand_name_generation"]
        
        # Handle the case where it might be a JSON string
        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except json.JSONDecodeError:
                raise ValueError("brand_name_generation contains invalid JSON")
                
        if not isinstance(raw_data, dict):
            raise ValueError(f"brand_name_generation should be a dictionary, got {type(raw_data)}")
            
        # Build categories
        categories = []
        total_names = 0
        
        # Category descriptions
        category_descriptions = {
            "Symbolic Names": "Names that leverage metaphorical associations and symbolism to convey brand values and attributes.",
            "Geographic Names": "Names based on locations that carry positive associations or relevant geographic connections.",
            "Descriptive Names": "Names that directly describe what the company or product does or offers.",
            "Experiential Names": "Names that evoke a feeling or experience associated with the brand.",
            "Founder/Personal Names": "Names based on real or fictional people, often conveying heritage or personality.",
            "Invented/Coined/Abstract Names": "Created or invented names with no pre-existing meaning, offering uniqueness and trademark protection."
        }
        
        # Process each category
        for category_name, names_list in raw_data.items():
            if not isinstance(names_list, list):
                logger.warning(f"Category {category_name} is not a list, skipping")
                continue
                
            # Process each name in the category
            brand_names = []
            for name_data in names_list:
                if not isinstance(name_data, dict):
                    logger.warning(f"Name in category {category_name} is not a dictionary, skipping")
                    continue
                    
                # Set category name if missing
                if "naming_category" not in name_data:
                    name_data["naming_category"] = category_name
                
                try:
                    brand_name = BrandName.model_validate(name_data)
                    brand_names.append(brand_name)
                    total_names += 1
                except Exception as e:
                    logger.warning(f"Failed to parse brand name: {e}")
                    
            # Create the category if we have names
            if brand_names:
                description = category_descriptions.get(category_name, f"Names using the {category_name} approach.")
                
                category = NameCategory(
                    category_name=category_name,
                    category_description=description,
                    names=brand_names
                )
                categories.append(category)
                
        # Make sure we have at least one category
        if not categories:
            raise ValueError("No valid categories found in brand_name_generation data")
            
        # Create the section
        methodology = (
            "The name generation process employed multiple naming approaches to create a diverse set of brand name options. "
            f"These approaches included {', '.join([cat.category_name for cat in categories])}. "
            "Each name was evaluated based on its alignment with the brand's personality and promise, "
            "as well as practical considerations such as memorability, pronounceability, and market differentiation."
        )
        
        return cls(
            introduction="Overview of the brand name generation process, presenting names categorized by naming approach.",
            categories=categories,
            methodology_and_approach=methodology,
            generated_names_overview={"total_count": total_names},
            evaluation_metrics={
                "metrics_summary": "Names were evaluated based on brand alignment, memorability, pronounceability, visual potential, audience relevance, and market differentiation."
            }
        )


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