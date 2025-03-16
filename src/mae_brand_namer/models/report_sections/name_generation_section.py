from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import logging
from .name_category import NameCategory
from .brand_name import BrandName

# Get logger
logger = logging.getLogger(__name__)

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
        logger = logging.getLogger(__name__)
        
        # Log the incoming data structure
        logger.debug(f"Parsing raw JSON with keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        
        try:
            # Get the brand_name_generation data - it could be at the top level or under the brand_name_generation key
            raw_data = None
            
            # Case 1: Data is already in the expected format with categories
            if isinstance(data, dict) and any(isinstance(value, list) for value in data.values()):
                if "brand_name_generation" not in data:
                    logger.debug("Data appears to be already in category format, not under brand_name_generation key")
                    raw_data = data
                
            # Case 2: Data is under brand_name_generation key
            if raw_data is None and isinstance(data, dict) and "brand_name_generation" in data:
                logger.debug("Found brand_name_generation key in data")
                raw_data = data["brand_name_generation"]
            
            # Case 3: Data is directly the expected data
            if raw_data is None:
                logger.debug("Using data as-is")
                raw_data = data
                
            # Handle the case where raw_data might be a JSON string
            if isinstance(raw_data, str):
                try:
                    logger.debug("Attempting to parse raw_data as JSON string")
                    raw_data = json.loads(raw_data)
                    logger.debug("Successfully parsed raw_data as JSON")
                except json.JSONDecodeError:
                    logger.error("Failed to parse raw_data as JSON string")
                    raise ValueError("brand_name_generation contains invalid JSON")
                    
            if not isinstance(raw_data, dict):
                logger.error(f"raw_data should be a dictionary, got {type(raw_data)}")
                raise ValueError(f"brand_name_generation should be a dictionary, got {type(raw_data)}")
            
            # If we have a nested brand_name_generation structure, extract it
            if "brand_name_generation" in raw_data and isinstance(raw_data["brand_name_generation"], dict):
                logger.debug("Found nested brand_name_generation structure, extracting it")
                raw_data = raw_data["brand_name_generation"]
                
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
                "Invented/Coined/Abstract Names": "Created or invented names with no pre-existing meaning, offering uniqueness and trademark protection.",
                "Evocative/Suggestive Names": "Names that suggest attributes or benefits without explicitly describing them."
            }
            
            # Skip these keys as they're not categories
            non_category_keys = ["introduction", "summary", "methodology_and_approach", 
                                "generated_names_overview", "evaluation_metrics"]
            
            # Process each category
            for category_name, names_list in raw_data.items():
                # Skip non-category keys
                if category_name in non_category_keys:
                    continue
                    
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
                        # Use model_validate instead of validate to support pydantic v2
                        brand_name = BrandName.model_validate(name_data)
                        brand_names.append(brand_name)
                        total_names += 1
                    except Exception as e:
                        logger.warning(f"Failed to parse brand name: {str(e)}")
                        
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
                logger.error("No valid categories found in brand_name_generation data")
                raise ValueError("No valid categories found in brand_name_generation data")
                
            # Create the methodology description
            all_category_names = [cat.category_name for cat in categories]
            methodology = (
                "The name generation process employed multiple naming approaches to create a diverse set of brand name options. "
                f"These approaches included {', '.join(all_category_names)}. "
                "Each name was evaluated based on its alignment with the brand's personality and promise, "
                "as well as practical considerations such as memorability, pronounceability, and market differentiation."
            )
            
            # Extract any existing metadata from raw_data if it matches our structure
            introduction = raw_data.get("introduction", 
                "Overview of the brand name generation process, presenting names categorized by naming approach.")
                
            summary = raw_data.get("summary", None)
            methodology_and_approach = raw_data.get("methodology_and_approach", methodology)
            generated_names_overview = raw_data.get("generated_names_overview", {"total_count": total_names})
            evaluation_metrics = raw_data.get("evaluation_metrics", {
                "metrics_summary": "Names were evaluated based on brand alignment, memorability, pronounceability, visual potential, audience relevance, and market differentiation."
            })
            
            # Log success
            logger.info(f"Successfully parsed brand name generation data with {len(categories)} categories and {total_names} total names")
            
            # Create and return the NameGenerationSection
            return cls(
                introduction=introduction,
                categories=categories,
                summary=summary,
                methodology_and_approach=methodology_and_approach,
                generated_names_overview=generated_names_overview,
                evaluation_metrics=evaluation_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in from_raw_json: {str(e)}")
            raise 