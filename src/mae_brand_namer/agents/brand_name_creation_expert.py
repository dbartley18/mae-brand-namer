"""Brand Name Creation Expert for generating strategic brand name candidates."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import re
import asyncio

from langchain.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from postgrest.exceptions import APIError
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.settings import settings

logger = get_logger(__name__)

class BrandNameCreationExpert:
    """Expert in strategic brand name generation following Alina Wheeler's methodology."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the BrandNameCreationExpert with necessary configurations."""
        # Initialize Supabase client
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "brand_name_generator"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.generation_prompt = load_prompt(str(prompt_dir / "generation.yaml"))
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.9,  # Higher temperature for more creative name generation
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="brand_name", description="The generated brand name candidate"),
            ResponseSchema(name="naming_category", description="The category or type of name (e.g., descriptive, abstract, evocative)"),
            ResponseSchema(name="brand_personality_alignment", description="How the name aligns with the defined brand personality"),
            ResponseSchema(name="brand_promise_alignment", description="The degree to which the name reflects the brand's promise and value proposition"),
            
            # Score and text fields
            ResponseSchema(name="target_audience_relevance", description="Score from 1-10 indicating how well the name resonates with the target audience"),
            ResponseSchema(name="market_differentiation", description="Score from 1-10 indicating how distinctive the name is in the market"),
            ResponseSchema(name="visual_branding_potential", description="Score from 1-10 indicating the name's potential for visual branding elements"),
            ResponseSchema(name="memorability_score", description="Score from 1-10 indicating how easily the name can be remembered", type="number"),
            ResponseSchema(name="pronounceability_score", description="Score from 1-10 indicating how easily the name can be pronounced", type="number"),
            
            # Details fields (text explanations)
            ResponseSchema(name="target_audience_relevance_details", description="2-3 bullet points explaining the target audience relevance"),
            ResponseSchema(name="market_differentiation_details", description="2-3 bullet points explaining the market differentiation"),
            ResponseSchema(name="visual_branding_potential_details", description="2-3 bullet points explaining the visual branding potential"),
            ResponseSchema(name="memorability_score_details", description="2-3 bullet points explaining the memorability score"),
            ResponseSchema(name="pronounceability_score_details", description="2-3 bullet points explaining the pronounceability score"),
            
            ResponseSchema(name="name_generation_methodology", description="The structured approach used to generate and refine the brand name"),
            ResponseSchema(name="rank", description="The ranking score assigned to the name based on strategic fit", type="number")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)

    async def generate_brand_names(
            self,
            run_id: str,
            brand_context: Dict[str, Any],
            brand_values: List[str],
            purpose: str,
            key_attributes: List[str],
            num_names_per_category: int = 3,
            categories: List[str] = None
        ) -> List[Dict[str, Any]]:
        """
        Generate brand name candidates based on the brand context, organized by naming categories.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_context (Dict[str, Any]): Brand context information
            brand_values (List[str]): List of brand values
            purpose (str): Brand purpose
            key_attributes (List[str]): Key brand attributes
            num_names_per_category (int, optional): Number of brand names to generate per category. Defaults to 3.
            categories (List[str], optional): List of naming categories to use. Defaults to all four categories.
            
        Returns:
            List[Dict[str, Any]]: List of generated brand names with their evaluations
        """
        generated_names = []
        timestamp = datetime.now().isoformat()
        
        # Default categories if none provided
        if not categories:
            categories = [
                "Descriptive Names",
                "Suggestive Names",
                "Abstract Names",
                "Experiential Names"
            ]
        
        # Validate required inputs
        if not run_id:
            raise ValueError("Missing required parameter: run_id")
        
        # Ensure brand values and key attributes are lists
        if brand_values and not isinstance(brand_values, list):
            brand_values = [str(brand_values)]
        if key_attributes and not isinstance(key_attributes, list):
            key_attributes = [str(key_attributes)]
            
        # Ensure we have a valid purpose
        if not purpose:
            purpose = "Not specified"
            
        logger.info(
            "Starting brand name generation", 
            extra={
                "run_id": run_id,
                "num_names_per_category": num_names_per_category,
                "categories": categories
            }
        )
        
        try:
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format())
            
            with tracing_v2_enabled():
                # Track all generated names to avoid duplicates across categories
                all_generated_names = []
                # Use a set for faster lookups when checking duplicates
                all_brand_names_set = set()
                
                # Check existing names already in the database for this run_id to avoid duplicates
                try:
                    # Query Supabase for existing brand names with this run_id
                    response = await self.supabase.execute_with_retry(
                        operation="select",
                        table="brand_name_generation",
                        data={"select": "brand_name", "run_id": run_id}
                    )
                    
                    # Add existing names to our duplicate checking set
                    if response and isinstance(response, list):
                        for item in response:
                            if item and "brand_name" in item:
                                normalized_name = self._normalize_name(item["brand_name"])
                                all_brand_names_set.add(normalized_name)
                                
                        logger.info(f"Found {len(response)} existing brand names for run_id: {run_id}")
                except Exception as e:
                    logger.warning(f"Error checking existing brand names: {str(e)}")
                
                # Normalize a name for duplicate checking
                def _normalize_name(name):
                    if not name:
                        return ""
                    # Remove special characters, convert to lowercase, and strip whitespace
                    return re.sub(r'[^\w\s]', '', name.lower()).strip()
                
                # Move the normalize function outside the loop to make it accessible for checking existing names
                self._normalize_name = _normalize_name
                
                # Generate up to twice the requested names to ensure we have enough after filtering duplicates
                max_generation_attempts = num_names_per_category * len(categories) * 2
                generation_attempts = 0
                duplicate_count = 0
                
                # Iterate through each naming category
                for category in categories:
                    logger.info(
                        f"Generating names for category: {category}",
                        extra={"run_id": run_id}
                    )
                    
                    # Generate specified number of names for this category
                    for i in range(num_names_per_category):
                        try:
                            logger.debug(
                                f"Generating brand name {i+1}/{num_names_per_category} for category {category}",
                                extra={"run_id": run_id}
                            )
                            
                            # Format existing names to avoid duplication
                            existing_names = "\n".join([f"- {name['brand_name']}" for name in all_generated_names])
                            if not existing_names:
                                existing_names = "None yet"
                            
                            # Set up generation prompt with category and existing names
                            generation_prompt = self.generation_prompt.format(
                                format_instructions=self.output_parser.get_format_instructions(),
                                brand_context=brand_context,
                                brand_values=brand_values,
                                purpose=purpose,
                                key_attributes=key_attributes,
                                category=category,
                                existing_names=existing_names
                            )
                            
                            # Create human message
                            human_message = HumanMessage(content=generation_prompt)
                            
                            # Try to generate a unique name with retries
                            max_retries = 3
                            retry_count = 0
                            is_duplicate = False
                            
                            while retry_count < max_retries:
                                try:
                                    # Get response from LLM
                                    response = await self.llm.ainvoke([system_message, human_message])
                                    
                                    # Parse the structured output
                                    parsed_output = self.output_parser.parse(response.content)
                                    
                                    # Ensure required fields exist
                                    if "brand_name" not in parsed_output or not parsed_output["brand_name"]:
                                        logger.warning(
                                            "Missing brand_name in LLM response, retrying",
                                            extra={"run_id": run_id}
                                        )
                                        retry_count += 1
                                        continue
                                    
                                    # Check for duplicates
                                    normalized_name = self._normalize_name(parsed_output["brand_name"])
                                    if normalized_name in all_brand_names_set:
                                        logger.warning(
                                            f"Generated duplicate brand name: {parsed_output['brand_name']}, retrying",
                                            extra={"run_id": run_id}
                                        )
                                        retry_count += 1
                                        is_duplicate = True
                                        
                                        # Update prompt with stronger duplicate avoidance instruction
                                        generation_prompt = self.generation_prompt.format(
                                            format_instructions=self.output_parser.get_format_instructions(),
                                            brand_context=brand_context,
                                            brand_values=brand_values,
                                            purpose=purpose,
                                            key_attributes=key_attributes,
                                            category=category,
                                            existing_names=existing_names + f"\n- {parsed_output['brand_name']} (JUST GENERATED - DO NOT REUSE)"
                                        )
                                        human_message = HumanMessage(content=generation_prompt)
                                        continue
                                    
                                    # If we got here, we have a unique name
                                    is_duplicate = False
                                    break
                                    
                                except Exception as e:
                                    logger.warning(
                                        f"Error during brand name generation attempt: {str(e)}, retrying",
                                        extra={"run_id": run_id}
                                    )
                                    retry_count += 1
                            
                            # If we couldn't generate a unique name after retries, skip this iteration
                            if is_duplicate:
                                logger.warning(
                                    f"Failed to generate unique brand name after {max_retries} attempts, skipping",
                                    extra={"run_id": run_id}
                                )
                                continue
                                
                            # Add metadata to output
                            parsed_output.update({
                                "run_id": run_id,
                                "timestamp": timestamp,
                                "category": category  # Add the category to the output
                            })
                            
                            # Store the name data in Supabase
                            await self._store_in_supabase(run_id, parsed_output)
                            
                            # Add to our duplicate checking set
                            all_brand_names_set.add(normalized_name)
                            
                            # For the return data, ensure all numeric fields are properly converted to floats
                            return_data = parsed_output.copy()
                            
                            # Ensure all numeric fields are float type for consistent processing downstream
                            simple_numeric_fields = [
                                "memorability_score",
                                "pronounceability_score",
                                "target_audience_relevance",
                                "market_differentiation",
                                "visual_branding_potential",
                                "rank"
                            ]
                            
                            for field in simple_numeric_fields:
                                try:
                                    if field in return_data:
                                        return_data[field] = float(return_data[field])
                                    else:
                                        return_data[field] = 5.0  # Default value
                                except (ValueError, TypeError):
                                    logger.warning(f"Could not convert {field} to float, using default value")
                                    return_data[field] = 5.0
                            
                            logger.debug(
                                "Generated valid brand name",
                                extra={
                                    "run_id": run_id,
                                    "brand_name": return_data["brand_name"],
                                    "category": category
                                }
                            )
                            
                            # Add to our tracking lists - use the original data for uniqueness checking
                            generated_names.append(return_data)
                            all_generated_names.append(parsed_output)
                            
                            logger.info(
                                f"Generated brand name {i+1}/{num_names_per_category} for category {category}",
                                extra={
                                    "run_id": run_id,
                                    "brand_name": return_data["brand_name"]
                                }
                            )
                        except Exception as e:
                            logger.error(
                                f"Error generating brand name {i+1}/{num_names_per_category} for category {category}",
                                extra={
                                    "run_id": run_id,
                                    "category": category,
                                    "error_type": type(e).__name__,
                                    "error_message": str(e)
                                }
                            )
                            # Continue generating other names even if one fails
                            continue
            
            if not generated_names:
                logger.error(
                    "Failed to generate any brand names",
                    extra={"run_id": run_id}
                )
                raise ValueError("Failed to generate any valid brand names")
                
            logger.info(
                "Brand name generation completed",
                extra={
                    "run_id": run_id,
                    "count": len(generated_names)
                }
            )
            return generated_names
                
        except APIError as e:
            logger.error(
                "Supabase API error in brand name generation",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            raise
                
        except Exception as e:
            logger.error(
                "Error generating brand names",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise ValueError(f"Failed to generate brand names: {str(e)}")

    async def _store_in_supabase(self, run_id: str, name_data: Dict[str, Any]) -> None:
        """
        Store the generated brand name information in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            name_data (Dict[str, Any]): The brand name data to store
            
        Raises:
            PostgrestError: If there's an error with the Supabase query
            APIError: If there's an API-level error with Supabase
            ValueError: If there's an error with data validation or preparation
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Validate required fields
            if not run_id:
                raise ValueError("Missing required field: run_id")
            if not name_data.get("brand_name"):
                raise ValueError("Missing required field: brand_name")
                
            # Prepare data for Supabase
            supabase_data = {
                "run_id": run_id,
                "brand_name": name_data["brand_name"],
            }
            
            # Add optional fields with safe defaults
            supabase_data["naming_category"] = name_data.get("naming_category", name_data.get("category", ""))
            supabase_data["brand_personality_alignment"] = name_data.get("brand_personality_alignment", "")
            supabase_data["brand_promise_alignment"] = name_data.get("brand_promise_alignment", "")
            
            # Handle score fields - these are text fields in the database for target_audience_relevance, 
            # market_differentiation, and visual_branding_potential, but numeric for memorability_score and pronounceability_score
            
            # First, handle text-based score fields
            for field in ["target_audience_relevance", "market_differentiation", "visual_branding_potential"]:
                try:
                    field_value = name_data.get(field)
                    
                    # If not provided, use default
                    if field_value is None:
                        supabase_data[field] = "Not evaluated"
                    else:
                        # Store as text
                        supabase_data[field] = str(field_value)
                except Exception as e:
                    logger.warning(f"Error processing {field}: {str(e)}. Using default value.")
                    supabase_data[field] = "Not evaluated"
            
            # Then, handle numeric score fields
            for field in ["memorability_score", "pronounceability_score"]:
                try:
                    field_value = name_data.get(field)
                    
                    # If not provided, use default
                    if field_value is None:
                        supabase_data[field] = 5.0
                    else:
                        # Convert to float
                        try:
                            supabase_data[field] = float(field_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid {field}: {field_value}, defaulting to 5.0")
                            supabase_data[field] = 5.0
                except Exception as e:
                    logger.warning(f"Error processing {field}: {str(e)}. Using default value.")
                    supabase_data[field] = 5.0
            
            # Handle details fields - these should be stored as strings
            for field in ["target_audience_relevance_details", "market_differentiation_details", 
                         "visual_branding_potential_details", "memorability_score_details", 
                         "pronounceability_score_details"]:
                try:
                    field_value = name_data.get(field)
                    
                    # If not provided, use default
                    if field_value is None or field_value == "":
                        supabase_data[field] = "No details provided"
                    else:
                        # Store the value as is (should already be a string)
                        supabase_data[field] = field_value
                except Exception as e:
                    logger.warning(f"Error processing {field}: {str(e)}. Using default value.")
                    supabase_data[field] = "Error processing details"
            
            # Name generation methodology is stored as plain text
            supabase_data["name_generation_methodology"] = name_data.get("name_generation_methodology", "")
            
            # Handle rank - stored as a float
            try:
                supabase_data["rank"] = float(name_data.get("rank", 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid rank: {name_data.get('rank')}, defaulting to 0")
                supabase_data["rank"] = 0.0
            
            # Handle timestamp - use ISO format which PostgreSQL can properly interpret
            try:
                if "timestamp" in name_data and name_data["timestamp"]:
                    # Try to parse the existing timestamp
                    if isinstance(name_data["timestamp"], str):
                        # Parse the string to a datetime object, then convert back to ISO format
                        timestamp_dt = datetime.fromisoformat(name_data["timestamp"].replace('Z', '+00:00'))
                        supabase_data["timestamp"] = timestamp_dt.isoformat()
                    elif isinstance(name_data["timestamp"], datetime):
                        # Already a datetime object, just convert to ISO
                        supabase_data["timestamp"] = name_data["timestamp"].isoformat()
                    else:
                        # Unknown format, use current time
                        raise ValueError("Invalid timestamp format")
                else:
                    # No timestamp provided, use current time
                    raise ValueError("No timestamp provided")
            except Exception as e:
                # If any error occurs with timestamp parsing, use current time
                logger.warning(f"Error processing timestamp: {str(e)}. Using current time.")
                supabase_data["timestamp"] = datetime.now().isoformat()
            
            # Define known valid fields for the brand_name_generation table - map fields to those in the database
            db_field_mapping = {
                "run_id": "run_id", 
                "brand_name": "brand_name", 
                "naming_category": "naming_category", 
                "category": "naming_category",  # Map category to naming_category
                "brand_personality_alignment": "brand_personality_alignment",
                "brand_promise_alignment": "brand_promise_alignment", 
                "target_audience_relevance": "target_audience_relevance", 
                "target_audience_relevance_details": "target_audience_relevance_details",
                "market_differentiation": "market_differentiation", 
                "market_differentiation_details": "market_differentiation_details",
                "visual_branding_potential": "visual_branding_potential",
                "visual_branding_potential_details": "visual_branding_potential_details", 
                "memorability_score": "memorability_score", 
                "memorability_score_details": "memorability_score_details",
                "pronounceability_score": "pronounceability_score", 
                "pronounceability_score_details": "pronounceability_score_details",
                "name_generation_methodology": "name_generation_methodology", 
                "timestamp": "timestamp", 
                "rank": "rank"
            }
            
            # Build filtered data using the field mapping
            filtered_data = {}
            for input_field, db_field in db_field_mapping.items():
                if input_field in supabase_data:
                    filtered_data[db_field] = supabase_data[input_field]
            
            # Handle special case for market_differentiation and market_differentiation_details
            # If market_differentiation_details is missing but market_differentiation exists, use that
            if "market_differentiation_details" not in filtered_data and "market_differentiation" in filtered_data:
                filtered_data["market_differentiation_details"] = filtered_data["market_differentiation"]
            
            # Special handling for target_audience_relevance field
            if "target_audience_relevance" not in filtered_data and "target_audience_relevance_details" in filtered_data:
                filtered_data["target_audience_relevance"] = filtered_data["target_audience_relevance_details"]
            
            # Visual branding potential field
            if "visual_branding_potential" not in filtered_data and "visual_branding_potential_details" in filtered_data:
                filtered_data["visual_branding_potential"] = filtered_data["visual_branding_potential_details"]
            
            # Log the data we're about to insert (for debugging)
            logger.debug(
                "Inserting brand name data into Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": filtered_data["brand_name"],
                    "table": "brand_name_generation",
                    "data": json.dumps(filtered_data)
                }
            )
            
            # Store in Supabase using the singleton client
            await self.supabase.execute_with_retry(
                operation="insert",
                table="brand_name_generation",
                data=filtered_data
            )
            
            logger.info(
                "Brand name stored in Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": filtered_data["brand_name"]
                }
            )
            
        except APIError as e:
            logger.error(
                "Supabase API error in brand name storage",
                extra={
                    "run_id": run_id,
                    "brand_name": name_data.get("brand_name", "unknown"),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None),
                    "data": json.dumps(name_data)
                }
            )
            raise
        except ValueError as e:
            logger.error(
                "Validation error in brand name storage",
                extra={
                    "run_id": run_id,
                    "brand_name": name_data.get("brand_name", "unknown"),
                    "error_type": "ValueError",
                    "error_message": str(e),
                    "data": json.dumps(name_data)
                }
            )
            raise
        except Exception as e:
            logger.error(
                "Error storing brand name in Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": name_data.get("brand_name", "unknown"),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "data": json.dumps(name_data)
                }
            )
            raise ValueError(f"Failed to store brand name in Supabase: {str(e)}") 