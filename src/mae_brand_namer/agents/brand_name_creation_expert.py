"""Brand Name Creation Expert for generating strategic brand name candidates."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import re
import asyncio
import traceback

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
from ..models.app_config import AppConfig

logger = get_logger(__name__)

class BrandNameCreationExpert:
    """Expert in strategic brand name generation following Alina Wheeler's methodology."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None, app_config: AppConfig = None):
        """Initialize the BrandNameCreationExpert with necessary configurations."""
        # Initialize Supabase client
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        # Get agent-specific configuration
        self.app_config = app_config or AppConfig()
        agent_name = "brand_name_creation_expert"
        self.temperature = self.app_config.get_temperature_for_agent(agent_name)
        
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "brand_name_generator"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.generation_prompt = load_prompt(str(prompt_dir / "generation.yaml"))
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
        # Initialize Gemini model with agent-specific temperature
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=self.temperature,
            google_api_key=os.getenv("GEMINI_API_KEY") or settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None
        )
        
        # Log the temperature setting being used
        logger.info(
            f"Initialized Brand Name Creation Expert with temperature: {self.temperature}",
            extra={"agent": agent_name, "temperature": self.temperature}
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="brand_name", description="The generated brand name candidate"),
            ResponseSchema(name="naming_category", description="The category or type of name (e.g., descriptive, abstract, suggestive, experiential)"),
            ResponseSchema(name="brand_personality_alignment", description="How the name aligns with the defined brand personality traits. Provide specific explanation connecting the name to personality attributes."),
            ResponseSchema(name="brand_promise_alignment", description="The degree to which the name reflects the brand's promise and value proposition. Explain the specific connection to the brand promise."),
            ResponseSchema(name="target_audience_relevance", description="Suitability of the name for the intended target audience on a scale of 1-10", type="number"),
            ResponseSchema(name="market_differentiation", description="How well the name stands out from competitors and reinforces brand positioning on a scale of 1-10", type="number"),
            ResponseSchema(name="memorability_score", description="Score from 1-10 indicating how easily the name can be remembered", type="number"),
            ResponseSchema(name="pronounceability_score", description="Score from 1-10 indicating how easily the name can be pronounced", type="number"),
            ResponseSchema(name="visual_branding_potential", description="How well the name lends itself to logos, typography, and digital branding on a scale of 1-10", type="number"),
            ResponseSchema(name="name_generation_methodology", description="The structured approach used to generate and refine the brand name. Include the specific naming strategy and technique applied."),
            ResponseSchema(name="rank", description="The ranking score assigned to the name based on strategic fit on a scale of 1-10", type="number")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)

    async def generate_brand_names(
            self,
            run_id: str,
            brand_context: Dict[str, Any],
            brand_values: List[str],
            purpose: str,
            key_attributes: List[str],
            num_names_per_category: int = 5
        ) -> List[Dict[str, Any]]:
        """
        Generate brand name candidates based on the brand context.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_context (Dict[str, Any]): Brand context information
            brand_values (List[str]): List of brand values
            purpose (str): Brand purpose
            key_attributes (List[str]): Key brand attributes
            num_names_per_category (int, optional): Number of brand names to generate per category. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: List of generated brand names with their evaluations
        """
        generated_names = []
        timestamp = datetime.now().isoformat()
        
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

        # Format brand_context for better LLM processing
        formatted_brand_context = ""
        if brand_context:
            # Extract key elements from brand_context for clearer prompt format
            context_elements = [
                ("Brand Promise", brand_context.get("brand_promise", "")),
                ("Brand Personality", brand_context.get("brand_personality", "")),
                ("Target Audience", brand_context.get("target_audience", "")),
                ("Market Positioning", brand_context.get("market_positioning", "")),
                ("Competitive Landscape", brand_context.get("competitive_landscape", "")),
                ("Industry Focus", brand_context.get("industry_focus", "")),
                ("Brand Mission", brand_context.get("brand_mission", ""))
            ]
            
            # Format each element that has a value
            for label, value in context_elements:
                if value:
                    if isinstance(value, list):
                        formatted_value = ", ".join(value)
                    else:
                        formatted_value = value
                    formatted_brand_context += f"{label}: {formatted_value}\n"
        else:
            formatted_brand_context = "Not provided"
            
        logger.info(
            "Starting brand name generation", 
            extra={
                "run_id": run_id,
                "names_per_category": num_names_per_category,
                "total_names_requested": num_names_per_category * 4  # 4 categories
            }
        )
        
        try:
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format())
            
            # Format brand values for better display
            formatted_brand_values = "\n".join([f"- {value}" for value in brand_values]) if brand_values else "Not specified"
            
            # Format key attributes for better display
            formatted_key_attributes = "\n".join([f"- {attr}" for attr in key_attributes]) if key_attributes else "Not specified"
            
            # Define naming categories
            categories = ["Descriptive", "Suggestive", "Abstract", "Experiential"]
            
            # Track all generated brand names to ensure uniqueness
            all_names = []
            
            # For each category, generate the requested number of names
            for category in categories:
                category_names = []
                attempts = 0
                max_attempts = num_names_per_category * 3  # Allow for retries
                
                logger.info(
                    f"Generating names for category: {category}",
                    extra={
                        "run_id": run_id,
                        "category": category,
                        "names_to_generate": num_names_per_category
                    }
                )
                
                while len(category_names) < num_names_per_category and attempts < max_attempts:
                    attempts += 1
                    
                    try:
                        logger.debug(
                            f"Attempt {attempts} for {category} name {len(category_names)+1}/{num_names_per_category}",
                            extra={"run_id": run_id}
                        )
                        
                        # Format existing names to avoid duplicates
                        existing_names_text = "None yet" if not all_names else "\n".join([f"- {name}" for name in all_names])
                        
                        # Set up message sequence with category-specific prompt
                        generation_prompt = self.generation_prompt.format(
                            format_instructions=self.output_parser.get_format_instructions(),
                            brand_context=formatted_brand_context,
                            brand_values=formatted_brand_values,
                            purpose=purpose,
                            key_attributes=formatted_key_attributes,
                            category=category,
                            existing_names=existing_names_text
                        )
                        
                        # Create human message
                        human_message = HumanMessage(content=generation_prompt)
                        
                        # Get response from LLM
                        response = await self.llm.ainvoke([system_message, human_message])
                        
                        # Log the raw response for debugging
                        logger.debug(
                            "Raw LLM response received",
                            extra={
                                "run_id": run_id,
                                "category": category,
                                "response_content": response.content[:1000] + "..." if len(response.content) > 1000 else response.content
                            }
                        )
                        
                        # Parse the structured output
                        try:
                            logger.debug(
                                "Attempting to parse structured output",
                                extra={
                                    "run_id": run_id,
                                    "parsing_schema": str(self.output_schemas)
                                }
                            )
                            
                            parsed_output = self.output_parser.parse(response.content)
                            
                            # Log successfully parsed output
                            logger.debug(
                                "Successfully parsed structured output",
                                extra={
                                    "run_id": run_id,
                                    "parsed_fields": list(parsed_output.keys())
                                }
                            )
                            
                            # Check if the parser returned a list of names instead of a single name
                            if isinstance(parsed_output, list):
                                logger.warning(
                                    "Parser returned multiple names, using only the first one",
                                    extra={
                                        "run_id": run_id,
                                        "num_names_returned": len(parsed_output)
                                    }
                                )
                                parsed_output = parsed_output[0] if parsed_output else {}
                            
                            # Check for nested names inside the response
                            nested_names = []
                            keys_to_remove = []
                            for key, value in parsed_output.items():
                                # Look for nested dictionaries that might be additional brand names
                                if isinstance(value, dict) and "brand_name" in value:
                                    nested_names.append(value)
                                    keys_to_remove.append(key)
                                # Or lists of dictionaries
                                elif isinstance(value, list) and all(isinstance(item, dict) and "brand_name" in item for item in value):
                                    nested_names.extend(value)
                                    keys_to_remove.append(key)
                            
                            # Remove the keys that contained nested names
                            for key in keys_to_remove:
                                parsed_output.pop(key, None)
                            
                            # If we found nested names, process them separately
                            if nested_names:
                                logger.warning(
                                    "Found nested brand names in the response, extracting them separately",
                                    extra={
                                        "run_id": run_id,
                                        "num_nested_names": len(nested_names)
                                    }
                                )
                                # We'll continue with the main parsed_output for this iteration
                                # and add the nested names to our processing queue
                                for nested_name in nested_names:
                                    # Process each nested name
                                    try:
                                        # Skip processing if brand_name is missing
                                        if "brand_name" not in nested_name or not nested_name["brand_name"]:
                                            continue
                                        
                                        # Skip if this name is already on our list
                                        if nested_name["brand_name"] in all_names:
                                            logger.warning(
                                                f"Skipping duplicate nested name: {nested_name['brand_name']}",
                                                extra={"run_id": run_id}
                                            )
                                            continue
                                            
                                        # Process this name later
                                        category_names.append(nested_name)
                                        all_names.append(nested_name["brand_name"])
                                        
                                        logger.info(
                                            f"Added nested name to processing queue",
                                            extra={
                                                "run_id": run_id,
                                                "brand_name": nested_name.get("brand_name", "unknown")
                                            }
                                        )
                                    except Exception as nested_error:
                                        logger.error(
                                            "Error processing nested brand name",
                                            extra={
                                                "run_id": run_id,
                                                "error_type": type(nested_error).__name__,
                                                "error_message": str(nested_error)
                                            }
                                        )
                            
                            # Validate numeric fields
                            numeric_fields = [
                                "target_audience_relevance",
                                "market_differentiation",
                                "memorability_score",
                                "pronounceability_score",
                                "visual_branding_potential",
                                "rank"
                            ]
                            
                            for field in numeric_fields:
                                if field in parsed_output:
                                    try:
                                        # Ensure it's a proper numeric value
                                        parsed_output[field] = float(parsed_output[field])
                                    except (ValueError, TypeError):
                                        logger.warning(
                                            f"Invalid numeric value for {field}, converting to default",
                                            extra={
                                                "run_id": run_id,
                                                "field": field,
                                                "value": parsed_output.get(field),
                                                "default": 5.0
                                            }
                                        )
                                        parsed_output[field] = 5.0
                        except Exception as parse_error:
                            logger.error(
                                "Error parsing LLM response",
                                extra={
                                    "run_id": run_id,
                                    "error_type": type(parse_error).__name__,
                                    "error_message": str(parse_error),
                                    "response_content": response.content[:500] + "..." if len(response.content) > 500 else response.content
                                }
                            )
                            
                            # Try to extract information as a fallback
                            try:
                                # First attempt to extract structured information even if full parsing failed
                                import json
                                import re
                                
                                # Initialize the fallback output with empty values we'll try to extract
                                partial_output = {
                                    "brand_name": "",
                                    "naming_category": category,  # Use the current category
                                    "brand_personality_alignment": "",
                                    "brand_promise_alignment": "",
                                    "target_audience_relevance": 0.0,
                                    "market_differentiation": 0.0,
                                    "memorability_score": 0,
                                    "pronounceability_score": 0,
                                    "visual_branding_potential": 0.0,
                                    "name_generation_methodology": "",
                                    "rank": 0
                                }
                                
                                # Try to extract each field individually using regex patterns
                                field_patterns = {
                                    "brand_name": r'"brand_name"\s*:\s*"([^"]+)"',
                                    "brand_personality_alignment": r'"brand_personality_alignment"\s*:\s*"([^"]+)"',
                                    "brand_promise_alignment": r'"brand_promise_alignment"\s*:\s*"([^"]+)"',
                                    "target_audience_relevance": r'"target_audience_relevance"\s*:\s*"?([0-9.]+)"?',
                                    "market_differentiation": r'"market_differentiation"\s*:\s*"?([0-9.]+)"?',
                                    "visual_branding_potential": r'"visual_branding_potential"\s*:\s*"?([0-9.]+)"?',
                                    "name_generation_methodology": r'"name_generation_methodology"\s*:\s*"([^"]+)"',
                                    "rank": r'"rank"\s*:\s*"?([0-9.]+)"?'
                                }
                                
                                # Try to extract each field
                                extraction_success = False
                                for field, pattern in field_patterns.items():
                                    match = re.search(pattern, response.content)
                                    if match and match.group(1):
                                        partial_output[field] = match.group(1)
                                        extraction_success = True
                                
                                # If we couldn't extract the brand name, try a more lenient pattern
                                if not partial_output["brand_name"]:
                                    # Look for anything that looks like a brand name in the response
                                    name_matches = re.findall(r'(?:brand name|name)[\s:]+"?([A-Z][a-zA-Z0-9]+)"?', response.content, re.IGNORECASE)
                                    if name_matches:
                                        partial_output["brand_name"] = name_matches[0]
                                        extraction_success = True
                                
                                # For critical fields that are still empty, try to infer from brand_context
                                if not partial_output["brand_personality_alignment"] and brand_context and "brand_personality" in brand_context:
                                    # Create a meaningful inference based on brand personality
                                    personality = brand_context.get("brand_personality", "")
                                    if personality:
                                        if isinstance(personality, list):
                                            personality = ", ".join(personality[:3])  # Take first 3 traits if it's a list
                                        partial_output["brand_personality_alignment"] = f"Aligns with {personality} personality traits"
                                
                                # For brand promise alignment
                                if not partial_output["brand_promise_alignment"] and brand_context and "brand_promise" in brand_context:
                                    promise = brand_context.get("brand_promise", "")
                                    if promise:
                                        partial_output["brand_promise_alignment"] = f"Reflects the brand promise: {promise}"
                                
                                # For methodology
                                if not partial_output["name_generation_methodology"]:
                                    # Default with category-specific methodology
                                    partial_output["name_generation_methodology"] = f"Generated using {category.lower()} naming approach"
                                
                                # Try to convert numeric fields
                                numeric_fields = [
                                    "target_audience_relevance",
                                    "market_differentiation",
                                    "visual_branding_potential",
                                    "memorability_score",
                                    "pronounceability_score",
                                    "rank"
                                ]
                                
                                for field in numeric_fields:
                                    if field in partial_output and partial_output[field]:
                                        try:
                                            partial_output[field] = float(partial_output[field])
                                        except (ValueError, TypeError):
                                            # If conversion fails, set default values
                                            if field in ["memorability_score", "pronounceability_score"]:
                                                partial_output[field] = 5
                                            else:
                                                partial_output[field] = 5.0  # Use more middle-ground score
                                
                                # Ensure we have a brand name
                                if not partial_output["brand_name"]:
                                    logger.error(
                                        "Failed to extract brand name even with fallback mechanisms",
                                        extra={"run_id": run_id, "response_content": response.content[:200]}
                                    )
                                    raise ValueError("Could not extract brand name")
                                
                                # If we've successfully extracted at least some information
                                if extraction_success:
                                    logger.warning(
                                        "Used fallback extraction to recover brand name data",
                                        extra={
                                            "run_id": run_id,
                                            "brand_name": partial_output["brand_name"],
                                            "fields_extracted": ", ".join([k for k, v in partial_output.items() if v])
                                        }
                                    )
                                    parsed_output = partial_output
                                else:
                                    # If we couldn't extract anything useful, re-raise
                                    raise ValueError("Failed to extract meaningful data")
                            except Exception as extraction_error:
                                logger.error(
                                    "Fallback extraction also failed",
                                    extra={
                                        "run_id": run_id,
                                        "error_type": type(extraction_error).__name__,
                                        "error_message": str(extraction_error)
                                    }
                                )
                                # Continue to next attempt
                                continue
                        
                        # Ensure required fields exist
                        if "brand_name" not in parsed_output or not parsed_output["brand_name"]:
                            logger.warning(
                                "Missing brand_name in LLM response, skipping",
                                extra={"run_id": run_id}
                            )
                            continue
                            
                        # Check for name uniqueness
                        if parsed_output["brand_name"] in all_names:
                            logger.warning(
                                f"Duplicate name generated: {parsed_output['brand_name']}, skipping",
                                extra={"run_id": run_id}
                            )
                            continue
                            
                        # Add metadata to output
                        parsed_output.update({
                            "run_id": run_id,
                            "timestamp": timestamp,
                            # Ensure naming_category is set to current category
                            "naming_category": category
                        })
                        
                        logger.debug(
                            "Generated valid brand name",
                            extra={
                                "run_id": run_id,
                                "brand_name": parsed_output["brand_name"],
                                "category": category
                            }
                        )
                        
                        # Store in Supabase
                        try:
                            await self._store_in_supabase(run_id, parsed_output)
                        except Exception as supabase_error:
                            logger.error(
                                "Error storing brand name in Supabase, but continuing with generation",
                                extra={
                                    "run_id": run_id,
                                    "brand_name": parsed_output.get("brand_name", "unknown"),
                                    "error_type": type(supabase_error).__name__,
                                    "error_message": str(supabase_error)
                                }
                            )
                            # We'll continue processing even if Supabase storage fails
                        
                        # Add to our category list and all names list
                        category_names.append(parsed_output)
                        all_names.append(parsed_output["brand_name"])
                        
                        logger.info(
                            f"Generated {category} name {len(category_names)}/{num_names_per_category}",
                            extra={
                                "run_id": run_id,
                                "brand_name": parsed_output["brand_name"]
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Error generating {category} name",
                            extra={
                                "run_id": run_id,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "error_detail": repr(e),
                                "traceback": traceback.format_exc()
                            }
                        )
                        # Continue trying to generate more names
                        continue
                
                # Add the names from this category to our overall list
                generated_names.extend(category_names)
                
                logger.info(
                    f"Completed generation for {category} category",
                    extra={
                        "run_id": run_id,
                        "category": category,
                        "names_generated": len(category_names)
                    }
                )
            
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
                    "total_names": len(generated_names)
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
        """Store brand name data in Supabase.
        
        Args:
            run_id (str): The run ID for tracking
            name_data (Dict[str, Any]): The brand name data to store
        """
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
            
            # Add text fields
            text_fields = [
                "naming_category", 
                "brand_personality_alignment", 
                "brand_promise_alignment", 
                "name_generation_methodology"
            ]
            
            for field in text_fields:
                supabase_data[field] = name_data.get(field, "")
                
            # Add numeric fields with appropriate handling
            numeric_fields = {
                "target_audience_relevance": 0.0,
                "market_differentiation": 0.0,
                "visual_branding_potential": 0.0,
                "memorability_score": 0.0,
                "pronounceability_score": 0.0,
                "rank": 0.0
            }
            
            for field, default in numeric_fields.items():
                try:
                    value = name_data.get(field, default)
                    # Ensure we have a valid numeric value
                    if value is None or value == "":
                        supabase_data[field] = default
                    else:
                        supabase_data[field] = float(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid {field} value in brand name data, using default",
                        extra={
                            "run_id": run_id,
                            "field": field,
                            "value": name_data.get(field),
                            "default": default
                        }
                    )
                    supabase_data[field] = default
            
            # Add timestamp
            if "timestamp" in name_data:
                supabase_data["timestamp"] = name_data["timestamp"]
            else:
                supabase_data["timestamp"] = datetime.now().isoformat()
                
            # Log the data we're about to insert (for debugging)
            logger.debug(
                "Inserting brand name data into Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": supabase_data["brand_name"],
                    "table": "brand_name_generation",
                    "data": json.dumps({k: str(v) for k, v in supabase_data.items()})
                }
            )
            
            # Store in Supabase using the singleton client
            await self.supabase.execute_with_retry(
                operation="insert",
                table="brand_name_generation",
                data=supabase_data
            )
            
            logger.info(
                "Brand name stored in Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": supabase_data["brand_name"]
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