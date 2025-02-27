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
            model="gemini-1.5-pro",
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
            ResponseSchema(name="target_audience_relevance", description="Suitability of the name for the intended target audience"),
            ResponseSchema(name="market_differentiation", description="How well the name stands out from competitors and reinforces brand positioning"),
            ResponseSchema(name="memorability_score", description="Score from 1-10 indicating how easily the name can be remembered", type="number"),
            ResponseSchema(name="pronounceability_score", description="Score from 1-10 indicating how easily the name can be pronounced", type="number"),
            ResponseSchema(name="visual_branding_potential", description="How well the name lends itself to logos, typography, and digital branding"),
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
            num_names: int = 5
        ) -> List[Dict[str, Any]]:
        """
        Generate brand name candidates based on the brand context.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_context (Dict[str, Any]): Brand context information
            brand_values (List[str]): List of brand values
            purpose (str): Brand purpose
            key_attributes (List[str]): Key brand attributes
            num_names (int, optional): Number of brand names to generate. Defaults to 5.
            
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
            
        logger.info(
            "Starting brand name generation", 
            extra={
                "run_id": run_id,
                "num_names_requested": num_names
            }
        )
        
        try:
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format())
            
            # Set up message sequence
            generation_prompt = self.generation_prompt.format(
                format_instructions=self.output_parser.get_format_instructions(),
                brand_context=brand_context,
                brand_values=brand_values,
                purpose=purpose,
                key_attributes=key_attributes
            )
            
            with tracing_v2_enabled():
                for i in range(num_names):
                    try:
                        logger.debug(
                            f"Generating brand name {i+1}/{num_names}",
                            extra={"run_id": run_id}
                        )
                        
                        # Format the generation prompt
                        formatted_prompt = generation_prompt
                        
                        # Create human message
                        human_message = HumanMessage(content=formatted_prompt)
                        
                        # Get response from LLM
                        response = await self.llm.ainvoke([system_message, human_message])
                        
                        # Parse the structured output
                        parsed_output = self.output_parser.parse(response.content)
                        
                        # Ensure required fields exist
                        if "brand_name" not in parsed_output or not parsed_output["brand_name"]:
                            logger.warning(
                                "Missing brand_name in LLM response, skipping",
                                extra={"run_id": run_id}
                            )
                            continue
                            
                        # Add metadata to output
                        parsed_output.update({
                            "run_id": run_id,
                            "timestamp": timestamp
                        })
                        
                        logger.debug(
                            "Generated valid brand name",
                            extra={
                                "run_id": run_id,
                                "brand_name": parsed_output["brand_name"]
                            }
                        )
                        
                        # Store in Supabase
                        await self._store_in_supabase(run_id, parsed_output)
                        generated_names.append(parsed_output)
                        
                        logger.info(
                            f"Generated brand name {i+1}/{num_names}",
                            extra={
                                "run_id": run_id,
                                "brand_name": parsed_output["brand_name"]
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Error generating brand name {i+1}/{num_names}",
                            extra={
                                "run_id": run_id,
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
            supabase_data["naming_category"] = name_data.get("naming_category", "")
            supabase_data["brand_personality_alignment"] = name_data.get("brand_personality_alignment", "")
            supabase_data["brand_promise_alignment"] = name_data.get("brand_promise_alignment", "")
            supabase_data["target_audience_relevance"] = name_data.get("target_audience_relevance", "")
            supabase_data["market_differentiation"] = name_data.get("market_differentiation", "")
            supabase_data["visual_branding_potential"] = name_data.get("visual_branding_potential", "")
            supabase_data["name_generation_methodology"] = name_data.get("name_generation_methodology", "")
            
            # Convert numeric fields safely
            try:
                supabase_data["memorability_score"] = float(name_data.get("memorability_score", 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid memorability_score: {name_data.get('memorability_score')}, defaulting to 0")
                supabase_data["memorability_score"] = 0.0
                
            try:
                supabase_data["pronounceability_score"] = float(name_data.get("pronounceability_score", 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid pronounceability_score: {name_data.get('pronounceability_score')}, defaulting to 0")
                supabase_data["pronounceability_score"] = 0.0
                
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
            
            # Define known valid fields for the brand_name_generation table
            valid_fields = [
                "run_id", "brand_name", "naming_category", "brand_personality_alignment",
                "brand_promise_alignment", "target_audience_relevance", "market_differentiation",
                "memorability_score", "pronounceability_score", "visual_branding_potential",
                "name_generation_methodology", "timestamp", "rank"
            ]
            
            # Filter out any fields that don't exist in the database schema
            filtered_data = {k: v for k, v in supabase_data.items() if k in valid_fields}
            
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