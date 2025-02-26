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
    
    def __init__(self, supabase: SupabaseManager = None):
        """Initialize the BrandNameCreationExpert with necessary configurations."""
        # Initialize Supabase client
        self.supabase = supabase or SupabaseManager()
        
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
        
        try:
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format())
            
            with tracing_v2_enabled():
                # Set up message sequence
                generation_prompt = self.generation_prompt.format(
                    brand_context=brand_context,
                    brand_values=brand_values,
                    purpose=purpose,
                    key_attributes=key_attributes
                )
                
                with tracing_v2_enabled():
                    for i in range(num_names):
                        try:
                            # Format the generation prompt
                            formatted_prompt = generation_prompt
                            
                            # Create human message
                            human_message = HumanMessage(content=formatted_prompt)
                            
                            # Get response from LLM
                            response = await self.llm.ainvoke([system_message, human_message])
                            
                            # Parse the structured output
                            parsed_output = self.output_parser.parse(response.content)
                            parsed_output.update({
                                "run_id": run_id,
                                "timestamp": timestamp,
                                "version": "1.0"
                            })
                            
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
            # Prepare data for Supabase
            supabase_data = {
                "run_id": run_id,
                "brand_name": name_data["brand_name"],
                "naming_category": name_data["naming_category"],
                "brand_personality_alignment": name_data["brand_personality_alignment"],
                "brand_promise_alignment": name_data["brand_promise_alignment"],
                "target_audience_relevance": name_data["target_audience_relevance"],
                "market_differentiation": name_data["market_differentiation"],
                "memorability_score": float(name_data["memorability_score"]),
                "pronounceability_score": float(name_data["pronounceability_score"]),
                "visual_branding_potential": name_data["visual_branding_potential"],
                "name_generation_methodology": name_data["name_generation_methodology"],
                "timestamp": name_data["timestamp"],
                "version": name_data.get("version", "1.0"),
                "rank": float(name_data["rank"])
            }
            
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
                    "brand_name": name_data["brand_name"]
                }
            )
            
        except KeyError as e:
            logger.error(
                "Missing key in brand name data",
                extra={
                    "run_id": run_id,
                    "error_type": "KeyError",
                    "error_message": str(e),
                    "missing_key": str(e)
                }
            )
            raise ValueError(f"Missing required field in brand name data: {str(e)}")
            
        except (TypeError, ValueError) as e:
            logger.error(
                "Invalid data type in brand name data",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise ValueError(f"Invalid data in brand name data: {str(e)}")
            
        except APIError as e:
            logger.error(
                "Supabase API error storing brand name",
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
                "Unexpected error storing brand name",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "brand_name": name_data.get("brand_name", "unknown")
                }
            )
            raise 