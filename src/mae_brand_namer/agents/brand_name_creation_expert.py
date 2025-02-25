"""Brand Name Creation Expert for generating strategic brand name candidates."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from langchain.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from supabase import APIError, PostgrestError

from ..utils.logging import get_logger
from ..utils.supabase_utils import supabase
from ..config.settings import settings

logger = get_logger(__name__)

class BrandNameCreationExpert:
    """Expert in strategic brand name generation following Alina Wheeler's methodology."""
    
    def __init__(self):
        """Initialize the BrandNameCreationExpert with necessary configurations."""
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
            
            with tracing_enabled(
                tags={
                    "agent": "BrandNameCreationExpert",
                    "task": "generate_brand_names",
                    "run_id": run_id,
                    "prompt_type": "brand_name_generation"
                }
            ):
                for i in range(num_names):
                    # Format the generation prompt
                    formatted_prompt = self.generation_prompt.format(
                        format_instructions=self.output_parser.get_format_instructions(),
                        brand_context={
                            "brand_promise": brand_context.get("brand_promise", "Not specified"),
                            "brand_values": brand_values,
                            "brand_personality": brand_context.get("brand_personality", []),
                            "brand_tone_of_voice": brand_context.get("brand_tone_of_voice", "Not specified"),
                            "brand_purpose": purpose,
                            "brand_mission": brand_context.get("brand_mission", "Not specified"),
                            "target_audience": brand_context.get("target_audience", "Not specified"),
                            "customer_needs": brand_context.get("customer_needs", []),
                            "market_positioning": brand_context.get("market_positioning", "Not specified"),
                            "competitive_landscape": brand_context.get("competitive_landscape", "Not specified"),
                            "industry_focus": brand_context.get("industry_focus", "Not specified"),
                            "industry_trends": brand_context.get("industry_trends", [])
                        }
                    )
                    
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
                
                return generated_names
                
        except Exception as e:
            logger.error(
                "Error generating brand names",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "brand_context": brand_context
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
            await supabase.execute_with_retry(
                operation="insert",
                table="brand_name_generation",
                data=supabase_data
            )
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "Error preparing brand name data for Supabase",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "data": name_data
                }
            )
            raise ValueError(f"Error preparing brand name data: {str(e)}")
            
        except (APIError, PostgrestError) as e:
            logger.error(
                "Error storing brand name in Supabase",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "data": supabase_data
                }
            )
            raise 