"""Brand context extraction and understanding expert."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path

from langchain.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.messages import SystemMessage, HumanMessage
from supabase.lib.exceptions import APIError, PostgrestError

from ..utils.logging import get_logger
from ..utils.supabase_utils import supabase
from ..config.settings import settings

logger = get_logger(__name__)

class BrandContextExpert:
    """Expert in understanding and extracting brand context from user input."""
    
    def __init__(self):
        """Initialize the BrandContextExpert with necessary configurations."""
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "brand_context"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.extraction_prompt = load_prompt(str(prompt_dir / "extraction.yaml"))
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="brand_promise", description="Core brand promise"),
            ResponseSchema(name="brand_values", description="List of brand values"),
            ResponseSchema(name="brand_personality", description="List of brand personality traits"),
            ResponseSchema(name="brand_tone_of_voice", description="Brand's tone of voice"),
            ResponseSchema(name="brand_purpose", description="Brand's purpose statement"),
            ResponseSchema(name="brand_mission", description="Brand's mission statement"),
            ResponseSchema(name="target_audience", description="Description of target audience"),
            ResponseSchema(name="customer_needs", description="List of customer needs"),
            ResponseSchema(name="market_positioning", description="Brand's market positioning"),
            ResponseSchema(name="competitive_landscape", description="Overview of competitive landscape"),
            ResponseSchema(name="industry_focus", description="Primary industry focus"),
            ResponseSchema(name="industry_trends", description="List of relevant industry trends")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
    
    async def extract_brand_context(self, user_prompt: str, run_id: str) -> Dict[str, Any]:
        """
        Extract brand context from user prompt.
        
        Args:
            user_prompt (str): User's description of the brand
            run_id (str): Unique identifier for this workflow run
            
        Returns:
            Dict[str, Any]: Extracted brand context
            
        Raises:
            ValueError: If extraction fails
            APIError: If Supabase operation fails
        """
        try:
            # Create system message from system prompt
            system_message = SystemMessage(content=self.system_prompt.format())
            
            # Format the extraction prompt
            formatted_prompt = self.extraction_prompt.format(
                format_instructions=self.output_parser.get_format_instructions(),
                user_prompt=user_prompt
            )
            
            # Create human message
            human_message = HumanMessage(content=formatted_prompt)
            
            with tracing_enabled(
                tags={
                    "agent": "BrandContextExpert",
                    "task": "extract_brand_context",
                    "run_id": run_id,
                    "prompt_type": "brand_context_extraction"
                }
            ):
                # Get response from LLM with both messages
                response = await self.llm.ainvoke([system_message, human_message])
                
                # Parse the structured output
                parsed_output = self.output_parser.parse(response.content)
                
                # Validate output
                self._validate_output(parsed_output)
                
                # Add metadata
                brand_context = {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    **parsed_output
                }
                
                # Store in Supabase
                await self._store_in_supabase(run_id, brand_context)
                
                return brand_context
                
        except Exception as e:
            logger.error(
                "Error extracting brand context",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "user_prompt": user_prompt
                }
            )
            raise ValueError(f"Failed to extract brand context: {str(e)}")
    
    def _validate_output(self, parsed_output: Dict[str, Any]) -> None:
        """Validate the parsed output contains all required fields."""
        required_fields = {schema.name for schema in self.output_schemas}
        missing_fields = required_fields - set(parsed_output.keys())
        
        if missing_fields:
            raise ValueError(f"Missing required fields in output: {missing_fields}")
    
    async def _store_in_supabase(self, run_id: str, brand_context: Dict[str, Any]) -> None:
        """Store the brand context in Supabase."""
        try:
            await supabase.execute_with_retry(
                operation="insert",
                table="brand_context",
                data=brand_context
            )
            
        except (APIError, PostgrestError) as e:
            logger.error(
                "Error storing brand context in Supabase",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "data": brand_context
                }
            )
            raise APIError(f"Failed to store brand context: {str(e)}") 