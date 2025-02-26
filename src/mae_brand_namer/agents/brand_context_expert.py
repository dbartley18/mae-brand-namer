"""
Brand context extraction and understanding expert.

This module provides the BrandContextExpert class which is responsible for
extracting and structuring brand context information from user input.
It analyzes the user's prompt to identify brand values, personality traits,
target audience, and other key branding elements.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from pathlib import Path

from langchain.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers import LangChainTracer
from postgrest.exceptions import APIError

from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.settings import settings

logger = get_logger(__name__)

class BrandContextExpert:
    """
    Expert in understanding and extracting brand context from user input.
    
    This class is responsible for analyzing user prompts to extract structured
    brand context information, including brand values, personality traits,
    target audience demographics, and market positioning. It serves as the 
    foundation for subsequent brand naming processes.
    
    Attributes:
        supabase (SupabaseManager): Connection manager for Supabase storage
        system_prompt (ChatPromptTemplate): System prompt for the LLM
        extraction_prompt (ChatPromptTemplate): Prompt template for extracting brand context
        llm (ChatGoogleGenerativeAI): LLM instance configured for context extraction
        output_schemas (List[ResponseSchema]): Schema definitions for structured output parsing
        output_parser (StructuredOutputParser): Parser for converting LLM responses to structured data
        
    Example:
        ```python
        # Create expert with dependency injection
        supabase = SupabaseManager()
        expert = BrandContextExpert(supabase=supabase)
        
        # Extract brand context from user prompt
        run_id = "brand-12345678"
        user_prompt = "Create a brand name for a tech startup focused on AI solutions"
        result = await expert.extract_brand_context(run_id, user_prompt)
        
        # Access extracted information
        brand_values = result["brand_values"]
        target_audience = result["target_audience"]
        ```
    """
    
    def __init__(self, supabase: Optional[SupabaseManager] = None):
        """
        Initialize the BrandContextExpert with necessary configurations.
        
        Args:
            supabase (Optional[SupabaseManager]): Supabase connection manager.
                If None, a new instance will be created.
                
        Raises:
            FileNotFoundError: If prompt files cannot be found
            ValueError: If prompt loading fails
        """
        # Initialize Supabase client
        self.supabase = supabase or SupabaseManager()
        
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "brand_context"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.extraction_prompt = load_prompt(str(prompt_dir / "extraction.yaml"))
        except Exception as e:
            logger.error(
                "Error loading prompts",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "prompt_dir": str(prompt_dir)
                }
            )
            raise
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks(),
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(
                name="brand_promise",
                description="The core brand promise and purpose statement"
            ),
            ResponseSchema(
                name="brand_values",
                description="List of 3-5 key brand values"
            ),
            ResponseSchema(
                name="brand_personality",
                description="List of 3-5 brand personality traits"
            ),
            ResponseSchema(
                name="brand_tone_of_voice",
                description="Description of the brand's tone of voice"
            ),
            ResponseSchema(
                name="target_audience",
                description="Description of the target audience"
            ),
            ResponseSchema(
                name="market_positioning",
                description="Description of the brand's market positioning"
            ),
        ]
        
        self.parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        self.format_instructions = self.parser.get_format_instructions()
    
    async def extract_brand_context(
        self,
        user_prompt: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Extract structured brand context from the user prompt.
        
        Args:
            user_prompt (str): The user's prompt describing the brand
            run_id (str): Unique identifier for the workflow run
            
        Returns:
            Dict[str, Any]: Structured brand context information
        """
        try:
            with tracing_v2_enabled():
                # Create message sequence
                system_message = SystemMessage(content=self.system_prompt.format())
                extraction_prompt = self.extraction_prompt.format(
                    format_instructions=self.format_instructions,
                    user_prompt=user_prompt
                )
                human_message = HumanMessage(content=extraction_prompt)
                
                # Invoke LLM
                response = await self.llm.ainvoke([system_message, human_message])
                content = response.content
                
                # Parse the response
                parsed_output = self.parser.parse(content)
                
                # Validate the output
                self._validate_output(parsed_output)
                
                # Add metadata
                parsed_output["run_id"] = run_id
                parsed_output["timestamp"] = datetime.now().isoformat()
                parsed_output["user_prompt"] = user_prompt
                
                # Store in Supabase
                await self._store_in_supabase(run_id, parsed_output)
                
                return parsed_output
                
        except APIError as e:
            logger.error(
                "Supabase API error in brand context extraction",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
            
        except Exception as e:
            logger.error(
                "Error in brand context extraction",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    def _validate_output(self, parsed_output: Dict[str, Any]) -> None:
        """Validate the parsed output to ensure all required fields are present."""
        required_fields = ["brand_promise", "brand_values", "brand_personality", 
                          "brand_tone_of_voice", "target_audience", "market_positioning"]
        
        missing_fields = [field for field in required_fields if field not in parsed_output]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in output: {missing_fields}")
    
    async def _store_in_supabase(self, run_id: str, brand_context: Dict[str, Any]) -> None:
        """Store the brand context in Supabase."""
        try:
            await self.supabase.execute_with_retry(
                operation="insert",
                table="brand_context",
                data=brand_context
            )
            
        except APIError as e:
            logger.error(
                "Supabase API error storing brand context",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None),
                    "data": brand_context
                }
            )
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error storing brand context in Supabase",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "data": brand_context
                }
            )
            raise 