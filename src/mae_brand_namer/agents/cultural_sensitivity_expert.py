"""Cultural Sensitivity Expert for analyzing brand names across cultural contexts."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from postgrest import APIError
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies
from ..utils.rate_limiter import google_api_limiter
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class CulturalSensitivityExpert:
    """Expert in analyzing brand names for cultural sensitivity and appropriateness."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the CulturalSensitivityExpert with dependencies."""
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        try:
            # Load prompts
            prompt_dir = Path(__file__).parent / "prompts" / "cultural_sensitivity"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
            
            # Log the loaded prompts for debugging
            logger.debug(f"Loaded cultural sensitivity system prompt")
            logger.debug(f"Loaded cultural sensitivity analysis prompt with variables: {self.analysis_prompt.input_variables}")
            
            # Define output schemas for structured parsing
            self.output_schemas = [
                ResponseSchema(name="cultural_connotations", description="Global cultural associations", type="string"),
                ResponseSchema(name="symbolic_meanings", description="Symbolic interpretations across cultures", type="string"),
                ResponseSchema(name="alignment_with_cultural_values", description="How well it aligns with cultural values", type="string"),
                ResponseSchema(name="religious_sensitivities", description="Religious associations or concerns", type="string"),
                ResponseSchema(name="social_political_taboos", description="Social or political issues to consider", type="string"),
                ResponseSchema(name="body_part_bodily_function_connotations", description="Whether it has associations with body parts or functions", type="boolean"),
                ResponseSchema(name="age_related_connotations", description="Age-related considerations", type="string"),
                ResponseSchema(name="gender_connotations", description="Gender associations or bias", type="string"),
                ResponseSchema(name="regional_variations", description="How meanings vary by region", type="string"),
                ResponseSchema(name="historical_meaning", description="Historical associations or context", type="string"),
                ResponseSchema(name="current_event_relevance", description="Relevance to current events", type="string"),
                ResponseSchema(name="overall_risk_rating", description="Overall risk rating (Low/Medium/High)", type="string"),
                ResponseSchema(name="notes", description="Additional cultural sensitivity observations", type="string"),
                ResponseSchema(name="rank", description="Overall cultural sensitivity score (1-10)", type="number")
            ]
            
            self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
            
            # Create prompt template using the loaded YAML files
            system_content = (self.system_prompt.format() + 
                "\n\nIMPORTANT: You must respond with a valid JSON object that matches EXACTLY the schema provided." +
                "\nThe JSON MUST contain all the fields specified below at the TOP LEVEL of the object." +
                "\nDo NOT nest fields under additional keys or create your own object structure." +
                "\nUse EXACTLY the field names provided in the schema - do not modify, merge, or rename any fields." +
                "\nDo not include any preamble or explanation outside the JSON structure." +
                "\nDo not use markdown formatting for the JSON."
                
            )
            
            # Extract and update analysis template if needed
            analysis_template = self.analysis_prompt.template
            
            # Update template to use singular brand_name instead of brand_names if needed
            if "{brand_names}" in analysis_template:
                analysis_template = analysis_template.replace("{brand_names}", "{brand_name}")
                logger.info("Updated cultural sensitivity template to use singular brand_name")
            
            # Create the prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_content),
                ("human", analysis_template)
            ])
            
            # Log the input variables for debugging
            logger.info(f"Cultural sensitivity prompt expects these variables: {self.prompt.input_variables}")
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=0.5,
                google_api_key=settings.google_api_key,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.error(
                "Error initializing CulturalSensitivityExpert",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str
    ) -> Dict[str, Any]:
        """Analyze a brand name for cultural sensitivity across major global regions.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            
        Returns:
            Dictionary with cultural sensitivity analysis results
        """
        try:
            # Setup event loop if not available
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Initialize result to track if we've successfully generated one
            result = None
            
            with tracing_v2_enabled():
                try:
                    # Log the brand name being analyzed
                    logger.info(f"Analyzing cultural sensitivity for brand name: '{brand_name}'")
                    
                    # Format the prompt with all required variables
                    required_vars = {}
                    
                    # Add the variables the template expects
                    if "brand_name" in self.prompt.input_variables:
                        required_vars["brand_name"] = brand_name
                    
                    if "brand_names" in self.prompt.input_variables:
                        required_vars["brand_names"] = brand_name  # Use singular name even if template expects plural
                    
                    if "format_instructions" in self.prompt.input_variables:
                        required_vars["format_instructions"] = self.output_parser.get_format_instructions()
                    
                    # Format the prompt with all required variables
                    formatted_prompt = self.prompt.format_messages(**required_vars)
                    
                    # Wait if needed to respect rate limits before making the LLM call
                    call_id = f"cultural_analysis_{brand_name}_{run_id[-8:]}"
                    wait_time = await google_api_limiter.wait_if_needed(call_id)
                    if wait_time > 0:
                        logger.info(f"Rate limited: waited {wait_time:.2f}s before LLM call for {brand_name}")
                    
                    # Get response from LLM
                    logger.info(f"Making LLM call for cultural analysis of '{brand_name}'")
                    response = await self.llm.ainvoke(formatted_prompt)
                    
                    # Parse the response according to the defined schema
                    content = response.content if hasattr(response, 'content') else str(response)
                    analysis = self.output_parser.parse(content)
                    
                    # Create a result dictionary with standard required fields
                    # Create without run_id to avoid LangGraph issues
                    result = {
                        "brand_name": brand_name,
                        "task_name": "cultural_sensitivity_analysis",
                    }
                    
                    # Add all analysis results but exclude run_id
                    for key, value in analysis.items():
                        if key != "run_id":  # Don't include run_id in the result
                            result[key] = value
                        
                    # Ensure rank is a float
                    if "rank" in result:
                        result["rank"] = float(result["rank"])
                        
                except Exception as e:
                    logger.error(f"Error in cultural sensitivity analysis: {str(e)}")
                    # Create a fallback result
                    result = {
                        "brand_name": brand_name,
                        "task_name": "cultural_sensitivity_analysis",
                        "cultural_connotations": "Error in analysis",
                        "symbolic_meanings": "Error in analysis",
                        "alignment_with_cultural_values": "Unknown",
                        "religious_sensitivities": "Unknown",
                        "social_political_taboos": "Unknown",
                        "body_part_bodily_function_connotations": False,
                        "age_related_connotations": "Unknown",
                        "gender_connotations": "Unknown",
                        "regional_variations": "Unknown",
                        "historical_meaning": "Unknown",
                        "current_event_relevance": "Unknown",
                        "overall_risk_rating": "High risk due to processing error",
                        "notes": f"Error during analysis: {str(e)}",
                        "rank": 5.0
                    }
            
            # Store results in Supabase (this doesn't affect what we return to LangGraph)
            await self._store_analysis(run_id, brand_name, result)
            
            # Make sure result doesn't contain run_id before returning
            return {key: value for key, value in result.items() if key != "run_id"}
            
        except APIError as e:
            logger.error(
                "Supabase API error in cultural sensitivity analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            # Return a fallback result without run_id
            return {
                "brand_name": brand_name,
                "task_name": "cultural_sensitivity_analysis",
                "cultural_connotations": "Database error",
                "symbolic_meanings": "Database error",
                "overall_risk_rating": "Unknown due to database error",
                "notes": f"Database error: {str(e)}",
                "rank": 5.0
            }
            
        except Exception as e:
            logger.error(
                "Error in cultural sensitivity analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # Return a fallback result without run_id
            return {
                "brand_name": brand_name,
                "task_name": "cultural_sensitivity_analysis",
                "cultural_connotations": "Error in analysis",
                "symbolic_meanings": "Error in analysis",
                "overall_risk_rating": "High risk due to processing error",
                "notes": f"Error during analysis: {str(e)}",
                "rank": 5.0
            }
    
    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Store cultural sensitivity analysis results in Supabase.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The analyzed brand name
            analysis: Analysis results to store
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Define schema fields for the cultural_sensitivity_analysis table
            schema_fields = [
                "cultural_connotations", "symbolic_meanings", "alignment_with_cultural_values",
                "religious_sensitivities", "social_political_taboos", "body_part_bodily_function_connotations",
                "age_related_connotations", "gender_connotations", "regional_variations",
                "historical_meaning", "current_event_relevance", "overall_risk_rating", 
                "notes", "rank"
            ]
            
            # Create the base record
            data = {
                "run_id": run_id,
                "brand_name": brand_name
            }
            
            # Process each field according to its expected type in the schema
            for field in schema_fields:
                if field in analysis and field != "task_name":  # Skip task_name
                    value = analysis.get(field)
                    
                    # Handle boolean fields
                    if field == "body_part_bodily_function_connotations":
                        # Convert string "true"/"false" to boolean if needed
                        if isinstance(value, str):
                            value = value.strip().lower() == "true" or value.strip() == "1"
                    
                    # Handle numeric fields
                    if field == "rank":
                        try:
                            if value is not None:
                                value = float(value)
                        except (ValueError, TypeError):
                            value = 5.0  # Default rank
                    
                    data[field] = value
            
            try:
                # Log detailed info about what we're storing
                logger.debug(f"Attempting to insert into 'cultural_sensitivity_analysis' table with data keys: {list(data.keys())}")
                logger.debug(f"Type of body_part_bodily_function_connotations: {type(data.get('body_part_bodily_function_connotations'))}")
                
                # Perform the insert with better error reporting
                result = await self.supabase.table("cultural_sensitivity_analysis").insert(data).execute()
                logger.info(f"Successfully stored cultural sensitivity analysis for brand name '{brand_name}' with run_id '{run_id}'")
                
            except Exception as e:
                logger.error(f"Failed to insert record: {str(e)}")
                logger.error(f"Data that failed: {data}")
                # Don't raise the exception - we'll continue with the process
            
        except APIError as e:
            logger.error(
                "Supabase API error storing cultural sensitivity analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            # Don't raise - allow the process to continue
            
        except Exception as e:
            logger.error(
                "Unexpected error storing cultural sensitivity analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # Don't raise - allow the process to continue 