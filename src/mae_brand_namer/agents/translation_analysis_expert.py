"""Translation Analysis Expert for evaluating brand names across languages."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import asyncio

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from postgrest.exceptions import APIError
from langchain.callbacks import tracing_enabled
from langchain_core.tracers.context import tracing_v2_enabled

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)

class TranslationAnalysisExpert:
    """Expert in analyzing translation implications and global adaptability of brand names.
    
    This expert evaluates how brand names translate across different languages and cultures,
    assessing potential issues, adaptability needs, and market-specific considerations.
    
    Attributes:
        supabase: Supabase client for data storage
        langsmith: LangSmith client for tracing (optional)
        role (str): The expert's role identifier
        goal (str): The expert's primary objective
        system_prompt: Loaded system prompt template
        output_schemas (List[ResponseSchema]): Schemas for structured output parsing
        prompt (ChatPromptTemplate): Configured prompt template
    """
    
    def __init__(self, dependencies: Dependencies) -> None:
        """Initialize the TranslationAnalysisExpert with dependencies.
        
        Args:
            dependencies: Container for application dependencies
        """
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        try:
            # Agent identity
            self.role = "Global Linguistic Adaptation & Translation Specialist"
            self.goal = (
                "Ensure effective translation of brand names across global markets "
                "while preserving meaning and avoiding negative connotations."
            )
            
            # Load prompts from YAML files
            prompt_dir = Path(__file__).parent / "prompts" / "translation_analysis"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            
            # Try to load analysis.yaml
            try:
                self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
                logger.debug(f"Loaded translation analysis prompt with variables: {self.analysis_prompt.input_variables}")
                
                # Extract template and update to use singular brand_name if needed
                analysis_template = self.analysis_prompt.template
                if "{brand_names}" in analysis_template:
                    analysis_template = analysis_template.replace("{brand_names}", "{brand_name}")
                    logger.info("Updated translation analysis template to use singular brand_name")
            except Exception as e:
                logger.warning(f"Could not load analysis.yaml prompt: {str(e)}")
                # Fallback to hardcoded template
                analysis_template = (
                    "Analyze the translation and global market adaptation needs for the "
                    "following brand name:\n"
                    "Brand Name: {brand_name}\n"
                    "Brand Context: {brand_context}\n"
                    "\nFormat your analysis according to this schema:\n"
                    "{format_instructions}\n\n"
                    "Remember to respond with ONLY a valid JSON object exactly matching the required schema."
                )
                logger.info("Using fallback template for translation analysis")
            
            # Create the system message with additional JSON formatting instructions
            system_content = (self.system_prompt.format() + 
                "\n\nIMPORTANT: You must respond with a valid JSON object that matches EXACTLY the schema provided." +
                "\nDo NOT nest fields under additional keys."
            )
            
            # Create the prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_content),
                ("human", analysis_template)
            ])
            
            # Log the input variables for debugging
            logger.info(f"Translation analysis prompt expects these variables: {self.prompt.input_variables}")
            
            # Define output schemas for structured parsing
            self.output_schemas = [
                ResponseSchema(
                    name="target_language",
                    description="Target language being analyzed",
                    type="string"
                ),
                ResponseSchema(
                    name="direct_translation",
                    description="Direct translation analysis",
                    type="string"
                ),
                ResponseSchema(
                    name="semantic_shift",
                    description="Analysis of meaning changes in this language",
                    type="string"
                ),
                ResponseSchema(
                    name="pronunciation_difficulty",
                    description="Assessment of pronunciation challenges",
                    type="string"
                ),
                ResponseSchema(
                    name="phonetic_similarity_undesirable",
                    description="Whether it sounds like something undesirable",
                    type="boolean"
                ),
                ResponseSchema(
                    name="phonetic_retention",
                    description="How well the original sound is preserved",
                    type="string"
                ),
                ResponseSchema(
                    name="cultural_acceptability",
                    description="Cultural acceptability in target language",
                    type="string"
                ),
                ResponseSchema(
                    name="adaptation_needed",
                    description="Whether adaptation is needed for this market",
                    type="boolean"
                ),
                ResponseSchema(
                    name="proposed_adaptation",
                    description="Suggested adaptation if needed",
                    type="string"
                ),
                ResponseSchema(
                    name="brand_essence_preserved",
                    description="How well brand essence is preserved",
                    type="string"
                ),
                ResponseSchema(
                    name="global_consistency_vs_localization",
                    description="Balance between global consistency and local adaptation",
                    type="string"
                ),
                ResponseSchema(
                    name="notes",
                    description="Additional observations about translation",
                    type="string"
                ),
                ResponseSchema(
                    name="rank",
                    description="Overall translation viability score (1-10)",
                    type="number"
                )
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(
                self.output_schemas
            )
            
            # Initialize Gemini model with tracing
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=settings.model_temperature,
                google_api_key=settings.google_api_key,
                convert_system_message_to_human=True,
                callbacks=[self.langsmith] if self.langsmith else []
            )
        except Exception as e:
            logger.error(
                "Error initializing TranslationAnalysisExpert",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze translation implications for a brand name across multiple languages.
        
        This evaluates how well a brand name translates to other major languages,
        identifies potential issues, and assesses global adaptability.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Optional additional brand context information
            
        Returns:
            Dict[str, Any]: Analysis results including translations, issues, etc.
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
                    logger.info(f"Analyzing translation for brand name: '{brand_name}'")
                    
                    # Format prompt with all required variables
                    required_vars = {}
                    
                    # Add the variables the template expects
                    if "brand_name" in self.prompt.input_variables:
                        required_vars["brand_name"] = brand_name
                    
                    if "brand_names" in self.prompt.input_variables:
                        required_vars["brand_names"] = brand_name  # Use singular name even if template expects plural
                    
                    if "format_instructions" in self.prompt.input_variables:
                        required_vars["format_instructions"] = self.output_parser.get_format_instructions()
                    
                    if "brand_context" in self.prompt.input_variables:
                        required_vars["brand_context"] = brand_context if brand_context else "A brand name for consideration"
                    
                    # Log the variables being passed to the template
                    logger.info(f"Formatting translation analysis prompt with variables: {list(required_vars.keys())}")
                    logger.debug(f"Brand name value: '{brand_name}'")
                    
                    # Format the prompt with all required variables
                    formatted_prompt = self.prompt.format_messages(**required_vars)
                    
                    # Log details about the formatted prompt for debugging
                    if len(formatted_prompt) > 1:
                        logger.debug(f"Formatted prompt sample: {formatted_prompt[1].content[:200]}...")
                        logger.info(f"Brand name in formatted prompt: {brand_name in formatted_prompt[1].content}")
                    
                    # Get response from LLM
                    response = await self.llm.ainvoke(formatted_prompt)
                    
                    # Log the raw response for debugging
                    logger.debug(f"Translation analysis response: {response.content[:200]}...")
                    
                    # Parse and validate response with error handling
                    try:
                        analysis = self.output_parser.parse(response.content)
                        
                        # Create a result with brand_name but without run_id (which is handled by LangGraph)
                        result = {
                            "brand_name": brand_name,
                            "task_name": "translation_analysis",
                        }
                        
                        # Add all analysis results
                        for key, value in analysis.items():
                            if key not in result and key != "run_id":  # Avoid duplicating keys or including run_id
                                result[key] = value
                                
                    except Exception as parse_error:
                        logger.error(f"Error parsing translation analysis: {str(parse_error)}")
                        # Create a minimal valid result
                        result = {
                            "brand_name": brand_name,
                            "task_name": "translation_analysis",
                            "english_pronunciation": "Error in analysis",
                            "spanish_translation": "Error in analysis",
                            "french_translation": "Error in analysis",
                            "german_translation": "Error in analysis",
                            "chinese_translation": "Error in analysis",
                            "japanese_translation": "Error in analysis",
                            "translation_consistency": "Error in analysis",
                            "problematic_translations": "Unknown due to processing error",
                            "translation_recommendation": "Unable to provide recommendation due to error",
                            "overall_global_adaptability": 5,
                            "notes": f"Error in analysis: {str(parse_error)}"
                        }
                    
                    # Store results
                    await self._store_analysis(run_id, brand_name, result)
                    
                except Exception as e:
                    logger.error(f"Error in translation analysis: {str(e)}")
                    # Create a fallback result
                    result = {
                        "brand_name": brand_name,
                        "task_name": "translation_analysis",
                        "english_pronunciation": "Error in analysis",
                        "spanish_translation": "Error in analysis",
                        "translation_recommendation": "Unable to provide recommendation due to error",
                        "overall_global_adaptability": 5,
                        "notes": f"Error during analysis: {str(e)}"
                    }
            
            return result
                
        except Exception as e:
            logger.error(
                "Translation analysis failed",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise ValueError(f"Failed to analyze brand name: {str(e)}")

    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Store translation analysis results in Supabase.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The analyzed brand name
            analysis: Analysis results to store
            
        Raises:
            Exception: If storage fails
        """
        try:
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "timestamp": datetime.now().isoformat(),
                **analysis
            }
            
            await self.supabase.table("translation_analysis").insert(data).execute()
            logger.info(f"Stored translation analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Unexpected error storing translation analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # Don't raise here - we don't want database errors to prevent the analysis from being returned
            