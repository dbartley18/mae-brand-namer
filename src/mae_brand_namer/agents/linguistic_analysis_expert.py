"""Expert in linguistic analysis of brand names."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
import traceback
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tracers.context import tracing_v2_enabled
from postgrest import APIError

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)


class LinguisticsExpert:
    """Expert in linguistic analysis of brand names.
    
    This expert analyzes brand names across multiple linguistic dimensions including
    phonetics, morphology, semantics, and pragmatics. It evaluates pronunciation,
    memorability, and overall linguistic effectiveness.
    
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
        """Initialize the LinguisticsExpert with dependencies.
        
        Args:
            dependencies: Container for application dependencies
        """
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        # Agent identity
        self.role = "Linguistic Analysis & Phonetic Optimization Expert"
        self.goal = (
            "Analyze brand names for linguistic qualities, phonetic appeal, and "
            "pronunciation characteristics to ensure optimal verbal and auditory "
            "brand expression."
        )
        
        try:
            # Load prompts from YAML files
            prompt_dir = Path(__file__).parent / "prompts" / "linguistics"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
            
            # Log the loaded prompts for debugging
            logger.debug(f"Loaded linguistics system prompt")
            logger.debug(f"Loaded linguistics analysis prompt with variables: {self.analysis_prompt.input_variables}")
            
            # Define output schemas for structured parsing
            self.output_schemas = [
                ResponseSchema(name="pronunciation_ease", description="Ease of pronunciation analysis", type="string"),
                ResponseSchema(name="euphony_vs_cacophony", description="Analysis of phonetic pleasantness vs harshness", type="string"),
                ResponseSchema(name="rhythm_and_meter", description="Analysis of rhythmic patterns and stress", type="string"),
                ResponseSchema(name="phoneme_frequency_distribution", description="Analysis of sound frequency and patterns", type="string"),
                ResponseSchema(name="sound_symbolism", description="How sounds contribute to meaning and associations", type="string"),
                ResponseSchema(name="word_class", description="Part of speech categorization", type="string"),
                ResponseSchema(name="morphological_transparency", description="Analysis of word structure and formation", type="string"),
                ResponseSchema(name="grammatical_gender", description="Grammatical gender implications if applicable", type="string"),
                ResponseSchema(name="inflectional_properties", description="How the name changes in different grammatical contexts", type="string"),
                ResponseSchema(name="ease_of_marketing_integration", description="How well the name fits in marketing contexts", type="string"),
                ResponseSchema(name="naturalness_in_collocations", description="How naturally the name fits in phrases", type="string"),
                ResponseSchema(name="homophones_homographs", description="Whether similar sounding or spelled words exist", type="boolean"),
                ResponseSchema(name="semantic_distance_from_competitors", description="How linguistically distinct from competitors", type="string"),
                ResponseSchema(name="neologism_appropriateness", description="If a new word, how well it works linguistically", type="string"),
                ResponseSchema(name="overall_readability_score", description="Overall linguistic accessibility", type="string"),
                ResponseSchema(name="notes", description="Additional linguistic observations", type="string"),
                ResponseSchema(name="rank", description="Overall linguistic effectiveness ranking", type="number")
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
            
            # Set up the prompt template with both prompts
            system_content = self.system_prompt.format()
            analysis_template = self.analysis_prompt.template
            
            # Update template to use singular brand_name instead of brand_names if needed
            if "{brand_names}" in analysis_template:
                analysis_template = analysis_template.replace("{brand_names}", "{brand_name}")
                logger.info("Updated linguistics analysis template to use singular brand_name")
            
            # Create proper prompt template using both prompts
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_content),
                ("human", analysis_template)
            ])
            
            # Log the input variables for debugging
            logger.info(f"Linguistics prompt expects these variables: {self.prompt.input_variables}")
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=0.5,
                google_api_key=settings.google_api_key,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.error(
                "Error initializing LinguisticsExpert",
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
        """Analyze the linguistic characteristics of a brand name.
        
        Performs a comprehensive linguistic analysis including phonetics,
        morphology, semantics, and pragmatics. Evaluates pronunciation,
        memorability, and overall effectiveness.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Optional additional brand context information
            
        Returns:
            Dictionary containing the linguistic analysis results
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
                    logger.info(f"Analyzing linguistics for brand name: '{brand_name}'")
                    
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
                    logger.info(f"Formatting linguistics prompt with variables: {list(required_vars.keys())}")
                    logger.debug(f"Brand name value: '{brand_name}'")
                    
                    # Format the prompt with all required variables
                    formatted_prompt = self.prompt.format_messages(**required_vars)
                    
                    # Log details about the formatted prompt for debugging
                    if len(formatted_prompt) > 1:
                        logger.debug(f"Formatted prompt sample: {formatted_prompt[1].content[:200]}...")
                        logger.info(f"Brand name in formatted prompt: {brand_name in formatted_prompt[1].content}")
                    
                    # Get response from LLM - with detailed logging
                    logger.info(f"Invoking LLM for linguistics analysis on '{brand_name}'")
                    try:
                        response = await self.llm.ainvoke(formatted_prompt)
                        logger.info(f"LLM invocation successful for linguistics analysis")
                    except Exception as llm_error:
                        logger.error(f"Error invoking LLM: {str(llm_error)}")
                        raise
                    
                    # Log the raw response for debugging
                    if hasattr(response, 'content'):
                        logger.debug(f"LLM response: {response.content[:200]}...")
                        logger.info(f"LLM response length: {len(response.content)}")
                    else:
                        logger.error(f"Unexpected response format from LLM: {type(response)}")
                    
                    # Parse and validate response with error handling
                    try:
                        analysis = self.output_parser.parse(response.content)
                        
                        # Create the result dictionary
                        result = {
                            "brand_name": brand_name,
                            "task_name": "linguistic_analysis"
                        }
                        
                        # Add all analysis fields to the result
                        for key, value in analysis.items():
                            result[key] = value
                            
                        # Ensure rank is a float
                        if "rank" in result:
                            result["rank"] = float(result["rank"])
                            
                    except Exception as parse_error:
                        logger.error(f"Error parsing linguistic analysis: {str(parse_error)}")
                        # Create a minimal valid result
                        result = {
                            "brand_name": brand_name,
                            "task_name": "linguistic_analysis",
                            "pronunciation_ease": "Average",
                            "euphony_vs_cacophony": "Neutral",
                            "rhythm_and_meter": "Standard",
                            "phoneme_frequency_distribution": "Typical",
                            "sound_symbolism": "Minimal",
                            "word_class": "Noun",
                            "morphological_transparency": "Clear",
                            "grammatical_gender": "Neutral",
                            "inflectional_properties": "Standard",
                            "ease_of_marketing_integration": "Moderate",
                            "naturalness_in_collocations": "Neutral",
                            "homophones_homographs": False,
                            "semantic_distance_from_competitors": "Unknown",
                            "neologism_appropriateness": "N/A",
                            "overall_readability_score": "Average",
                            "notes": "Error in analysis process",
                            "rank": 5.0
                        }
                    
                    # Store results
                    await self._store_analysis(run_id, brand_name, result)
                    
                except Exception as e:
                    logger.error(f"Error in linguistic analysis: {str(e)}")
                    # Create a fallback result
                    result = {
                        "brand_name": brand_name,
                        "task_name": "linguistic_analysis",
                        "pronunciation_ease": "Error in analysis",
                        "euphony_vs_cacophony": "Error in analysis",
                        "rhythm_and_meter": "Error in analysis",
                        "sound_symbolism": "Error in analysis",
                        "word_class": "Unknown",
                        "overall_readability_score": "Unknown",
                        "notes": f"Error during analysis: {str(e)}",
                        "rank": 5.0
                    }
            
            return result
            
        except APIError as e:
            logger.error(
                "Supabase API error in linguistic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            # Return a fallback result instead of raising the error
            return {
                "brand_name": brand_name,
                "task_name": "linguistic_analysis",
                "pronunciation_ease": "Database error",
                "euphony_vs_cacophony": "Database error",
                "word_class": "Unknown",
                "notes": f"Database error: {str(e)}",
                "rank": 5.0
            }
            
        except Exception as e:
            logger.error(
                "Error in linguistic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "trace": "".join(traceback.format_exception(type(e), e, e.__traceback__))
                }
            )
            # Return a fallback result instead of raising the error
            return {
                "brand_name": brand_name,
                "task_name": "linguistic_analysis",
                "pronunciation_ease": "Error in analysis",
                "euphony_vs_cacophony": "Error in analysis",
                "word_class": "Unknown",
                "notes": f"Error in analysis: {str(e)}",
                "rank": 5.0
            }
    
    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Store linguistic analysis results in Supabase.
        
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
            # Log what we're trying to store
            logger.debug(f"Preparing to store linguistic analysis for '{brand_name}'")
            
            # Create data structure for Supabase
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "task_name": "linguistic_analysis",
            }
            
            # Add analysis fields, skipping any known problematic fields
            skip_fields = ["timestamp", "task_name", "run_id"]  # Fields to skip
            for key, value in analysis.items():
                if key not in skip_fields and key != "brand_name":  # Skip fields that might cause issues
                    # Handle potential JSON serialization issues
                    if isinstance(value, (dict, list)):
                        try:
                            # Convert to string to avoid serialization issues
                            data[key] = json.dumps(value)
                        except Exception:
                            # If we can't serialize, store as string
                            data[key] = str(value)
                    else:
                        data[key] = value
            
            # Log the keys we're storing
            logger.info(f"Storing linguistic analysis with fields: {list(data.keys())}")
            
            # Perform the insert
            await self.supabase.table("linguistic_analysis").insert(data).execute()
            logger.info(f"Successfully stored linguistic analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing linguistic analysis in Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # Don't raise here - we don't want database errors to prevent the analysis from being returned 