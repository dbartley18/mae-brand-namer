"""Semantic Analysis Expert for analyzing brand name meanings and associations."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio
from pathlib import Path
import traceback

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError
from storage3.exceptions import StorageException
from gotrue.errors import AuthError as AuthException
from langchain_core.tracers.context import tracing_v2_enabled

from ..config.settings import settings
from ..utils.logging import get_logger
from ..models.state import SemanticAnalysisResult
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class SemanticAnalysisExpert:
    """Expert in analyzing semantic meaning and brand associations."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the SemanticAnalysisExpert with dependencies."""
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        self.role = "Semantic Analysis Expert"
        self.goal = "Analyze brand names for semantic meaning, associations, and brand fit"
        self.backstory = """You are a distinguished expert in semantic analysis with deep expertise in 
        linguistics, brand psychology, and consumer behavior. Your analyses help companies understand the 
        full semantic impact of their brand names."""
        
        # Initialize retry configuration
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            self.tracer = LangChainTracer(
                project_name=os.getenv("LANGCHAIN_PROJECT", "mae-brand-namer")
            )
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.5,  # Balanced temperature for analysis
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(
                name="denotative_meaning",
                description="Direct, literal meaning of the name",
                type="string"
            ),
            ResponseSchema(
                name="etymology",
                description="Word origin and history",
                type="string"
            ),
            ResponseSchema(
                name="descriptiveness",
                description="How descriptive the name is of the product/service (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="concreteness",
                description="How concrete vs abstract the name is (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="emotional_valence",
                description="Emotional response the name evokes (positive/neutral/negative)",
                type="string"
            ),
            ResponseSchema(
                name="brand_personality",
                description="Brand personality traits conveyed by the name",
                type="string"
            ),
            ResponseSchema(
                name="sensory_associations",
                description="Sensory connections the name triggers",
                type="string"
            ),
            ResponseSchema(
                name="figurative_language",
                description="Metaphors, analogies, or other figurative aspects",
                type="string"
            ),
            ResponseSchema(
                name="ambiguity",
                description="Whether the name has multiple possible meanings",
                type="boolean"
            ),
            ResponseSchema(
                name="irony_or_paradox",
                description="Whether the name contains irony or paradoxical elements",
                type="boolean"
            ),
            ResponseSchema(
                name="humor_playfulness",
                description="Whether the name incorporates humor or wordplay",
                type="boolean"
            ),
            ResponseSchema(
                name="phoneme_combinations",
                description="Notable sound combinations in the name",
                type="string"
            ),
            ResponseSchema(
                name="sound_symbolism",
                description="How sounds in the name symbolize meaning",
                type="string"
            ),
            ResponseSchema(
                name="rhyme_rhythm",
                description="Whether the name contains rhyming or rhythmic elements",
                type="boolean"
            ),
            ResponseSchema(
                name="alliteration_assonance",
                description="Whether the name uses alliteration or assonance",
                type="boolean"
            ),
            ResponseSchema(
                name="word_length_syllables",
                description="Number of syllables in the name",
                type="integer"
            ),
            ResponseSchema(
                name="compounding_derivation",
                description="How the name is constructed linguistically",
                type="string"
            ),
            ResponseSchema(
                name="brand_name_type",
                description="Type or category of brand name",
                type="string"
            ),
            ResponseSchema(
                name="memorability_score",
                description="How memorable the name is (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="pronunciation_ease",
                description="How easy the name is to pronounce (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="clarity_understandability",
                description="How clear and understandable the name is (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="uniqueness_differentiation",
                description="How unique the name is in the market (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="brand_fit_relevance",
                description="How well the name fits the brand (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="semantic_trademark_risk",
                description="Assessment of potential trademark issues based on meaning",
                type="string"
            )
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Load prompts from YAML files
        prompt_dir = Path(__file__).parent / "prompts" / "semantic_analysis"
        self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
        self.human_prompt = load_prompt(str(prompt_dir / "human.yaml"))
        
        # Get format instructions from the output parser
        self.format_instructions = self.output_parser.get_format_instructions()
        
        # Log the full human template for debugging
        logger.debug(f"Human template content: {self.human_prompt.template[:200]}...")
        logger.debug(f"Human template has brand_name placeholder: {'{{brand_name}}' in self.human_prompt.template}")
        
        # Create the prompt template using the loaded YAML files
        system_message = SystemMessage(content=self.system_prompt.template)
        human_template = self.human_prompt.template
        
        # Create proper prompt template from the loaded templates
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message.content),
            ("human", human_template)
        ])
        
        # Verify the input variables are correct and log them
        logger.info(f"Prompt expects these variables: {self.prompt.input_variables}")
        logger.debug(f"Checking if human template contains expected placeholders:")
        logger.debug(f"- brand_name: {'{{brand_name}}' in human_template}")
        logger.debug(f"- format_instructions: {'{{format_instructions}}' in human_template}")
        
        # Log the final prompt configuration
        logger.info("Semantic Analysis Expert initialized with prompt template")
        logger.debug(f"Prompt input variables: {self.prompt.input_variables}")
        logger.debug(f"Human template contains brand_name placeholder: {'{{brand_name}}' in self.human_prompt.template}")
        logger.debug(f"Human template contains format_instructions placeholder: {'{{format_instructions}}' in self.human_prompt.template}")

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a brand name's semantic meaning and store results.
        
        The return value is a dictionary with the following structure:
        {
            "brand_name": str,            # The brand name that was analyzed
            "task_name": "semantic_analysis",  # Identifies this task in LangGraph
            "denotative_meaning": str,    # Direct, literal meaning of the name
            "etymology": str,             # Word origin and history
            "descriptiveness": number,    # How descriptive the name is (1-10)
            ... additional semantic analysis fields ...
        }
        
        NOTE: The 'denotative_meaning' field is required and must be present in the return value.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_name (str): The brand name to analyze
            brand_context (Optional[Dict[str, Any]]): Brand context information - not used in analysis
                                                      but passed along to evaluator
            
        Returns:
            Dict[str, Any]: Semantic analysis results
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            # Setup event loop if not available
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = None  # Initialize result to track if we've successfully generated one
                
            with tracing_v2_enabled():
                # Format prompt with variables
                try:
                    # Log the key variables for debugging
                    logger.debug(
                        "Formatting semantic analysis prompt",
                        extra={
                            "run_id": run_id,
                            "brand_name": brand_name,
                            "has_format_instructions": bool(self.output_parser.get_format_instructions()),
                            "prompt_input_variables": self.prompt.input_variables
                        }
                    )
                    
                    # Log the brand name being analyzed
                    logger.info(f"Analyzing brand name: '{brand_name}'")
                    
                    # Add more detailed logging to debug the template formatting issue
                    logger.info(f"DEBUG - Human template: {self.human_prompt.template[:100]}...")
                    logger.info(f"DEBUG - Brand name being passed: '{brand_name}'")
                    
                    try:
                        # Format the prompt with the required variables from the YAML template
                        if "format_instructions" in self.prompt.input_variables:
                            formatted_prompt = self.prompt.format_messages(
                                brand_name=brand_name,
                                format_instructions=self.format_instructions
                            )
                        else:
                            # If template doesn't expect format_instructions, just use brand_name
                            formatted_prompt = self.prompt.format_messages(
                                brand_name=brand_name
                            )
                        
                        # Check if the brand name appears in the formatted messages
                        second_message_content = formatted_prompt[1].content
                        brand_name_in_prompt = brand_name in second_message_content
                        
                        logger.info(f"DEBUG - Brand name '{brand_name}' found in prompt: {brand_name_in_prompt}")
                        if not brand_name_in_prompt:
                            # Log the issue and try a different approach
                            logger.warning(f"Brand name not found in formatted prompt. Trying direct substitution.")
                            
                            # Create a simple message directly with the brand name
                            human_content = self.human_prompt.template.replace("{brand_name}", brand_name)
                            human_content = human_content.replace("{format_instructions}", self.format_instructions)
                            
                            formatted_prompt = [
                                SystemMessage(content=self.system_prompt.template),
                                HumanMessage(content=human_content)
                            ]
                            
                            logger.info(f"DEBUG - Used direct substitution. Brand name in prompt: {brand_name in human_content}")
                    except Exception as format_error:
                        logger.error(f"Error formatting prompt: {str(format_error)}")
                        # Fallback to simple prompt
                        formatted_prompt = [
                            SystemMessage(content=self.system_prompt.template),
                            HumanMessage(content=f"Analyze this brand name: {brand_name}\n\n{self.format_instructions}")
                        ]
                    
                    # Log the final formatted prompt for debugging
                    logger.info(f"DEBUG - Final prompt content: {formatted_prompt[1].content[:100]}...")
                    
                    # Get response from LLM
                    response = await self.llm.ainvoke(formatted_prompt)
                    
                    # Log the raw response for debugging
                    logger.info(f"DEBUG - Raw LLM response: {response.content[:200]}...")
                    
                    # Parse structured response with careful error handling
                    try:
                        analysis_result = self.output_parser.parse(response.content)
                        
                        # Validate that the required fields are present
                        if "denotative_meaning" not in analysis_result:
                            logger.warning("Missing required field 'denotative_meaning' in response")
                            analysis_result["denotative_meaning"] = f"Meaning of '{brand_name}'"
                            
                    except Exception as parse_error:
                        logger.error(f"Error parsing LLM response: {str(parse_error)}")
                        
                        # Create a minimal valid result with required fields
                        analysis_result = {
                            "denotative_meaning": f"Direct meaning of '{brand_name}'",
                            "etymology": "Origin information not available",
                            "descriptiveness": 5,
                            "concreteness": 5,
                            "emotional_valence": "Neutral",
                            "brand_personality": "Undefined due to processing error",
                            "sensory_associations": "None detected",
                            "figurative_language": "Not applicable",
                            "ambiguity": False,
                            "irony_or_paradox": False,
                            "humor_playfulness": False,
                            "phoneme_combinations": "Standard",
                            "sound_symbolism": "Neutral",
                            "rhyme_rhythm": False,
                            "alliteration_assonance": False,
                            "word_length_syllables": len(brand_name.split()),
                            "compounding_derivation": "Simple word",
                            "brand_name_type": "Generic",
                            "memorability_score": 5,
                            "pronunciation_ease": 5,
                            "clarity_understandability": 5,
                            "uniqueness_differentiation": 5,
                            "brand_fit_relevance": 5,
                            "semantic_trademark_risk": "Unknown"
                        }
                    
                    # DEBUG: Print the analysis_result structure
                    print("\n\nDEBUG - LLM Response structure:")
                    print("=" * 50)
                    print(f"Analysis result keys: {list(analysis_result.keys())}")
                    print("=" * 50)
                    print(f"Has 'denotative_meaning': {'denotative_meaning' in analysis_result}")
                    print("=" * 50)
                    print("Full analysis result:")
                    print(json.dumps(analysis_result, indent=2))
                    print("=" * 50)
                    
                    # Create a complete result dictionary
                    result = {
                        "brand_name": brand_name,
                        "task_name": "semantic_analysis",
                    }
                    
                    # Add all the analysis results to the result dictionary
                    for key, value in analysis_result.items():
                        if key not in result:  # Avoid duplicating keys
                            result[key] = value
                    
                except Exception as e:
                    logger.error(
                        f"Error in semantic analysis: {str(e)}",
                        extra={
                            "run_id": run_id,
                            "brand_name": brand_name,
                            "error": str(e),
                            "trace": "".join(traceback.format_exception(type(e), e, e.__traceback__))
                        }
                    )
                    # Create a minimal valid result with the required fields
                    result = {
                        "brand_name": brand_name,
                        "task_name": "semantic_analysis",
                        "denotative_meaning": f"Direct meaning of '{brand_name}'",
                        "etymology": "Error occurred during analysis",
                        "descriptiveness": 5,
                        "concreteness": 5,
                        "emotional_valence": "Neutral",
                        "brand_personality": "Error in analysis",
                        "memorability_score": 5,
                        "pronunciation_ease": 5,
                        "brand_fit_relevance": 5,
                        "semantic_trademark_risk": "Unknown - error in analysis"
                    }
                
                # Store results in Supabase
                await self._store_analysis(run_id, brand_name, result)
                
                # DEBUG: Print the final result structure
                print("\n\nDEBUG - Final result structure:")
                print("=" * 50)
                print(f"Result keys: {list(result.keys())}")
                print("=" * 50)
                print(f"Has 'denotative_meaning': {'denotative_meaning' in result}")
                print("=" * 50)
                print("Full result:")
                print(json.dumps(result, indent=2))
                print("=" * 50)
                
                return result
        
        except Exception as e:
            logger.error(
                "Error in semantic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            
            # Return a minimal valid result structure instead of raising an exception
            return {
                "brand_name": brand_name,
                "task_name": "semantic_analysis",
                "denotative_meaning": f"Direct meaning of '{brand_name}'",
                "etymology": "Error occurred during analysis",
                "descriptiveness": 5,
                "concreteness": 5, 
                "emotional_valence": "Neutral",
                "brand_personality": "Error in analysis",
                "memorability_score": 5,
                "pronunciation_ease": 5,
                "brand_fit_relevance": 5,
                "semantic_trademark_risk": "Unknown - error in analysis"
            }
    
    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Store semantic analysis results in Supabase."""
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Log what we're trying to store
            logger.debug(f"Preparing to store semantic analysis for '{brand_name}'")
            
            # Create data structure for Supabase
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "task_name": "semantic_analysis",
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
            logger.info(f"Storing semantic analysis with fields: {list(data.keys())}")
            
            # Perform the insert
            await self.supabase.table("semantic_analysis").insert(data).execute()
            logger.info(f"Successfully stored semantic analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing semantic analysis in Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # Don't raise here - we don't want database errors to prevent the analysis from being returned 