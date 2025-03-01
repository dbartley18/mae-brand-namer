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
from ..utils.rate_limiter import google_api_limiter
from ..utils.supabase_utils import SupabaseManager
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
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the TranslationAnalysisExpert with dependencies."""
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
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
                
            result = None  # Initialize result to track if we've successfully generated one
                
            with tracing_v2_enabled():
                try:
                    # Log the brand name being analyzed
                    logger.info(f"Analyzing translation for brand name: '{brand_name}'")
                    
                    # Format prompt with required variables
                    required_vars = {}
                    
                    # Add the variables the template expects
                    if "brand_name" in self.prompt.input_variables:
                        required_vars["brand_name"] = brand_name
                    
                    if "brand_names" in self.prompt.input_variables:
                        required_vars["brand_names"] = brand_name  # Use singular name even if template expects plural
                    
                    if "format_instructions" in self.prompt.input_variables:
                        required_vars["format_instructions"] = self.output_parser.get_format_instructions()
                    
                    if "brand_context" in self.prompt.input_variables and brand_context:
                        required_vars["brand_context"] = brand_context
                    
                    # Format the prompt with all required variables
                    formatted_prompt = self.prompt.format_messages(**required_vars)
                    
                    # Create a unique call ID for rate limiting
                    call_id = f"translation_{brand_name}_{run_id}"
                    
                    # Check if we need to wait due to rate limiting
                    wait_time = await google_api_limiter.wait_if_needed(call_id)
                    if wait_time > 0:
                        logger.info(f"Rate limited: Waited {wait_time:.2f}s before making translation analysis call")
                    
                    # Get response from LLM
                    response = await self.llm.ainvoke(formatted_prompt)
                    
                    # Log the raw response for debugging
                    logger.info(f"Raw LLM response: {response.content[:200]}...")
                    
                    # Parse and process the response
                    try:
                        # First, try to parse as a JSON array since that's what our prompt requests
                        parsed_json = None
                        is_array = False
                        
                        try:
                            parsed_json = json.loads(response.content)
                            is_array = isinstance(parsed_json, list)
                        except json.JSONDecodeError:
                            logger.warning("Response is not valid JSON, falling back to structured parser")
                        
                        # Create a result with the standard required fields
                        result = {
                            "brand_name": brand_name,
                            "task_name": "translation_analysis",
                        }
                        
                        # Process the response based on whether it's an array or not
                        if is_array and parsed_json:
                            # Array response - store each language analysis and compute summary metrics
                            total_languages = len(parsed_json)
                            total_rank = 0
                            languages_analyzed = []
                            problematic_languages = []
                            adaptation_needed = False
                            
                            # Store each language analysis in Supabase
                            for lang_data in parsed_json:
                                try:
                                    # Prepare data for storage
                                    lang_name = lang_data.get("target_language", "Unknown")
                                    languages_analyzed.append(lang_name)
                                    
                                    # Check if language has issues
                                    has_issues = False
                                    
                                    # Get rank (ensure it's a number)
                                    rank = lang_data.get("rank", 5)
                                    if isinstance(rank, str):
                                        try:
                                            rank = float(rank)
                                        except (ValueError, TypeError):
                                            rank = 5.0
                                    
                                    # Low rank or adaptation needed indicates an issue
                                    if rank < 5 or lang_data.get("adaptation_needed", False):
                                        has_issues = True
                                        problematic_languages.append(lang_name)
                                        adaptation_needed = True
                                    
                                    # Add to total for average calculation
                                    total_rank += rank
                                    
                                    # Store in Supabase with proper types
                                    storage_data = {
                                        "run_id": run_id,
                                        "brand_name": brand_name,
                                        "timestamp": datetime.now().isoformat(),
                                        **lang_data  # Include all fields from the language analysis
                                    }
                                    
                                    # Sanitize boolean fields
                                    for field in ["phonetic_similarity_undesirable", "adaptation_needed"]:
                                        if field in storage_data:
                                            value = storage_data[field]
                                            if isinstance(value, str):
                                                storage_data[field] = value.strip().lower() == "true" or value.strip() == "1"
                                    
                                    # Ensure rank is a float
                                    if "rank" in storage_data:
                                        try:
                                            storage_data["rank"] = float(storage_data["rank"])
                                        except (ValueError, TypeError):
                                            storage_data["rank"] = 5.0
                                    
                                    # Store in Supabase
                                    logger.info(f"Storing translation analysis for {lang_name}")
                                    await self.supabase.table("translation_analysis").insert(storage_data).execute()
                                except Exception as store_error:
                                    logger.error(f"Error storing {lang_name} analysis: {str(store_error)}")
                            
                            # Add summary fields to result
                            if total_languages > 0:
                                # Global adaptability as average of ranks
                                result["target_language"] = "Global"
                                result["direct_translation"] = "Multiple languages analyzed"
                                result["overall_global_adaptability"] = round(total_rank / total_languages, 1)
                                result["global_languages_analyzed"] = ", ".join(languages_analyzed)
                                result["rank"] = result["overall_global_adaptability"]
                                
                                # Problem summary
                                if problematic_languages:
                                    result["adaptation_needed"] = True
                                    result["problematic_translations"] = f"Adaptation needed for: {', '.join(problematic_languages)}"
                                    result["proposed_adaptation"] = "Consider market-specific adaptations for problematic languages."
                                else:
                                    result["adaptation_needed"] = False
                                    result["problematic_translations"] = "No major translation issues detected"
                                    result["proposed_adaptation"] = "No adaptation needed, name works globally."
                                
                                # Add other required fields from schema with default values
                                result["semantic_shift"] = "See individual language analyses for details"
                                result["pronunciation_difficulty"] = "Varies by language, see individual analyses"
                                result["phonetic_similarity_undesirable"] = False
                                result["phonetic_retention"] = "See individual language analyses"
                                result["cultural_acceptability"] = "See individual language analyses"
                                result["brand_essence_preserved"] = "See individual language analyses"
                                result["global_consistency_vs_localization"] = "See individual language analyses"
                                result["notes"] = f"Analyzed {total_languages} languages: {', '.join(languages_analyzed)}"
                            else:
                                # No languages analyzed, set defaults
                                result["target_language"] = "Global"
                                result["direct_translation"] = "No languages analyzed"
                                result["rank"] = 5.0
                                
                        else:
                            # Not an array - try using the output parser
                            try:
                                # Parse with structured output parser
                                parsed_result = self.output_parser.parse(response.content)
                                
                                # Add fields to result
                                for key, value in parsed_result.items():
                                    result[key] = value
                                
                                # Ensure target_language is set
                                if "target_language" not in result or not result["target_language"]:
                                    result["target_language"] = "Global"
                                
                                # Store in Supabase
                                storage_data = {
                                    "run_id": run_id,
                                    "brand_name": brand_name,
                                    "timestamp": datetime.now().isoformat(),
                                    **result  # Include all fields
                                }
                                
                                # Remove task_name as it's not in the schema
                                if "task_name" in storage_data:
                                    del storage_data["task_name"]
                                
                                await self.supabase.table("translation_analysis").insert(storage_data).execute()
                                logger.info(f"Stored translation analysis for {result.get('target_language', 'Global')}")
                            except Exception as parse_error:
                                logger.error(f"Error parsing response: {str(parse_error)}")
                                # Set default values for required fields
                                result["target_language"] = "Global"
                                result["direct_translation"] = "Error analyzing translation"
                                result["semantic_shift"] = "Error in analysis"
                                result["pronunciation_difficulty"] = "Unknown due to error"
                                result["phonetic_similarity_undesirable"] = False
                                result["phonetic_retention"] = "Unknown due to error"
                                result["cultural_acceptability"] = "Unknown due to error"
                                result["adaptation_needed"] = False
                                result["proposed_adaptation"] = "N/A - Error in analysis"
                                result["brand_essence_preserved"] = "Unknown due to error"
                                result["global_consistency_vs_localization"] = "Unknown due to error"
                                result["notes"] = f"Error in analysis: {str(parse_error)}"
                                result["rank"] = 5.0
                    
                    except Exception as e:
                        logger.error(f"Error processing translation analysis: {str(e)}")
                        # Create a minimal valid result with required fields
                        result = {
                            "brand_name": brand_name,
                            "task_name": "translation_analysis",
                            "target_language": "Global",
                            "direct_translation": "Error processing analysis",
                            "semantic_shift": "Error in analysis",
                            "pronunciation_difficulty": "Unknown due to error",
                            "phonetic_similarity_undesirable": False,
                            "phonetic_retention": "Unknown due to error",
                            "cultural_acceptability": "Unknown due to error",
                            "adaptation_needed": False,
                            "proposed_adaptation": "N/A - Error in analysis",
                            "brand_essence_preserved": "Unknown due to error",
                            "global_consistency_vs_localization": "Unknown due to error",
                            "notes": f"Error in analysis: {str(e)}",
                            "rank": 5.0
                        }
                
                except Exception as e:
                    logger.error(f"Error in translation analysis: {str(e)}")
                    # Create a fallback result with all required fields
                    result = {
                        "brand_name": brand_name,
                        "task_name": "translation_analysis",
                        "target_language": "Global",
                        "direct_translation": "Error in analysis",
                        "semantic_shift": "Error in analysis",
                        "pronunciation_difficulty": "Unknown due to error",
                        "phonetic_similarity_undesirable": False,
                        "phonetic_retention": "Unknown due to error",
                        "cultural_acceptability": "Unknown due to error",
                        "adaptation_needed": False,
                        "proposed_adaptation": "N/A - Error in analysis",
                        "brand_essence_preserved": "Unknown due to error",
                        "global_consistency_vs_localization": "Unknown due to error",
                        "notes": f"Error during analysis: {str(e)}",
                        "rank": 5.0
                    }
            
            # Return the result (without the run_id to avoid LangGraph issues)
            return {k: v for k, v in result.items() if k != "run_id"}
                
        except Exception as e:
            logger.error(
                "Translation analysis failed",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            # Return a minimal valid result
            return {
                "brand_name": brand_name,
                "task_name": "translation_analysis",
                "target_language": "Global",
                "direct_translation": "Fatal error in analysis",
                "semantic_shift": "Fatal error in analysis",
                "pronunciation_difficulty": "Unknown due to fatal error",
                "phonetic_similarity_undesirable": False,
                "phonetic_retention": "Unknown due to fatal error",
                "cultural_acceptability": "Unknown due to fatal error",
                "adaptation_needed": False,
                "proposed_adaptation": "N/A - Fatal error in analysis",
                "brand_essence_preserved": "Unknown due to fatal error",
                "global_consistency_vs_localization": "Unknown due to fatal error",
                "notes": f"Fatal error in analysis: {str(e)}",
                "rank": 0.0
            }

    async def _store_language_analysis(
        self,
        run_id: str,
        brand_name: str,
        language_data: Dict[str, Any]
    ) -> None:
        """Store a single language analysis in Supabase.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The analyzed brand name
            language_data: Language-specific analysis data to store
        """
        try:
            # Define the expected schema fields
            schema_fields = [
                "target_language", "direct_translation", "semantic_shift", 
                "pronunciation_difficulty", "phonetic_similarity_undesirable", 
                "phonetic_retention", "cultural_acceptability", "adaptation_needed", 
                "proposed_adaptation", "brand_essence_preserved", 
                "global_consistency_vs_localization", "notes", "rank"
            ]
            
            # Prepare data for Supabase
            record = {
                "run_id": run_id,
                "brand_name": brand_name,
                "timestamp": datetime.now().isoformat()
            }
            
            # Default target language if missing
            if "target_language" not in language_data or not language_data["target_language"]:
                language_data["target_language"] = "English"
                
            # Process each field according to its expected type
            for field in schema_fields:
                if field in language_data:
                    value = language_data[field]
                    
                    # Handle boolean fields
                    if field in ["phonetic_similarity_undesirable", "adaptation_needed"]:
                        if isinstance(value, str):
                            value = value.strip().lower() == "true" or value.strip() == "1"
                    
                    # Handle numeric fields
                    if field == "rank":
                        try:
                            if value is not None:
                                value = float(value)
                        except (ValueError, TypeError):
                            value = 5.0  # Default rank
                    
                    record[field] = value
            
            # Log what we're storing
            logger.info(f"Storing analysis for language: {record.get('target_language', 'Unknown')}")
            
            # Insert into Supabase
            try:
                await self.supabase.table("translation_analysis").insert(record).execute()
                logger.info(f"Successfully stored translation analysis for {record.get('target_language', 'Unknown')}")
            except Exception as insert_error:
                logger.error(f"Failed to insert translation analysis: {str(insert_error)}")
            
        except Exception as e:
            logger.error(f"Error preparing translation analysis data: {str(e)}")
            # Don't raise - we don't want to stop the workflow due to storage errors

    # Keep the original _store_analysis method for backward compatibility
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
            # Define the schema fields expected by Supabase
            schema_fields = [
                "direct_translation", "semantic_shift", "pronunciation_difficulty",
                "phonetic_similarity_undesirable", "phonetic_retention",
                "cultural_acceptability", "adaptation_needed", "proposed_adaptation",
                "brand_essence_preserved", "global_consistency_vs_localization",
                "notes", "rank"
            ]
            
            # Check if we have a list of language analyses
            if "translation_analysis_summary" in analysis and analysis.get("translation_analysis_summary"):
                logger.info(f"Storing multiple language analyses for brand name '{brand_name}'")
                
                # Extract the language analyses from the summary
                language_analyses = analysis.get("translation_analysis_summary", {})
                
                # Create a batch of records to insert
                records = []
                
                # Generate a record for each language analysis
                for language, lang_analysis in language_analyses.items():
                    # Ensure required fields are present with correct types
                    record = {
                        "run_id": run_id,
                        "brand_name": brand_name,
                        "target_language": language,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Process fields according to schema
                    for field in schema_fields:
                        if field in lang_analysis:
                            value = lang_analysis.get(field)
                            
                            # Handle boolean fields
                            if field in ["phonetic_similarity_undesirable", "adaptation_needed"]:
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
                            
                            record[field] = value
                    
                    # Ensure required fields have default values if missing
                    if "target_language" not in record or not record["target_language"]:
                        record["target_language"] = language
                        
                    if "direct_translation" not in record:
                        record["direct_translation"] = "Not provided"
                        
                    records.append(record)
                
                # Insert the records into Supabase
                if records:
                    try:
                        logger.debug(f"Inserting records: {records}")
                        await self.supabase.table("translation_analysis").insert(records).execute()
                        logger.info(f"Stored {len(records)} language analyses for brand name '{brand_name}' with run_id '{run_id}'")
                    except Exception as e:
                        logger.error(f"Failed to insert batch records: {str(e)}")
                        # Try inserting one by one
                        for record in records:
                            try:
                                await self.supabase.table("translation_analysis").insert(record).execute()
                            except Exception as single_err:
                                logger.error(f"Failed to insert single record: {str(single_err)}")
                else:
                    logger.warning(f"No language analyses to store for brand name '{brand_name}'")
                    
            else:
                # Handle single analysis record
                data = {
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "target_language": analysis.get("target_language", "English"),  # Default to English if not specified
                    "timestamp": datetime.now().isoformat()
                }
                
                # Process fields according to schema
                for field in schema_fields:
                    if field in analysis:
                        value = analysis.get(field)
                        
                        # Handle boolean fields
                        if field in ["phonetic_similarity_undesirable", "adaptation_needed"]:
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
                
                # Ensure required fields
                if not data.get("target_language"):
                    data["target_language"] = "English"
                    
                try:
                    logger.debug(f"Inserting single record: {data}")
                    await self.supabase.table("translation_analysis").insert(data).execute()
                    logger.info(f"Stored translation analysis for brand name '{brand_name}' with run_id '{run_id}'")
                except Exception as e:
                    logger.error(f"Failed to insert single record: {str(e)}")
            
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
            