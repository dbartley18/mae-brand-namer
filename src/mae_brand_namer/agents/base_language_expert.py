"""Base class for language-specific translation experts."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from postgrest.exceptions import APIError

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies
from ..utils.rate_limiter import google_api_limiter
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class BaseLanguageTranslationExpert:
    """Base class for language-specific translation analysis experts.
    
    This class provides common functionality for analyzing brand names in specific languages,
    with language-specific experts inheriting and extending this functionality.
    """
    
    def __init__(self, 
                 dependencies=None, 
                 supabase: SupabaseManager = None,
                 language_code: str = None,
                 language_name: str = None):
        """Initialize the language translation expert with dependencies.
        
        Args:
            dependencies: Optional dependencies container
            supabase: Optional Supabase manager
            language_code: ISO code for the language (e.g., 'es', 'zh', 'ar')
            language_name: Full name of the language (e.g., 'Spanish', 'Mandarin Chinese')
        """
        # Set up dependencies
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        # Language identification
        self.language_code = language_code
        self.language_name = language_name
        
        # Set up role and goal based on language
        self.role = f"{self.language_name} Translation Analysis Expert"
        self.goal = f"Analyze brand names for their linguistic and cultural fit in {self.language_name}"
        
        try:
            # Load common prompts
            self._load_prompts()
            
            # Define output schemas for structured parsing
            self._setup_output_schemas()
            
            # Initialize LLM
            self._setup_llm()
        except Exception as e:
            logger.error(
                f"Error initializing {self.language_name} Translation Expert",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    def _load_prompts(self):
        """Load and prepare prompts from YAML files.
        
        This method loads base prompts and allows language-specific customization.
        """
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
                logger.info(f"Updated {self.language_name} translation analysis template to use singular brand_name")
        except Exception as e:
            logger.warning(f"Could not load analysis.yaml prompt: {str(e)}")
            # Fallback to hardcoded template
            analysis_template = (
                f"Analyze the translation and cultural fit of the following brand name in {self.language_name}:\n"
                "Brand Name: {brand_name}\n"
                "Brand Context: {brand_context}\n"
                "\nFormat your analysis according to this schema:\n"
                "{format_instructions}\n\n"
                "Remember to respond with ONLY a valid JSON object exactly matching the required schema."
            )
            logger.info(f"Using fallback template for {self.language_name} translation analysis")
        
        # Add language-specific context to the system prompt
        language_specific_additions = self._get_language_specific_prompt_additions()
        
        # Create the system message with additional formatting instructions
        system_content = (self.system_prompt.format() + 
            "\n\n" + language_specific_additions +
            "\n\nIMPORTANT: You must respond with a valid JSON object that matches EXACTLY the schema provided." +
            "\nDo NOT nest fields under additional keys." +
            f"\nYou are analyzing ONLY for {self.language_name}. Do not analyze other languages." +
            "\n\nCRITICAL: Your analysis MUST be written in English, regardless of the target language. Only the 'direct_translation' field and examples within other fields should contain text in the target language. All explanations and analysis must be in English."
        )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_content),
            ("human", analysis_template)
        ])
        
        # Log the input variables for debugging
        logger.info(f"{self.language_name} translation prompt expects these variables: {self.prompt.input_variables}")
    
    def _get_language_specific_prompt_additions(self) -> str:
        """Get language-specific additions to the system prompt.
        
        This method should be overridden by language-specific subclasses.
        
        Returns:
            String containing language-specific prompt content
        """
        return f"""
        You are a specialized {self.language_name} language and cultural translation expert.
        Focus only on how the brand name would be perceived, pronounced, and understood 
        by {self.language_name} speakers. Consider cultural nuances specific to {self.language_name}-speaking regions.
        """
    
    def _setup_output_schemas(self):
        """Define output schemas for structured parsing."""
        self.output_schemas = [
            ResponseSchema(
                name="target_language",
                description=f"Should be set to {self.language_name}",
                type="string"
            ),
            ResponseSchema(
                name="direct_translation",
                description=f"Direct translation or meaning in {self.language_name}",
                type="string"
            ),
            ResponseSchema(
                name="semantic_shift",
                description=f"Analysis of meaning changes in {self.language_name}",
                type="string"
            ),
            ResponseSchema(
                name="pronunciation_difficulty",
                description=f"Assessment of pronunciation challenges for {self.language_name} speakers",
                type="string"
            ),
            ResponseSchema(
                name="phonetic_similarity_undesirable",
                description=f"Whether it sounds like something undesirable in {self.language_name}",
                type="boolean"
            ),
            ResponseSchema(
                name="phonetic_retention",
                description="How well the original sound is preserved when pronounced by native speakers",
                type="string"
            ),
            ResponseSchema(
                name="cultural_acceptability",
                description=f"Cultural acceptability in {self.language_name}-speaking regions",
                type="string"
            ),
            ResponseSchema(
                name="adaptation_needed",
                description=f"Whether adaptation is needed for {self.language_name} markets",
                type="boolean"
            ),
            ResponseSchema(
                name="proposed_adaptation",
                description="Suggested adaptation if needed",
                type="string"
            ),
            ResponseSchema(
                name="brand_essence_preserved",
                description="How well brand essence is preserved in this language",
                type="string"
            ),
            ResponseSchema(
                name="global_consistency_vs_localization",
                description="Balance between global consistency and local adaptation",
                type="string"
            ),
            ResponseSchema(
                name="notes",
                description=f"Additional observations about {self.language_name} translation",
                type="string"
            ),
            ResponseSchema(
                name="rank",
                description=f"Overall viability score in {self.language_name} markets (1-10)",
                type="number"
            )
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.output_schemas
        )
    
    def _setup_llm(self):
        """Set up the language model with appropriate configuration."""
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=settings.model_temperature,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else []
        )
    
    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
    ) -> Dict[str, Any]:
        """Analyze a brand name for translation implications in this specific language.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            
        Returns:
            Dict[str, Any]: Analysis results for this language
        """
        try:
            # Setup event loop if not available
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Initialize result
            result = None
                
            with tracing_v2_enabled():
                try:
                    # Log the brand name being analyzed
                    logger.info(f"Analyzing {self.language_name} translation for brand name: '{brand_name}'")
                    
                    # Format prompt with required variables
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
                    
                    # Create a unique call ID for rate limiting
                    call_id = f"{self.language_code}_translation_{brand_name}_{run_id[-8:]}"
                    
                    # Check if we need to wait due to rate limiting
                    wait_time = await google_api_limiter.wait_if_needed(call_id)
                    if wait_time > 0:
                        logger.info(f"Rate limited: Waited {wait_time:.2f}s before making {self.language_name} translation analysis call")
                    
                    # Get response from LLM
                    response = await self.llm.ainvoke(formatted_prompt)
                    
                    # Parse and process the response
                    try:
                        # Parse with structured output parser
                        analysis = self.output_parser.parse(response.content)
                        
                        # Create the result dictionary with standard required fields
                        result = {
                            "brand_name": brand_name,
                            "task_name": "translation_analysis",
                        }
                        
                        # Add analysis fields to result
                        for key, value in analysis.items():
                            if key != "run_id":  # Skip run_id to avoid LangGraph conflicts
                                result[key] = value
                        
                        # Ensure target_language is set correctly
                        result["target_language"] = self.language_name
                            
                        # Ensure rank is a float
                        if "rank" in result:
                            result["rank"] = float(result["rank"])
                        
                        # Apply language-specific validation
                        result = self._validate_language_specific_output(result)
                            
                        # Store in Supabase
                        await self._store_analysis(run_id, brand_name, result)
                        
                    except Exception as parse_error:
                        logger.error(f"Error parsing {self.language_name} translation analysis: {str(parse_error)}")
                        # Create a minimal valid result
                        result = {
                            "brand_name": brand_name,
                            "task_name": "translation_analysis",
                            "target_language": self.language_name,
                            "direct_translation": "Error analyzing translation",
                            "semantic_shift": "Error in analysis",
                            "pronunciation_difficulty": "Unknown due to error",
                            "phonetic_similarity_undesirable": False,
                            "phonetic_retention": "Unknown due to error",
                            "cultural_acceptability": "Unknown due to error",
                            "adaptation_needed": False,
                            "proposed_adaptation": "N/A - Error in analysis",
                            "brand_essence_preserved": "Unknown due to error",
                            "global_consistency_vs_localization": "Unknown due to error",
                            "notes": f"Error in analysis: {str(parse_error)}",
                            "rank": 5.0
                        }
                
                except Exception as e:
                    logger.error(f"Error in {self.language_name} translation analysis: {str(e)}")
                    # Create a fallback result
                    result = {
                        "brand_name": brand_name,
                        "task_name": "translation_analysis",
                        "target_language": self.language_name,
                        "direct_translation": "Error in analysis",
                        "semantic_shift": "Error in analysis",
                        "pronunciation_difficulty": "Unknown due to error",
                        "phonetic_retention": "Unknown due to error",
                        "global_consistency_vs_localization": "Unknown due to error",
                        "notes": f"Error during analysis: {str(e)}",
                        "rank": 5.0
                    }
            
            # Make sure result doesn't contain run_id before returning
            return {key: value for key, value in result.items() if key != "run_id"}
            
        except Exception as e:
            logger.error(
                f"Unexpected error in {self.language_name} translation analysis",
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
                "task_name": "translation_analysis",
                "target_language": self.language_name,
                "direct_translation": "Error in analysis",
                "semantic_shift": "Error in analysis",
                "pronunciation_difficulty": "Unknown due to error",
                "phonetic_retention": "Unknown due to error",
                "global_consistency_vs_localization": "Unknown due to error",
                "notes": f"Unexpected error in analysis: {str(e)}",
                "rank": 5.0
            }
    
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
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Define schema fields for the translation_analysis table
            schema_fields = [
                "target_language", "direct_translation", "semantic_shift", 
                "pronunciation_difficulty", "phonetic_similarity_undesirable", 
                "phonetic_retention", "cultural_acceptability", "adaptation_needed",
                "proposed_adaptation", "brand_essence_preserved", 
                "global_consistency_vs_localization", "notes", "rank"
            ]
            
            # Create the base record with required fields
            # CRITICAL: run_id, brand_name, and target_language are NOT NULL columns
            data = {
                "run_id": run_id,  # REQUIRED - NOT NULL in database
                "brand_name": brand_name,  # REQUIRED - NOT NULL in database
                "target_language": self.language_name  # REQUIRED - NOT NULL in database
            }
            
            # Ensure these fields are never null
            if not data["run_id"] or data["run_id"] == "":
                logger.error(f"Missing required run_id for {self.language_name} analysis!")
                return  # Skip this insert rather than sending invalid data
                
            if not data["brand_name"] or data["brand_name"] == "":
                logger.error(f"Missing required brand_name for {self.language_name} analysis!")
                return  # Skip this insert rather than sending invalid data
                
            if not data["target_language"] or data["target_language"] == "":
                logger.error(f"Missing required target_language for {self.language_name} analysis!")
                return  # Skip this insert rather than sending invalid data
            
            # Process each field according to its expected type in the schema
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
                    
                    # Skip overwriting target_language from analysis as we've already set it
                    if field == "target_language":
                        continue
                    
                    data[field] = value
            
            try:
                logger.debug(f"Inserting {self.language_name} analysis record: {data}")
                await self.supabase.table("translation_analysis").insert(data).execute()
                logger.info(f"Stored {self.language_name} translation analysis for brand name '{brand_name}' with run_id '{run_id}'")
            except Exception as e:
                logger.error(f"Failed to insert {self.language_name} record: {str(e)}")
                raise
            
        except APIError as e:
            logger.error(
                f"Supabase API error storing {self.language_name} translation analysis",
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
                f"Unexpected error storing {self.language_name} translation analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # Don't raise - allow the process to continue
    
    def _validate_language_specific_output(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and potentially modify output for language-specific requirements.
        
        This method should be overridden by language-specific subclasses
        to implement any language-specific validation or transformation.
        
        Args:
            analysis: The analysis output to validate
            
        Returns:
            Validated and potentially modified analysis
        """
        # Base implementation performs basic validation
        
        # Ensure target_language is set correctly
        if "target_language" in analysis:
            analysis["target_language"] = self.language_name
            
        return analysis 