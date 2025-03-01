"""Expert in linguistic analysis of brand names."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.messages import HumanMessage, SystemMessage
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
            
            # Set up the prompt template
            self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
            self.prompt = ChatPromptTemplate.from_template(self.analysis_prompt.template)
            
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
            with tracing_enabled(
                tags={
                    "agent": "LinguisticsExpert",
                    "run_id": run_id
                }
            ):
                # Format prompt with parser instructions
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=str(brand_context) if isinstance(brand_context, dict) else brand_context
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse and validate response
                analysis = self.output_parser.parse(response.content)
                
                # Store results
                await self._store_analysis(run_id, brand_name, analysis)
                
                return {
                    "brand_name": brand_name,
                    "task_name": "linguistic_analysis",
                    "pronunciation_ease": analysis["pronunciation_ease"],
                    "euphony_vs_cacophony": analysis["euphony_vs_cacophony"],
                    "rhythm_and_meter": analysis["rhythm_and_meter"],
                    "phoneme_frequency_distribution": analysis["phoneme_frequency_distribution"],
                    "sound_symbolism": analysis["sound_symbolism"],
                    "word_class": analysis["word_class"],
                    "morphological_transparency": analysis["morphological_transparency"],
                    "grammatical_gender": analysis["grammatical_gender"],
                    "inflectional_properties": analysis["inflectional_properties"],
                    "ease_of_marketing_integration": analysis["ease_of_marketing_integration"],
                    "naturalness_in_collocations": analysis["naturalness_in_collocations"],
                    "homophones_homographs": analysis["homophones_homographs"],
                    "semantic_distance_from_competitors": analysis["semantic_distance_from_competitors"],
                    "neologism_appropriateness": analysis["neologism_appropriateness"],
                    "overall_readability_score": analysis["overall_readability_score"],
                    "notes": analysis["notes"],
                    "rank": float(analysis["rank"])
                }
                
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
            raise
            
        except Exception as e:
            logger.error(
                "Error in linguistic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
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
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "timestamp": datetime.now().isoformat(),
                **analysis
            }
            
            await self.supabase.table("linguistic_analysis").insert(data).execute()
            logger.info(f"Stored linguistic analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except APIError as e:
            logger.error(
                "Supabase API error storing linguistic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error storing linguistic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise 