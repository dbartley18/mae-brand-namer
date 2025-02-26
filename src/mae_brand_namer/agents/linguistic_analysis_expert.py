"""Expert in linguistic analysis of brand names."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

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
                ResponseSchema(name="pronunciation_score", description="Score from 1-10 for pronunciation ease"),
                ResponseSchema(name="euphony_score", description="Score from 1-10 for phonetic pleasantness"),
                ResponseSchema(name="rhythm_and_meter", description="Analysis of rhythmic patterns"),
                ResponseSchema(name="phoneme_analysis", description="Analysis of individual sound units"),
                ResponseSchema(name="sound_symbolism", description="How sounds contribute to meaning"),
                ResponseSchema(name="word_formation", description="Morphological analysis of word structure"),
                ResponseSchema(name="grammatical_category", description="Part of speech and grammatical analysis"),
                ResponseSchema(name="semantic_analysis", description="Meaning analysis"),
                ResponseSchema(name="register_appropriateness", description="How well it fits intended contexts"),
                ResponseSchema(name="marketing_potential", description="Linguistic marketability assessment"),
                ResponseSchema(name="memorability_score", description="Score from 1-10 for ease of remembering"),
                ResponseSchema(name="notes", description="Additional linguistic observations"),
                ResponseSchema(name="rank", description="Overall linguistic effectiveness score (1-10)")
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
            
            # Set up the prompt template
            self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
            self.prompt = ChatPromptTemplate.from_template(self.analysis_prompt.template)
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.2,
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
        brand_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the linguistic characteristics of a brand name.
        
        Performs a comprehensive linguistic analysis including phonetics,
        morphology, semantics, and pragmatics. Evaluates pronunciation,
        memorability, and overall effectiveness.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Additional brand context information
            
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
                    brand_context=brand_context
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse and validate response
                analysis = self.output_parser.parse(response.content)
                
                # Store results
                await self._store_analysis(run_id, brand_name, analysis)
                
                return {
                    "brand_name": brand_name,
                    "pronunciation_ease": analysis["pronunciation_score"],
                    "euphony_vs_cacophony": analysis["euphony_score"],
                    "rhythm_and_meter": analysis["rhythm_and_meter"],
                    "phoneme_analysis": analysis["phoneme_analysis"],
                    "sound_symbolism": analysis["sound_symbolism"],
                    "word_formation": analysis["word_formation"],
                    "grammatical_category": analysis["grammatical_category"],
                    "semantic_analysis": analysis["semantic_analysis"],
                    "register_appropriateness": analysis["register_appropriateness"],
                    "marketing_potential": analysis["marketing_potential"],
                    "memorability_score": analysis["memorability_score"],
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
        try:
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "timestamp": datetime.now().isoformat(),
                **analysis
            }
            
            await self.supabase.table("linguistic_analysis").insert(data).execute()
            
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