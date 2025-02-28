"""Semantic Analysis Expert for analyzing brand name meanings and associations."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio
from pathlib import Path

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
        
        # Create the prompt template with metadata for LangGraph Studio
        system_message = SystemMessage(
            content=self.system_prompt.format(
                format_instructions=self.output_parser.get_format_instructions()
            ),
            additional_kwargs={
                "metadata": {
                    "agent_type": "semantic_analyzer",
                    "methodology": "Alina Wheeler's Designing Brand Identity"
                }
            }
        )
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=self.human_prompt.template)
        ])

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a brand name's semantic meaning and store results.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_name (str): The brand name to analyze
            brand_context (Dict[str, Any]): Brand context information
            
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
                
            with tracing_v2_enabled():
                # Format prompt with variables
                formatted_prompt = self.prompt.format_messages(
                    brand_name=brand_name,
                    brand_context=str(brand_context) if isinstance(brand_context, dict) else brand_context,
                    format_instructions=self.output_parser.get_format_instructions()
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse structured response
                analysis_result = self.output_parser.parse(response.content)
                
                # Store results in Supabase
                await self._store_analysis(run_id, brand_name, analysis_result)
                
                # Create a result dictionary with brand_name (without duplicating run_id since that's handled by LangGraph)
                result = {
                    "brand_name": brand_name,
                    "task_name": "semantic_analysis",
                }
                
                # Add all analysis results, avoiding duplicate keys
                for key, value in analysis_result.items():
                    if key not in result:  # Avoid duplicating keys
                        result[key] = value
                
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
            raise ValueError(f"Semantic analysis failed: {str(e)}")
    
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
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "task_name": "semantic_analysis",
                "timestamp": datetime.now().isoformat(),
                **analysis
            }
            
            await self.supabase.table("semantic_analysis").insert(data).execute()
            logger.info(f"Stored semantic analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing semantic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise 