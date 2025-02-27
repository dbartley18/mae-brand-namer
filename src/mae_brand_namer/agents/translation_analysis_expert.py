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

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies
from ..models.app_config import AppConfig

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
    
    def __init__(self, dependencies=None, supabase=None, app_config: AppConfig = None):
        """Initialize the TranslationAnalysisExpert with dependencies."""
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase
            self.langsmith = None

        # Get agent-specific configuration
        self.app_config = app_config or AppConfig()
        agent_name = "translation_analysis_expert"
        self.temperature = self.app_config.get_temperature_for_agent(agent_name)
        
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "translation_analysis"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            
            # Define output schemas for structured parsing
            self.output_schemas = [
                ResponseSchema(
                    name="direct_translation",
                    description="Direct translation analysis",
                    type="string"
                ),
                ResponseSchema(
                    name="semantic_shift",
                    description="Analysis of meaning changes across languages",
                    type="string"
                ),
                ResponseSchema(
                    name="pronunciation_difficulty",
                    description="Assessment of pronunciation challenges",
                    type="string"
                ),
                ResponseSchema(
                    name="cultural_implications",
                    description="Cultural context and implications",
                    type="string"
                ),
                ResponseSchema(
                    name="localization_needs",
                    description="Required adaptations for local markets",
                    type="array"
                ),
                ResponseSchema(
                    name="market_specific_issues",
                    description="Issues in specific target markets",
                    type="array"
                ),
                ResponseSchema(
                    name="global_viability_score",
                    description="Overall global viability score (1-10)",
                    type="number"
                ),
                ResponseSchema(
                    name="language_specific_scores",
                    description="Viability scores by language",
                    type="object"
                ),
                ResponseSchema(
                    name="adaptation_recommendations",
                    description="Recommendations for global adaptation",
                    type="array"
                ),
                ResponseSchema(
                    name="risk_assessment",
                    description="Potential risks in translation/adaptation",
                    type="array"
                )
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(
                self.output_schemas
            )
            
            # Create the prompt template with metadata
            system_message = SystemMessage(
                content=self.system_prompt.format(),
                additional_kwargs={
                    "metadata": {
                        "agent_type": "translation_analyzer",
                        "methodology": "Global Brand Adaptation Framework"
                    }
                }
            )
            human_template = (
                "Analyze the translation and global market adaptation needs for the "
                "following brand name:\n"
                "Brand Name: {brand_name}\n"
                "Brand Context: {brand_context}\n"
                "\nFormat your analysis according to this schema:\n"
                "{format_instructions}"
            )
            self.prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessage(content=human_template)
            ])
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
        # Initialize Gemini model with agent-specific temperature
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=self.temperature,
            google_api_key=os.getenv("GEMINI_API_KEY") or settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None
        )
        
        # Log the temperature setting being used
        logger.info(
            f"Initialized Translation Analysis Expert with temperature: {self.temperature}",
            extra={"agent": agent_name, "temperature": self.temperature}
        )

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the translation implications of a brand name.
        
        Evaluates how the brand name translates across different languages and
        cultures, identifying potential issues and adaptation needs.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Additional brand context information
            
        Returns:
            Dictionary containing the translation analysis results
            
        Raises:
            ValueError: If the analysis fails
        """
        try:
            with tracing_enabled(
                tags={
                    "agent": "TranslationAnalysisExpert",
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
                
                return analysis
                
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
            
            await self.supabase.table("translation_analysis").insert(data).execute()
            logger.info(f"Stored translation analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing translation analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise 