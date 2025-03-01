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
        
        # Agent identity
        self.role = "Global Linguistic Adaptation & Translation Specialist"
        self.goal = (
            "Ensure effective translation of brand names across global markets "
            "while preserving meaning and avoiding negative connotations."
        )
        
        # Load prompts from YAML files
        prompt_dir = Path(__file__).parent / "prompts" / "translation_analysis"
        self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
        
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
        
        # Create the prompt template with metadata
        system_message = SystemMessage(
            content=self.system_prompt.format() + 
                "\n\nIMPORTANT: You must respond with a valid JSON object that matches EXACTLY the schema provided." +
                "\nThe JSON MUST contain all the fields specified below at the TOP LEVEL of the object." +
                "\nDo NOT nest fields under additional keys or create your own object structure." +
                "\nUse EXACTLY the field names provided in the schema - do not modify, merge, or rename any fields." +
                "\nDo not include any preamble or explanation outside the JSON structure." +
                "\nDo not use markdown formatting for the JSON." +
                "\n\nExample of correct structure:" +
                "\n{" +
                '\n  "target_language": "text here",' +
                '\n  "direct_translation": "text here",' +
                "\n  ... other fields exactly as named ..." +
                "\n}",
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
            "{format_instructions}\n\n"
            "Remember to respond with ONLY a valid JSON object exactly matching the required schema.\n"
            "Use the EXACT field names from the schema at the top level of your JSON response.\n"
            "Do NOT create nested structures or use different field names."
        )
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a brand name for translation considerations in major world languages.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Optional additional brand context information
            
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
                    brand_context=str(brand_context) if isinstance(brand_context, dict) else brand_context
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse and validate response
                analysis = self.output_parser.parse(response.content)
                
                # Store results
                await self._store_analysis(run_id, brand_name, analysis)
                
                # Create a result with brand_name but without run_id (which is handled by LangGraph)
                result = {
                    "brand_name": brand_name,
                    "task_name": "translation_analysis",
                }
                
                # Add all analysis results
                for key, value in analysis.items():
                    if key not in result and key != "run_id":  # Avoid duplicating keys or including run_id
                        result[key] = value
                
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