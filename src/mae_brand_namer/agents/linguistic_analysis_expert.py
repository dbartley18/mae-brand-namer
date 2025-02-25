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
        
        # Load prompts from YAML files
        prompt_dir = Path(__file__).parent / "prompts" / "linguistics"
        self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(
                name="phonetic_analysis",
                description="Analysis of sound patterns and pronunciation",
                type="string"
            ),
            ResponseSchema(
                name="morphological_analysis", 
                description="Word structure and formation analysis",
                type="string"
            ),
            ResponseSchema(
                name="semantic_analysis",
                description="Meaning and associations analysis",
                type="string"
            ),
            ResponseSchema(
                name="pragmatic_analysis",
                description="Contextual usage and effectiveness",
                type="string"
            ),
            ResponseSchema(
                name="pronunciation_score",
                description="Ease of pronunciation score (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="memorability_score",
                description="Memorability assessment score (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="distinctiveness_score",
                description="Linguistic uniqueness score (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="cultural_fit_score",
                description="Cultural appropriateness score (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="overall_linguistic_score",
                description="Overall linguistic effectiveness (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="potential_issues",
                description="List of potential linguistic issues",
                type="array"
            ),
            ResponseSchema(
                name="recommendations",
                description="Linguistic optimization recommendations",
                type="array"
            )
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.output_schemas
        )
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None
        )
        
        # Create the prompt template with metadata
        system_message = SystemMessage(
            content=self.system_prompt.format(),
            additional_kwargs={
                "metadata": {
                    "agent_type": "linguistics_analyzer",
                    "methodology": "Alina Wheeler's Designing Brand Identity"
                }
            }
        )
        human_template = (
            "Analyze the linguistic characteristics of the following brand name:\n"
            "Brand Name: {brand_name}\n"
            "Brand Context: {brand_context}\n"
            "\nFormat your analysis according to this schema:\n"
            "{format_instructions}"
        )
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

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
            
        Raises:
            ValueError: If the analysis fails
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
                
                return analysis
                
        except Exception as e:
            logger.error(
                "Linguistic analysis failed",
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
        """Store linguistic analysis results in Supabase.
        
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
            
            await self.supabase.table("linguistic_analysis").insert(data).execute()
            
        except Exception as e:
            logger.error(
                "Error storing linguistic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise 