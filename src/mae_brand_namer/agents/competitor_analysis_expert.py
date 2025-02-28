"""Expert in analyzing brand names in the context of market competition."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.messages import HumanMessage, SystemMessage

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)


class CompetitorAnalysisExpert:
    """Expert in analyzing brand names in the context of market competition.
    
    This expert evaluates brand names against market competitors, analyzing
    differentiation, positioning, and competitive advantage potential.
    
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
        """Initialize the CompetitorAnalysisExpert with dependencies.
        
        Args:
            dependencies: Container for application dependencies
        """
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        # Agent identity
        self.role = "Competitor Analysis & Market Positioning Expert"
        self.goal = (
            "Evaluate brand names in the context of market competition, analyzing "
            "differentiation, positioning, and competitive advantage potential."
        )
        
        # Load prompts from YAML files
        prompt_dir = Path(__file__).parent / "prompts" / "competitor_analysis"
        self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(
                name="market_positioning",
                description="Analysis of market positioning potential",
                type="string"
            ),
            ResponseSchema(
                name="competitor_similarity",
                description="Similarity to competitor brands",
                type="string"
            ),
            ResponseSchema(
                name="differentiation_potential",
                description="Potential for market differentiation",
                type="string"
            ),
            ResponseSchema(
                name="brand_strength",
                description="Relative brand strength assessment",
                type="string"
            ),
            ResponseSchema(
                name="market_share_potential",
                description="Potential market share capture",
                type="string"
            ),
            ResponseSchema(
                name="competitive_advantage",
                description="Sources of competitive advantage",
                type="string"
            ),
            ResponseSchema(
                name="market_barriers",
                description="Barriers to market entry",
                type="string"
            ),
            ResponseSchema(
                name="trademark_risk",
                description="Risk of trademark conflicts",
                type="string"
            ),
            ResponseSchema(
                name="brand_confusion_risk",
                description="Risk of brand confusion",
                type="string"
            ),
            ResponseSchema(
                name="market_readiness",
                description="Market readiness assessment",
                type="string"
            ),
            ResponseSchema(
                name="competitor_recommendations",
                description="Strategic recommendations",
                type="array"
            ),
            ResponseSchema(
                name="competitive_score",
                description="Overall competitive strength score (1-10)",
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
            callbacks=settings.get_langsmith_callbacks()
        )
        
        # Create the prompt template with metadata
        system_message = SystemMessage(
            content=self.system_prompt.format(),
            additional_kwargs={
                "metadata": {
                    "agent_type": "competitor_analyzer",
                    "methodology": "Market Competition Framework"
                }
            }
        )
        human_template = (
            "Analyze the competitive positioning of the following brand name:\n"
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
        """Analyze the competitive positioning of a brand name.
        
        Evaluates the brand name against market competitors, analyzing
        differentiation, positioning, and competitive advantage potential.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Additional brand context information
            
        Returns:
            Dictionary containing the competitive analysis results
            
        Raises:
            ValueError: If the analysis fails
        """
        try:
            with tracing_enabled(
                tags={
                    "agent": "CompetitorAnalysisExpert",
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
                "Competitor analysis failed",
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
        """Store competitor analysis results in Supabase.
        
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
            
            await self.supabase.table("competitor_analysis").insert(data).execute()
            logger.info(f"Stored competitor analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing competitor analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise 