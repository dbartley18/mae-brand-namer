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
from langchain_core.tools import Tool

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
                name="competitor_name",
                description="Name of the primary competitor being analyzed",
                type="string"
            ),
            ResponseSchema(
                name="competitor_naming_style",
                description="Whether competitors use descriptive, abstract, or other naming styles",
                type="string"
            ),
            ResponseSchema(
                name="competitor_keywords",
                description="Common words or themes in competitor brand names",
                type="string"
            ),
            ResponseSchema(
                name="competitor_positioning",
                description="How competitors position their brands in the market",
                type="string"
            ),
            ResponseSchema(
                name="competitor_strengths",
                description="Strengths of competitor brand names",
                type="string"
            ),
            ResponseSchema(
                name="competitor_weaknesses",
                description="Weaknesses of competitor brand names",
                type="string"
            ),
            ResponseSchema(
                name="competitor_differentiation_opportunity",
                description="How to create differentiation from competitors",
                type="string"
            ),
            ResponseSchema(
                name="differentiation_score",
                description="Quantified differentiation from competitors (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="risk_of_confusion",
                description="Likelihood of brand confusion with competitors",
                type="string"
            ),
            ResponseSchema(
                name="target_audience_perception",
                description="How consumers may compare this name to competitors",
                type="string"
            ),
            ResponseSchema(
                name="competitive_advantage_notes",
                description="Any competitive advantages of the brand name",
                type="string"
            ),
            ResponseSchema(
                name="trademark_conflict_risk",
                description="Potential conflicts with existing trademarks",
                type="string"
            )
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.output_schemas
        )
        
        # Create Google Search tool
        search_tool = {
            "type": "google_search",
            "google_search": {}
        }
        
        # Initialize Gemini model with tracing and Google Search tool
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=settings.model_temperature,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks(),
            tools=[search_tool]
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
            # Map the analysis fields to the Supabase schema fields
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "competitor_name": analysis.get("competitor_name", ""),
                "competitor_naming_style": analysis.get("competitor_naming_style", ""),
                "competitor_keywords": analysis.get("competitor_keywords", ""),
                "competitor_positioning": analysis.get("competitor_positioning", ""),
                "competitor_strengths": analysis.get("competitor_strengths", ""),
                "competitor_weaknesses": analysis.get("competitor_weaknesses", ""),
                "competitor_differentiation_opportunity": analysis.get("competitor_differentiation_opportunity", ""),
                "differentiation_score": analysis.get("differentiation_score", 0),
                "risk_of_confusion": analysis.get("risk_of_confusion", ""),
                "target_audience_perception": analysis.get("target_audience_perception", ""),
                "competitive_advantage_notes": analysis.get("competitive_advantage_notes", ""),
                "trademark_conflict_risk": analysis.get("trademark_conflict_risk", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # Ensure differentiation_score is a float between 0 and 10
            if "differentiation_score" in data:
                try:
                    score = float(data["differentiation_score"])
                    data["differentiation_score"] = max(0, min(10, score))
                except (ValueError, TypeError):
                    data["differentiation_score"] = 0
                    logger.warning(f"Invalid differentiation_score value: {data['differentiation_score']}, defaulting to 0")
            
            # Log the data being stored
            logger.info(f"Storing competitor analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
            await self.supabase.table("competitor_analysis").insert(data).execute()
            logger.info(f"Successfully stored competitor analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
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