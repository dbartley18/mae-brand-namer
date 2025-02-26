"""Market research expert for evaluating brand names."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from langchain.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.messages import HumanMessage, SystemMessage
from postgrest.exceptions import APIError

from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.settings import settings

logger = get_logger(__name__)

class MarketResearchExpert:
    """Expert in market research and brand name evaluation."""
    
    def __init__(self, supabase: SupabaseManager = None):
        """Initialize the MarketResearchExpert with necessary configurations."""
        # Initialize Supabase client
        self.supabase = supabase or SupabaseManager()
        
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "market_research"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.research_prompt = load_prompt(str(prompt_dir / "research.yaml"))
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.3,  # Lower temperature for more consistent analysis
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="market_opportunity", description="Analysis of market opportunity and potential"),
            ResponseSchema(name="target_audience_fit", description="Analysis of fit with target audience"),
            ResponseSchema(name="competitive_analysis", description="Analysis of competitive landscape"),
            ResponseSchema(name="market_viability", description="Assessment of market viability"),
            ResponseSchema(name="market_opportunity_score", description="Score from 1-10 for market opportunity", type="number"),
            ResponseSchema(name="target_audience_score", description="Score from 1-10 for target audience fit", type="number"),
            ResponseSchema(name="competitive_advantage_score", description="Score from 1-10 for competitive advantage", type="number"),
            ResponseSchema(name="scalability_score", description="Score from 1-10 for scalability potential", type="number"),
            ResponseSchema(name="overall_market_score", description="Overall market viability score from 1-10", type="number"),
            ResponseSchema(name="potential_risks", description="List of potential market risks"),
            ResponseSchema(name="recommendations", description="Strategic recommendations")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)

    async def analyze_market_potential(
            self,
            run_id: str,
            brand_names: List[str],
            brand_context: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
        """
        Analyze market potential for brand names.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_names (List[str]): List of brand names to analyze
            brand_context (Dict[str, Any]): Brand context information
            
        Returns:
            List[Dict[str, Any]]: List of market analyses for each brand name
            
        Raises:
            ValueError: If analysis fails
            APIError: If Supabase operation fails
        """
        analyses = []
        timestamp = datetime.now().isoformat()
        
        try:
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format())
            
            with tracing_enabled(
                tags={
                    "agent": "MarketResearchExpert",
                    "task": "analyze_market_potential",
                    "run_id": run_id,
                    "prompt_type": "market_research"
                }
            ):
                for brand_name in brand_names:
                    # Format the research prompt
                    formatted_prompt = self.research_prompt.format(
                        format_instructions=self.output_parser.get_format_instructions(),
                        brand_names=[brand_name],
                        brand_context=brand_context
                    )
                    
                    # Create human message
                    human_message = HumanMessage(content=formatted_prompt)
                    
                    # Get response from LLM
                    response = await self.llm.ainvoke([system_message, human_message])
                    
                    # Parse the structured output
                    parsed_output = self.output_parser.parse(response.content)
                    parsed_output.update({
                        "run_id": run_id,
                        "brand_name": brand_name,
                        "timestamp": timestamp,
                        "version": "1.0"
                    })
                    
                    # Store in Supabase
                    await self._store_in_supabase(run_id, parsed_output)
                    analyses.append(parsed_output)
                
                return analyses
                
        except Exception as e:
            logger.error(
                "Error analyzing market potential",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "brand_names": brand_names
                }
            )
            raise ValueError(f"Failed to analyze market potential: {str(e)}")

    async def _store_in_supabase(self, run_id: str, analysis_data: Dict[str, Any]) -> None:
        """
        Store the market research analysis in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            analysis_data (Dict[str, Any]): The analysis data to store
            
        Raises:
            PostgrestError: If there's an error with the Supabase query
            APIError: If there's an API-level error with Supabase
            ValueError: If there's an error with data validation or preparation
        """
        try:
            # Prepare data for Supabase
            supabase_data = {
                "run_id": run_id,
                "brand_name": analysis_data["brand_name"],
                "market_opportunity": analysis_data["market_opportunity"],
                "target_audience_fit": analysis_data["target_audience_fit"],
                "competitive_analysis": analysis_data["competitive_analysis"],
                "market_viability": analysis_data["market_viability"],
                "market_opportunity_score": float(analysis_data["market_opportunity_score"]),
                "target_audience_score": float(analysis_data["target_audience_score"]),
                "competitive_advantage_score": float(analysis_data["competitive_advantage_score"]),
                "scalability_score": float(analysis_data["scalability_score"]),
                "overall_market_score": float(analysis_data["overall_market_score"]),
                "potential_risks": analysis_data["potential_risks"],
                "recommendations": analysis_data["recommendations"],
                "timestamp": analysis_data["timestamp"],
                "version": analysis_data.get("version", "1.0")
            }
            
            # Store in Supabase using the singleton client
            await self.supabase.execute_with_retry(
                operation="insert",
                table="market_research",
                data=supabase_data
            )
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "Error preparing market research data for Supabase",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "data": analysis_data
                }
            )
            raise ValueError(f"Error preparing market research data: {str(e)}")
            
        except (APIError) as e:
            logger.error(
                "Error storing market research in Supabase",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "data": supabase_data
                }
            )
            raise 