"""Cultural Sensitivity Expert for analyzing brand names across cultural contexts."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from postgrest import APIError

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)

class CulturalSensitivityExpert:
    """Expert in analyzing brand names for cultural sensitivity and appropriateness."""
    
    def __init__(self, dependencies: Dependencies):
        """Initialize the CulturalSensitivityExpert."""
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        try:
            # Load prompts
            prompt_dir = Path(__file__).parent / "prompts" / "cultural_sensitivity"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            
            # Define output schemas for structured parsing
            self.output_schemas = [
                ResponseSchema(name="cultural_connotations", description="Global cultural associations", type="string"),
                ResponseSchema(name="symbolic_meanings", description="Symbolic interpretations across cultures", type="string"),
                ResponseSchema(name="alignment_with_cultural_values", description="How well it aligns with cultural values", type="string"),
                ResponseSchema(name="religious_sensitivities", description="Religious associations or concerns", type="string"),
                ResponseSchema(name="social_political_taboos", description="Social or political issues to consider", type="string"),
                ResponseSchema(name="body_part_bodily_function_connotations", description="Whether it has associations with body parts or functions", type="boolean"),
                ResponseSchema(name="age_related_connotations", description="Age-related considerations", type="string"),
                ResponseSchema(name="gender_connotations", description="Gender associations or bias", type="string"),
                ResponseSchema(name="regional_variations", description="How meanings vary by region", type="string"),
                ResponseSchema(name="historical_meaning", description="Historical associations or context", type="string"),
                ResponseSchema(name="current_event_relevance", description="Relevance to current events", type="string"),
                ResponseSchema(name="overall_risk_rating", description="Overall risk rating (Low/Medium/High)", type="string"),
                ResponseSchema(name="notes", description="Additional cultural sensitivity observations", type="string"),
                ResponseSchema(name="rank", description="Overall cultural sensitivity score (1-10)", type="number")
            ]
            
            self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
            
            # Create prompt template
            system_message = SystemMessage(content=self.system_prompt.format() + 
                "\n\nIMPORTANT: You must respond with a valid JSON object that matches EXACTLY the schema provided." +
                "\nThe JSON MUST contain all the fields specified below at the TOP LEVEL of the object." +
                "\nDo NOT nest fields under additional keys or create your own object structure." +
                "\nUse EXACTLY the field names provided in the schema - do not modify, merge, or rename any fields." +
                "\nDo not include any preamble or explanation outside the JSON structure." +
                "\nDo not use markdown formatting for the JSON." +
                "\n\nExample of correct structure:" +
                "\n{" +
                '\n  "cultural_connotations": "text here",' +
                '\n  "symbolic_meanings": "text here",' +
                "\n  ... other fields exactly as named ..." +
                "\n}")
            human_template = (
                "Analyze the cultural sensitivity of the following brand name:\n"
                "Brand Name: {brand_name}\n"
                "Brand Context: {brand_context}\n\n"
                "Format your analysis according to this schema:\n"
                "{format_instructions}\n\n"
                "Remember to respond with ONLY a valid JSON object exactly matching the required schema.\n"
                "Use the EXACT field names from the schema at the top level of your JSON response.\n"
                "Do NOT create nested structures or use different field names."
            )
            self.prompt = ChatPromptTemplate.from_messages([
                system_message,
                HumanMessage(content=human_template)
            ])
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                temperature=0.5,
                google_api_key=settings.google_api_key,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.error(
                "Error initializing CulturalSensitivityExpert",
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
        """Analyze a brand name for cultural sensitivity concerns.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Brand context information
            
        Returns:
            Dictionary with cultural sensitivity analysis results
        """
        try:
            with tracing_enabled(
                tags={
                    "agent": "CulturalSensitivityExpert",
                    "run_id": run_id
                }
            ):
                # Format the prompt
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=str(brand_context) if isinstance(brand_context, dict) else brand_context
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse the response
                analysis = self.output_parser.parse(response.content)
                
                # Store results
                await self._store_analysis(run_id, brand_name, analysis)
                
                # Return the analysis
                return {
                    "brand_name": brand_name,
                    "task_name": "cultural_sensitivity_analysis",
                    "cultural_connotations": analysis["cultural_connotations"],
                    "symbolic_meanings": analysis["symbolic_meanings"],
                    "alignment_with_cultural_values": analysis["alignment_with_cultural_values"],
                    "religious_sensitivities": analysis["religious_sensitivities"],
                    "social_political_taboos": analysis["social_political_taboos"],
                    "body_part_bodily_function_connotations": analysis["body_part_bodily_function_connotations"],
                    "age_related_connotations": analysis["age_related_connotations"],
                    "gender_connotations": analysis["gender_connotations"],
                    "regional_variations": analysis["regional_variations"],
                    "historical_meaning": analysis["historical_meaning"],
                    "current_event_relevance": analysis["current_event_relevance"],
                    "overall_risk_rating": analysis["overall_risk_rating"],
                    "notes": analysis["notes"],
                    "rank": float(analysis["rank"])
                }
                
        except APIError as e:
            logger.error(
                "Supabase API error in cultural sensitivity analysis",
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
                "Error in cultural sensitivity analysis",
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
        """Store cultural sensitivity analysis results in Supabase.
        
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
            
            await self.supabase.table("cultural_sensitivity_analysis").insert(data).execute()
            logger.info(f"Stored cultural sensitivity analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except APIError as e:
            logger.error(
                "Supabase API error storing cultural sensitivity analysis",
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
                "Unexpected error storing cultural sensitivity analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise 