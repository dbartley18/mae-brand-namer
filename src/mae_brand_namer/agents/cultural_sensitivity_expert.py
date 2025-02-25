"""Cultural Sensitivity Expert for analyzing brand names across cultural contexts."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)

class CulturalSensitivityExpert:
    """Expert in analyzing cultural sensitivity and global market adaptation of brand names."""
    
    def __init__(self, dependencies: Dependencies):
        """Initialize the CulturalSensitivityExpert with dependencies."""
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        # Agent identity
        self.role = "Cultural Sensitivity & Global Market Adaptation Expert"
        self.goal = """Conduct an in-depth cultural sensitivity and localization analysis to ensure that proposed brand names 
        are globally acceptable, free from unintended connotations, and aligned with cultural, religious, and social norms."""
        self.backstory = """You are an expert in cross-cultural branding, linguistic anthropology, and global market adaptation. Your work 
        ensures that brand names resonate positively across diverse cultural contexts while avoiding unintended 
        associations, taboos, or offensive meanings."""
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="cultural_connotations", description="Cultural associations across target markets", type="string"),
            ResponseSchema(name="symbolic_meanings", description="Symbolic meanings in different cultures", type="string"),
            ResponseSchema(name="alignment_with_cultural_values", description="How well the name aligns with societal norms", type="string"),
            ResponseSchema(name="religious_sensitivities", description="Any unintended religious implications", type="string"),
            ResponseSchema(name="social_political_taboos", description="Potential sociopolitical sensitivities", type="string"),
            ResponseSchema(name="body_part_bodily_function_connotations", description="Unintended anatomical/physiological meanings", type="string"),
            ResponseSchema(name="age_related_connotations", description="Age-related perception of the name", type="string"),
            ResponseSchema(name="gender_connotations", description="Any unintentional gender bias", type="string"),
            ResponseSchema(name="regional_variations", description="Perception in different dialects and subcultures", type="string"),
            ResponseSchema(name="historical_meaning", description="Historical or traditional significance", type="string"),
            ResponseSchema(name="current_event_relevance", description="Connection to current events or trends", type="string"),
            ResponseSchema(name="overall_risk_rating", description="Overall cultural risk assessment", type="string"),
            ResponseSchema(name="notes", description="Additional cultural observations", type="string"),
            ResponseSchema(name="rank", description="Overall cultural sensitivity score (1-10)", type="number")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Create the prompt template with metadata for LangGraph Studio
        system_message = SystemMessage(
            content=f"""You are a Cultural Sensitivity Expert with the following profile:
            Role: {self.role}
            Goal: {self.goal}
            Backstory: {self.backstory}
            
            Analyze the provided brand name based on its cultural implications across global markets.
            Consider:
            1. Cultural Connotations & Symbolic Meanings
            2. Religious & Social Sensitivities
            3. Demographic & Regional Interpretations
            4. Historical & Current Context
            5. Overall Cultural Risk Assessment
            
            Format your response according to the following schema:
            {{format_instructions}}
            """,
            additional_kwargs={
                "metadata": {
                    "agent_type": "cultural_analyzer",
                    "methodology": "Alina Wheeler's Designing Brand Identity"
                }
            }
        )
        human_template = """Analyze the cultural sensitivity of the following brand name:
        Brand Name: {brand_name}
        Brand Context: {brand_context}
        """
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
        """
        Analyze the cultural sensitivity of a single brand name.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_name (str): The brand name to analyze
            brand_context (Dict[str, Any]): Additional brand context
            
        Returns:
            Dict[str, Any]: Analysis results
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            with tracing_enabled(tags={"agent": "CulturalSensitivityExpert", "run_id": run_id}):
                # Format prompt with parser instructions
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=json.dumps(brand_context, indent=2)
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse structured response
                analysis_result = self.output_parser.parse(response.content)
                
                # Store results in Supabase
                await self._store_analysis(run_id, brand_name, analysis_result)
                
                return {
                    "cultural_analysis": {
                        "brand_name": brand_name,
                        **analysis_result
                    }
                }
                
        except Exception as e:
            logger.error(
                "Error in cultural sensitivity analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise ValueError(f"Cultural sensitivity analysis failed: {str(e)}")
    
    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Store cultural sensitivity analysis results in Supabase."""
        try:
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "timestamp": datetime.now().isoformat(),
                **analysis
            }
            
            await self.supabase.table("cultural_sensitivity_analysis").insert(data).execute()
            
        except Exception as e:
            logger.error(
                "Error storing cultural sensitivity analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise 