"""Survey Simulation Expert for simulating market research surveys."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class SurveySimulationExpert:
    """Expert in simulating market research surveys for brand name evaluation."""
    
    def __init__(self, supabase: SupabaseManager = None):
        """Initialize the SurveySimulationExpert with necessary configurations."""
        # Agent identity
        self.role = "Market Research & Consumer Insights Specialist"
        self.goal = """Simulate comprehensive market research surveys to evaluate brand name reception 
        across different consumer segments, measuring emotional impact, functional associations, and market potential."""
        self.backstory = """You are an expert market researcher specializing in brand name testing and 
        consumer insights. Your deep understanding of consumer psychology, market dynamics, and survey 
        methodology allows you to accurately simulate how different market segments would respond to 
        proposed brand names."""
        
        # Initialize retry configuration
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        
        # Initialize Supabase client
        self.supabase = supabase or SupabaseManager()
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if settings.tracing_enabled:
            self.tracer = LangChainTracer(project_name=settings.langsmith_project)
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.tracer] if self.tracer else None
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="persona_segment", description="Target audience segments analyzed", type="array"),
            ResponseSchema(name="brand_promise_perception_score", description="How well the name conveys brand promise (1-10)", type="number"),
            ResponseSchema(name="personality_fit_score", description="Alignment with brand personality (1-10)", type="number"),
            ResponseSchema(name="emotional_association", description="Key emotional responses", type="array"),
            ResponseSchema(name="functional_association", description="Practical/functional associations", type="array"),
            ResponseSchema(name="competitive_differentiation_score", description="Distinctiveness from competitors (1-10)", type="number"),
            ResponseSchema(name="psychometric_sentiment_mapping", description="Detailed sentiment analysis", type="object"),
            ResponseSchema(name="competitor_benchmarking_score", description="Performance vs competitors (1-10)", type="number"),
            ResponseSchema(name="simulated_market_adoption_score", description="Predicted market acceptance (1-10)", type="number"),
            ResponseSchema(name="qualitative_feedback_summary", description="Summary of simulated feedback"),
            ResponseSchema(name="raw_qualitative_feedback", description="Detailed feedback responses", type="object"),
            ResponseSchema(name="final_survey_recommendation", description="Final recommendation based on survey"),
            ResponseSchema(name="strategic_ranking", description="Overall strategic ranking (1-N)", type="integer")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Create the prompt template with metadata for LangGraph Studio
        system_message = SystemMessage(
            content=f"""You are a Survey Simulation Expert with the following profile:
            Role: {self.role}
            Goal: {self.goal}
            Backstory: {self.backstory}
            
            Simulate market research surveys for the provided brand name.
            Consider:
            1. Target Audience Segmentation
            2. Brand Promise & Personality Alignment
            3. Emotional & Functional Associations
            4. Competitive Differentiation
            5. Market Adoption Potential
            
            Format your response according to the following schema:
            {{format_instructions}}
            """,
            additional_kwargs={
                "metadata": {
                    "agent_type": "survey_simulator",
                    "methodology": "Alina Wheeler's Designing Brand Identity"
                }
            }
        )
        human_template = """Simulate market research survey results for the following brand name:
        Brand Name: {brand_name}
        Brand Context: {brand_context}
        Target Audience: {target_audience}
        Brand Values: {brand_values}
        Competitive Analysis: {competitive_analysis}
        """
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

    def simulate_survey(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Optional[Dict[str, Any]] = None,
        target_audience: Optional[str] = None,
        brand_values: Optional[List[str]] = None,
        competitive_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate market research survey results for a single brand name.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_name (str): The brand name to analyze
            brand_context (Optional[Dict[str, Any]]): Additional brand context
            target_audience (Optional[str]): Target audience description
            brand_values (Optional[List[str]]): List of brand values
            competitive_analysis (Optional[Dict[str, Any]]): Results from competitor analysis
            
        Returns:
            Dict[str, Any]: Survey simulation results
        """
        with tracing_enabled(tags={"agent": "SurveySimulationExpert", "run_id": run_id}):
            try:
                # Format the prompt with parser instructions
                formatted_prompt = self.prompt.format(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=json.dumps(brand_context) if brand_context else "{}",
                    target_audience=target_audience or "Not specified",
                    brand_values=", ".join(brand_values) if brand_values else "Not specified",
                    competitive_analysis=json.dumps(competitive_analysis) if competitive_analysis else "{}"
                )
                
                # Get response from LLM
                response = self.llm.invoke(formatted_prompt)
                
                # Parse the structured output
                parsed_output = self.output_parser.parse(response.content)
                
                # Add required fields
                simulation_results = {
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "timestamp": datetime.now().isoformat(),
                    **parsed_output
                }
                
                # Store in Supabase
                self._store_in_supabase(run_id, simulation_results)
                
                return simulation_results
                
            except Exception as e:
                error_msg = f"Error simulating survey for brand name '{brand_name}': {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _store_in_supabase(self, run_id: str, simulation_results: Dict[str, Any]) -> None:
        """
        Store the survey simulation results in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            simulation_results (Dict[str, Any]): The simulation results to store
        """
        try:
            # Ensure numeric fields are properly typed
            numeric_fields = [
                "brand_promise_perception_score", "personality_fit_score",
                "competitive_differentiation_score", "competitor_benchmarking_score",
                "simulated_market_adoption_score", "strategic_ranking"
            ]
            for field in numeric_fields:
                if field in simulation_results:
                    simulation_results[field] = float(simulation_results[field])
            
            # Ensure array fields are properly formatted
            array_fields = ["persona_segment", "emotional_association", "functional_association"]
            for field in array_fields:
                if field in simulation_results and isinstance(simulation_results[field], str):
                    simulation_results[field] = simulation_results[field].split(",")
            
            # Insert into survey_simulation table
            self.supabase.table("survey_simulation").insert(simulation_results).execute()
            
        except Exception as e:
            error_msg = f"Error storing survey simulation in Supabase: {str(e)}"
            logger.error(error_msg)
            raise  # Re-raise to handle in calling function 