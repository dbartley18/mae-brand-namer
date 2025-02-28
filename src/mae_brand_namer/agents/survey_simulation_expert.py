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
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tracers.context import tracing_v2_enabled

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class SurveySimulationExpert:
    """Expert in simulating market research surveys for brand name evaluation."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
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
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if settings.langchain_tracing_v2:
            self.tracer = LangChainTracer(project_name=settings.langsmith_project)
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
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
        with tracing_v2_enabled():
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
            
            # Filter to include only fields that exist in the database schema
            valid_fields = [
                "run_id", "brand_name", "timestamp", 
                "persona_segment", "emotional_association", "functional_association", 
                "likability_score", "memorability_score", "relevance_score", 
                "differentiation_score", "appeal_score", "overall_score", 
                "open_feedback", "purchase_intent", "recommendations"
            ]
            
            # First create a copy of the data with valid fields
            filtered_data = {k: v for k, v in simulation_results.items() if k in valid_fields}
            
            # Ensure array fields are properly formatted
            array_fields = ["persona_segment", "emotional_association", "functional_association"]
            for field in array_fields:
                if field in filtered_data and filtered_data[field]:
                    # If it's a string, convert from comma-separated string to proper PostgreSQL array format
                    if isinstance(filtered_data[field], str):
                        # Split by comma, strip whitespace, and filter out empty strings
                        items = [item.strip() for item in filtered_data[field].split(',') if item.strip()]
                        # Convert to PostgreSQL array format
                        filtered_data[field] = items
            
            # Insert into survey_simulation table
            self.supabase.table("survey_simulation").insert(filtered_data).execute()
            
        except Exception as e:
            error_msg = f"Error storing survey simulation in Supabase: {str(e)}"
            logger.error(error_msg)
            raise  # Re-raise to handle in calling function 