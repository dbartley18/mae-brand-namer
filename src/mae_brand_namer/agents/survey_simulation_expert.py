"""Survey Simulation Expert for simulating market research surveys."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio
import re
from pathlib import Path
import random

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tracers.context import tracing_v2_enabled

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.dependencies import Dependencies

logger = get_logger(__name__)

class SurveySimulationExpert:
    """Expert in simulating market research surveys for brand name evaluation using synthetic personas."""
    
    def __init__(self, dependencies: Dependencies = None):
        """
        Initialize the SurveySimulationExpert with necessary dependencies.

        Args:
            dependencies (Dependencies, optional): Dependencies injection container.
        """
        # Set up dependencies
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.dependencies = Dependencies()
            self.supabase = self.dependencies.supabase
            self.langsmith = None
        
        # Agent identity
        self.role = "Market Research and Consumer Insights Specialist"
        self.goal = """Simulate high-precision, AI-driven market research that mirrors enterprise-level survey methodologies, 
        generating deep, behavioral insights on how shortlisted brand names resonate across segmented target audiences. 
        Provide data-backed, statistically robust name performance assessments that drive high-stakes branding decisions."""
        self.backstory = """You are the pinnacle of AI-driven market research—a leading expert in enterprise brand analysis and consumer 
        perception modeling. With mastery in synthetic persona simulation, psychometric analysis, and AI-enhanced survey 
        methodologies, you go far beyond traditional market research firms. 
        
        Instead of basic questionnaires, you orchestrate AI-powered simulations using behaviorally programmed synthetic 
        personas that think, react, and evaluate brand names as the brands targeted real-world consumers would. Your simulations generate 
        scalable, statistically significant insights, producing not just data, but actionable intelligence that can 
        influence multi-million-dollar branding decisions.

        You are guided by Alina Wheeler's 'Designing Brand Identity' methodology, ensuring every evaluation is aligned 
        with brand promise, personality, and strategic positioning. Your goal is to deliver enterprise-grade, 
        high-confidence brand perception analysis, helping decision-makers identify names that dominate their category 
        and create lasting market impact."""
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            self.tracer = LangChainTracer(
                project_name=os.getenv("LANGCHAIN_PROJECT", "mae-brand-namer")
            )
            
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="brand_name", description="The brand name being evaluated"),
            ResponseSchema(name="persona_segment", description="Array of target audience segments analyzed (e.g. ['Young Professionals', 'Small Business Owners'])"),
            ResponseSchema(name="brand_promise_perception_score", description="How well the name conveys brand promise (0-10 scale)"),
            ResponseSchema(name="personality_fit_score", description="Alignment with brand personality (0-10 scale)"),
            ResponseSchema(name="emotional_association", description="Array of key emotional responses evoked by the name"),
            ResponseSchema(name="functional_association", description="Array of practical/functional associations with the name"),
            ResponseSchema(name="competitive_differentiation_score", description="Distinctiveness from competitors (0-10 scale)"),
            ResponseSchema(name="psychometric_sentiment_mapping", description="JSON mapping emotional dimensions to scores"),
            ResponseSchema(name="competitor_benchmarking_score", description="Performance vs competitors (0-10 scale)"),
            ResponseSchema(name="simulated_market_adoption_score", description="Predicted market acceptance (0-10 scale)"),
            ResponseSchema(name="qualitative_feedback_summary", description="Summary of qualitative feedback from simulated personas"),
            ResponseSchema(name="raw_qualitative_feedback", description="JSON object with detailed feedback from different personas"),
            ResponseSchema(name="final_survey_recommendation", description="Final recommendation based on survey results"),
            ResponseSchema(name="strategic_ranking", description="Overall strategic ranking (1=best)"),
            ResponseSchema(name="individual_personas", description="Array of individual persona responses with demographic details and feedback"),
            
            # Company-related fields
            ResponseSchema(name="industry", description="Industry the brand operates in"),
            ResponseSchema(name="company_size_employees", description="Number of employees at the company"),
            ResponseSchema(name="company_size_revenue", description="Annual revenue of the company in millions"),
            ResponseSchema(name="market_share", description="Estimated market share percentage"),
            ResponseSchema(name="company_structure", description="Organizational structure (e.g., Hierarchical, Flat, Matrix)"),
            ResponseSchema(name="geographic_location", description="Primary geographic location of operations"),
            ResponseSchema(name="technology_stack", description="Description of technology used"),
            ResponseSchema(name="company_growth_stage", description="Growth stage (e.g., Startup, Growth, Mature)"),
            ResponseSchema(name="company_culture", description="Type of company culture"),
            ResponseSchema(name="financial_stability", description="Assessment of financial stability"),
            ResponseSchema(name="supply_chain", description="Description of supply chain"),
            ResponseSchema(name="legal_regulatory_environment", description="Regulatory environment affecting the brand"),
            
            # Persona-related fields
            ResponseSchema(name="job_title", description="Typical job title of target persona"),
            ResponseSchema(name="seniority", description="Seniority level (e.g., Entry-level, Mid-level, Executive)"),
            ResponseSchema(name="years_of_experience", description="Average years of professional experience"),
            ResponseSchema(name="department", description="Department within organization"),
            ResponseSchema(name="education_level", description="Highest education level attained"),
            ResponseSchema(name="goals_and_challenges", description="Key professional goals and challenges"),
            ResponseSchema(name="values_and_priorities", description="Core values and priorities"),
            ResponseSchema(name="decision_making_style", description="Approach to decision making"),
            ResponseSchema(name="information_sources", description="Primary sources of information"),
            ResponseSchema(name="communication_preferences", description="Preferred communication channels"),
            ResponseSchema(name="pain_points", description="Primary pain points related to products/services"),
            ResponseSchema(name="technological_literacy", description="Level of comfort with technology"),
            ResponseSchema(name="attitude_towards_risk", description="Risk tolerance level"),
            ResponseSchema(name="purchasing_behavior", description="Habits related to making purchases"),
            ResponseSchema(name="online_behavior", description="Online activity patterns"),
            ResponseSchema(name="interaction_with_brand", description="How they typically interact with brands"),
            ResponseSchema(name="professional_associations", description="Professional groups or associations"),
            ResponseSchema(name="technical_skills", description="Key technical skills"),
            ResponseSchema(name="language", description="Primary language"),
            ResponseSchema(name="learning_style", description="Preferred way of learning"),
            ResponseSchema(name="networking_habits", description="Approach to professional networking"),
            ResponseSchema(name="professional_aspirations", description="Career goals and aspirations"),
            ResponseSchema(name="influence_within_company", description="Level of influence in decision making"),
            ResponseSchema(name="channel_preferences", description="Preferred marketing/sales channels"),
            ResponseSchema(name="event_attendance", description="Types of events they attend"),
            ResponseSchema(name="content_consumption_habits", description="How they consume content"),
            ResponseSchema(name="vendor_relationship_preferences", description="Preferred vendor relationship style")
        ]
        
        # Configure the output parser
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Load prompts from template files
        self._load_prompts()
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )

    def _load_prompts(self):
        """Load prompts from template files."""
        try:
            # Get path to prompt templates
            prompt_dir = Path(__file__).parent / "prompts" / "survey_simulation"
            
            # Load prompt templates
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.simulation_prompt = load_prompt(str(prompt_dir / "simulation.yaml"))
            
            # Get format instructions from the output parser
            self.format_instructions = self.output_parser.get_format_instructions()
            
            # Create the prompt template
            system_message = SystemMessage(content=self.system_prompt.template)
            human_template = self.simulation_prompt.template
            
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_message.content),
                ("human", human_template)
            ])
            
            # Log information about the loaded prompt
            logger.info(f"Prompt expects these variables: {self.prompt_template.input_variables}")
            logger.info("Survey Simulation Expert initialized with prompt template")
            
            logger.info("Successfully loaded survey simulation prompt templates")
        except Exception as e:
            logger.error(f"Error loading prompt templates: {str(e)}")
            raise

    async def simulate_survey(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Dict[str, Any] = None,
        target_audience: List[str] = None,
        brand_values: List[str] = None,
        competitive_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Simulate a survey for a brand name with 100 individual persona responses.
        
        Args:
            run_id: The run ID for the workflow
            brand_name: The brand name to evaluate
            brand_context: Context about the brand
            target_audience: Target audience segments
            brand_values: Brand values
            competitive_analysis: Competitive analysis data
            
        Returns:
            Dict containing survey simulation results with 100 individual persona responses
        """
        try:
            logger.info(f"Simulating survey for brand name: {brand_name} (run: {run_id})")
            
            # Format brand context
            formatted_brand_context = "Not provided"
            if brand_context:
                formatted_brand_context = "\n".join([f"{k}: {v}" for k, v in brand_context.items()])
            
            # Format target audience
            formatted_target_audience = "General consumers"
            if target_audience:
                formatted_target_audience = ", ".join(target_audience)
            
            # Format brand values
            formatted_brand_values = "Not specified"
            if brand_values:
                formatted_brand_values = ", ".join(brand_values)
            
            # Format competitive analysis
            formatted_competitive_analysis = "No competitive analysis available"
            if competitive_analysis:
                formatted_competitive_analysis = json.dumps(competitive_analysis, indent=2)
            
            # Get format instructions from the output parser
            format_instructions = self.output_parser.get_format_instructions()
            
            # Create the prompt with all inputs
            prompt_inputs = {
                "format_instructions": format_instructions,
                "brand_name": brand_name,
                "brand_context": formatted_brand_context,
                "target_audience": formatted_target_audience,
                "brand_values": formatted_brand_values,
                "competitive_analysis": formatted_competitive_analysis,
                "generate_individual_personas": True,
                "num_personas": 100
            }
            
            # Format the prompt
            prompt = self.prompt_template.format_prompt(**prompt_inputs)
            
            # Log the prompt for debugging
            logger.debug(f"Survey simulation prompt for {brand_name}: {prompt.to_string()}")
            
            # Invoke the LLM
            response = await self.llm.ainvoke(prompt.to_messages())
            
            # Log the response for debugging
            logger.debug(f"Survey simulation response for {brand_name}: {response.content}")
            
            # Parse the response
            try:
                output = self.output_parser.parse(response.content)
                logger.info(f"Successfully parsed survey simulation for {brand_name}")
            except Exception as parse_error:
                logger.warning(f"Error parsing survey simulation for {brand_name}: {str(parse_error)}")
                logger.warning("Attempting manual extraction...")
                output = self._manual_extract_fields(response.content)
            
            # Normalize the output
            normalized_output = self._normalize_survey_output(output)
            
            # Check if individual_personas were generated by the LLM
            if not normalized_output.get("individual_personas"):
                logger.warning(f"LLM did not generate individual personas for {brand_name}. Using fallback method.")
                # Only use the fallback method if the LLM didn't generate personas
                normalized_output["individual_personas"] = self._generate_individual_personas(
                    brand_name, 
                    normalized_output, 
                    formatted_target_audience
                )
            else:
                persona_count = len(normalized_output["individual_personas"])
                logger.info(f"LLM successfully generated {persona_count} personas for {brand_name}")
            
            # Create the final results
            simulation_results = {
                "brand_name": brand_name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                **normalized_output
            }
            
            # Store in Supabase
            await self._store_in_supabase(run_id, simulation_results)
            
            return simulation_results
            
        except Exception as e:
            error_msg = f"Error simulating survey for brand name '{brand_name}': {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _manual_extract_fields(self, content: str) -> Dict[str, Any]:
        """Manually extract fields from the response if structured parsing fails."""
        result = {}
        
        # Extract persona_segment
        persona_match = re.search(r'"persona_segment"\s*:\s*(\[.*?\])', content, re.DOTALL)
        if persona_match:
            try:
                result["persona_segment"] = json.loads(persona_match.group(1).replace("'", '"'))
            except:
                result["persona_segment"] = ["General Consumer", "Professional", "Young Adult"]
        
        # Extract numeric scores (0-10 range)
        score_fields = [
            "brand_promise_perception_score", "personality_fit_score", 
            "competitive_differentiation_score", "competitor_benchmarking_score",
            "simulated_market_adoption_score"
        ]
        for field in score_fields:
            match = re.search(fr'"{field}"\s*:\s*(\d+(\.\d+)?)', content)
            if match:
                result[field] = float(match.group(1))
            else:
                # Default to middle score if not found
                result[field] = 5.0
        
        # Extract other numeric fields
        other_numeric_fields = [
            "company_size_employees", "company_size_revenue", "market_share", "years_of_experience"
        ]
        for field in other_numeric_fields:
            match = re.search(fr'"{field}"\s*:\s*(\d+(\.\d+)?)', content)
            if match:
                result[field] = float(match.group(1))
            else:
                result[field] = None
        
        # Extract arrays
        array_fields = ["emotional_association", "functional_association"]
        for field in array_fields:
            match = re.search(fr'"{field}"\s*:\s*(\[.*?\])', content, re.DOTALL)
            if match:
                try:
                    result[field] = json.loads(match.group(1).replace("'", '"'))
                except:
                    result[field] = ["Undefined"]
            else:
                result[field] = ["Undefined"]
        
        # Extract objects
        object_fields = ["psychometric_sentiment_mapping", "raw_qualitative_feedback"]
        for field in object_fields:
            match = re.search(fr'"{field}"\s*:\s*(\{{.*?\}})', content, re.DOTALL)
            if match:
                try:
                    result[field] = json.loads(match.group(1).replace("'", '"'))
                except:
                    result[field] = {"error": "Could not parse"}
            else:
                result[field] = {"error": "Not provided"}
        
        # Extract text fields
        text_fields = [
            "qualitative_feedback_summary", "final_survey_recommendation", 
            "industry", "company_structure", "geographic_location", "technology_stack",
            "company_growth_stage", "company_culture", "financial_stability", "supply_chain",
            "legal_regulatory_environment", "job_title", "seniority", "department",
            "education_level", "goals_and_challenges", "values_and_priorities",
            "decision_making_style", "information_sources", "communication_preferences",
            "pain_points", "technological_literacy", "attitude_towards_risk",
            "purchasing_behavior", "online_behavior", "interaction_with_brand",
            "professional_associations", "technical_skills", "language",
            "learning_style", "networking_habits", "professional_aspirations",
            "influence_within_company", "channel_preferences", "event_attendance",
            "content_consumption_habits", "vendor_relationship_preferences"
        ]
        for field in text_fields:
            match = re.search(fr'"{field}"\s*:\s*"(.+?)"', content, re.DOTALL)
            if match:
                result[field] = match.group(1)
            else:
                result[field] = "Not available"
        
        # Extract strategic ranking
        rank_match = re.search(r'"strategic_ranking"\s*:\s*(\d+)', content)
        if rank_match:
            result["strategic_ranking"] = int(rank_match.group(1))
        else:
            result["strategic_ranking"] = 5
        
        # Try to extract individual_personas
        personas_match = re.search(r'"individual_personas"\s*:\s*(\[.*?\])', content, re.DOTALL)
        if personas_match:
            try:
                result["individual_personas"] = json.loads(personas_match.group(1).replace("'", '"'))
            except:
                result["individual_personas"] = []
        else:
            result["individual_personas"] = []
        
        return result
    
    def _normalize_survey_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate the survey output to match expected formats."""
        normalized = {}
        
        # Ensure persona_segment is a list
        if "persona_segment" in output:
            if isinstance(output["persona_segment"], str):
                # Convert comma-separated string to list
                normalized["persona_segment"] = [p.strip() for p in output["persona_segment"].split(",")]
            else:
                normalized["persona_segment"] = output["persona_segment"]
        else:
            normalized["persona_segment"] = ["General Consumer"]
        
        # Ensure numeric fields are proper floats in range 0-10
        numeric_fields = [
            "brand_promise_perception_score", "personality_fit_score", 
            "competitive_differentiation_score", "competitor_benchmarking_score",
            "simulated_market_adoption_score"
        ]
        for field in numeric_fields:
            if field in output:
                try:
                    normalized[field] = float(output[field])
                    normalized[field] = max(0.0, min(10.0, normalized[field]))
                except (ValueError, TypeError):
                    normalized[field] = 5.0
            else:
                normalized[field] = 5.0
        
        # Additional numeric fields that could have other ranges
        additional_numeric_fields = [
            "company_size_employees", "company_size_revenue", "market_share", "years_of_experience"
        ]
        for field in additional_numeric_fields:
            if field in output:
                try:
                    normalized[field] = float(output[field])
                except (ValueError, TypeError):
                    normalized[field] = None
            else:
                normalized[field] = None
        
        # Ensure array fields are proper lists
        array_fields = ["emotional_association", "functional_association"]
        for field in array_fields:
            if field in output:
                if isinstance(output[field], str):
                    # Convert comma-separated string to list
                    normalized[field] = [item.strip() for item in output[field].split(",") if item.strip()]
                else:
                    normalized[field] = output[field]
            else:
                normalized[field] = ["Undefined"]
        
        # Ensure object fields are proper dicts
        object_fields = ["psychometric_sentiment_mapping", "raw_qualitative_feedback"]
        for field in object_fields:
            if field in output:
                if isinstance(output[field], str):
                    try:
                        normalized[field] = json.loads(output[field])
                    except json.JSONDecodeError:
                        normalized[field] = {field: output[field]}
                else:
                    normalized[field] = output[field]
            else:
                normalized[field] = {}
        
        # Ensure text fields are proper strings
        text_fields = [
            "qualitative_feedback_summary", "final_survey_recommendation", 
            "industry", "company_structure", "geographic_location", "technology_stack",
            "company_growth_stage", "company_culture", "financial_stability", "supply_chain",
            "legal_regulatory_environment", "job_title", "seniority", "department",
            "education_level", "goals_and_challenges", "values_and_priorities",
            "decision_making_style", "information_sources", "communication_preferences",
            "pain_points", "technological_literacy", "attitude_towards_risk",
            "purchasing_behavior", "online_behavior", "interaction_with_brand",
            "professional_associations", "technical_skills", "language",
            "learning_style", "networking_habits", "professional_aspirations",
            "influence_within_company", "channel_preferences", "event_attendance",
            "content_consumption_habits", "vendor_relationship_preferences"
        ]
        for field in text_fields:
            if field in output:
                normalized[field] = str(output[field])
            else:
                normalized[field] = "Not provided"
        
        # Ensure strategic_ranking is an integer
        if "strategic_ranking" in output:
            try:
                normalized["strategic_ranking"] = int(output["strategic_ranking"])
            except (ValueError, TypeError):
                normalized["strategic_ranking"] = 5
        else:
            normalized["strategic_ranking"] = 5
        
        # Handle individual_personas field
        if "individual_personas" in output:
            if isinstance(output["individual_personas"], list):
                normalized["individual_personas"] = output["individual_personas"]
            elif isinstance(output["individual_personas"], str):
                try:
                    # Try to parse as JSON
                    normalized["individual_personas"] = json.loads(output["individual_personas"])
                except json.JSONDecodeError:
                    # If parsing fails, set to empty list
                    normalized["individual_personas"] = []
            else:
                normalized["individual_personas"] = []
        else:
            normalized["individual_personas"] = []
        
        return normalized

    def _generate_individual_personas(self, brand_name: str, survey_data: Dict[str, Any], target_audience: str) -> List[Dict[str, Any]]:
        """Generate 100 individual persona responses based on the aggregated survey data.
        
        This is only used as a fallback if the LLM doesn't generate individual personas.
        In such cases, we'll create placeholder personas with minimal data that can be
        displayed in the UI, but we don't attempt to generate realistic personas as that
        is the LLM's job.
        
        Args:
            brand_name: The brand name being evaluated
            survey_data: The aggregated survey data
            target_audience: The target audience description
            
        Returns:
            List of placeholder persona responses
        """
        # Parse target audience into segments
        audience_segments = [segment.strip() for segment in target_audience.split(',')]
        if not audience_segments or audience_segments[0] == "General consumers":
            audience_segments = ["General Consumer"]
        
        # Get base scores from survey data
        base_scores = {
            "brand_promise_perception_score": survey_data.get("brand_promise_perception_score", 5.0),
            "personality_fit_score": survey_data.get("personality_fit_score", 5.0),
            "competitive_differentiation_score": survey_data.get("competitive_differentiation_score", 5.0),
            "competitor_benchmarking_score": survey_data.get("competitor_benchmarking_score", 5.0),
            "simulated_market_adoption_score": survey_data.get("simulated_market_adoption_score", 5.0)
        }
        
        # Get emotional and functional associations
        emotional_associations = survey_data.get("emotional_association", ["Neutral"])
        functional_associations = survey_data.get("functional_association", ["Generic"])
        
        # Create placeholder personas - only used if LLM fails to generate them
        individual_personas = []
        for i in range(10):  # Only create 10 placeholders as a fallback
            segment = audience_segments[i % len(audience_segments)]
            
            # Create basic placeholder persona
            persona = {
                "persona_id": i + 1,
                "segment": segment,
                "age_group": "25-34",
                "gender": "Not specified",
                "income_level": "Medium",
                "education_level": "Bachelor's",
                "job_title": f"{segment} Professional",
                "seniority": "Mid-level",
                "years_of_experience": 5,
                "department": "General",
                "goals_and_challenges": "Typical industry goals and challenges",
                "values_and_priorities": "Industry-standard values",
                "decision_making_style": "Balanced",
                "information_sources": "Industry publications",
                "communication_preferences": "Email",
                "pain_points": "Common industry pain points",
                "technological_literacy": "Intermediate",
                "attitude_towards_risk": "Moderate",
                "purchasing_behavior": "Researches before purchasing",
                "online_behavior": "Regular online user",
                "interaction_with_brand": "Potential customer",
                "professional_associations": "Industry association",
                "technical_skills": "Industry-relevant skills",
                "language": "English",
                "learning_style": "Visual",
                "networking_habits": "Regular networking",
                "professional_aspirations": "Career advancement",
                "influence_within_company": "Some influence",
                "channel_preferences": "Email and social media",
                "event_attendance": "Industry conferences",
                "content_consumption_habits": "Articles and videos",
                "vendor_relationship_preferences": "Professional relationship",
                "industry": survey_data.get("industry", "Technology"),
                "company_size_employees": 250,
                "company_size_revenue": 5.0,
                "market_share": 5.0,
                "company_structure": "Hierarchical",
                "geographic_location": "Urban",
                "technology_stack": "Standard industry technology",
                "company_growth_stage": "Growth",
                "company_culture": "Professional",
                "financial_stability": "Stable",
                "supply_chain": "Standard",
                "legal_regulatory_environment": "Standard regulations",
                "scores": base_scores,
                "emotional_associations": emotional_associations[:1],
                "functional_associations": functional_associations[:1],
                "qualitative_feedback": f"This persona would likely view the brand name '{brand_name}' as acceptable but not outstanding.",
                "purchase_intent": "Maybe"
            }
            
            individual_personas.append(persona)
        
        logger.warning(f"Using {len(individual_personas)} placeholder personas as LLM did not generate individual personas")
        return individual_personas

    async def _store_in_supabase(self, run_id: str, simulation_results: Dict[str, Any]) -> None:
        """Store survey simulation results in Supabase.
        
        Args:
            run_id: The run ID for the workflow
            simulation_results: The survey simulation results
        """
        try:
            logger.info(f"Storing survey simulation for brand name: {simulation_results.get('brand_name', 'Unknown')} (run: {run_id})")
            
            # Prepare data for insertion
            data_to_insert = {
                "run_id": run_id,
                "brand_name": simulation_results.get("brand_name", "Unknown"),
                "persona_segment": simulation_results.get("persona_segment", []),
                "brand_promise_perception_score": simulation_results.get("brand_promise_perception_score", 5.0),
                "personality_fit_score": simulation_results.get("personality_fit_score", 5.0),
                "emotional_association": simulation_results.get("emotional_association", ["Neutral"]),
                "functional_association": simulation_results.get("functional_association", ["Generic"]),
                "competitive_differentiation_score": simulation_results.get("competitive_differentiation_score", 5.0),
                "psychometric_sentiment_mapping": simulation_results.get("psychometric_sentiment_mapping", {}),
                "competitor_benchmarking_score": simulation_results.get("competitor_benchmarking_score", 5.0),
                "simulated_market_adoption_score": simulation_results.get("simulated_market_adoption_score", 5.0),
                "qualitative_feedback_summary": simulation_results.get("qualitative_feedback_summary", "Not provided"),
                "raw_qualitative_feedback": simulation_results.get("raw_qualitative_feedback", {}),
                "final_survey_recommendation": simulation_results.get("final_survey_recommendation", "Not provided"),
                "strategic_ranking": simulation_results.get("strategic_ranking", 5),
                "individual_personas": simulation_results.get("individual_personas", []),
                
                # Company-related fields
                "industry": simulation_results.get("industry", None),
                "company_size_employees": simulation_results.get("company_size_employees", None),
                "company_size_revenue": simulation_results.get("company_size_revenue", None),
                "market_share": simulation_results.get("market_share", None),
                "company_structure": simulation_results.get("company_structure", None),
                "geographic_location": simulation_results.get("geographic_location", None),
                "technology_stack": simulation_results.get("technology_stack", None),
                "company_growth_stage": simulation_results.get("company_growth_stage", None),
                "company_culture": simulation_results.get("company_culture", None),
                "financial_stability": simulation_results.get("financial_stability", None),
                "supply_chain": simulation_results.get("supply_chain", None),
                "legal_regulatory_environment": simulation_results.get("legal_regulatory_environment", None),
                
                # Persona-related fields
                "job_title": simulation_results.get("job_title", None),
                "seniority": simulation_results.get("seniority", None),
                "years_of_experience": simulation_results.get("years_of_experience", None),
                "department": simulation_results.get("department", None),
                "education_level": simulation_results.get("education_level", None),
                "goals_and_challenges": simulation_results.get("goals_and_challenges", None),
                "values_and_priorities": simulation_results.get("values_and_priorities", None),
                "decision_making_style": simulation_results.get("decision_making_style", None),
                "information_sources": simulation_results.get("information_sources", None),
                "communication_preferences": simulation_results.get("communication_preferences", None),
                "pain_points": simulation_results.get("pain_points", None),
                "technological_literacy": simulation_results.get("technological_literacy", None),
                "attitude_towards_risk": simulation_results.get("attitude_towards_risk", None),
                "purchasing_behavior": simulation_results.get("purchasing_behavior", None),
                "online_behavior": simulation_results.get("online_behavior", None),
                "interaction_with_brand": simulation_results.get("interaction_with_brand", None),
                "professional_associations": simulation_results.get("professional_associations", None),
                "technical_skills": simulation_results.get("technical_skills", None),
                "language": simulation_results.get("language", None),
                "learning_style": simulation_results.get("learning_style", None),
                "networking_habits": simulation_results.get("networking_habits", None),
                "professional_aspirations": simulation_results.get("professional_aspirations", None),
                "influence_within_company": simulation_results.get("influence_within_company", None),
                "channel_preferences": simulation_results.get("channel_preferences", None),
                "event_attendance": simulation_results.get("event_attendance", None),
                "content_consumption_habits": simulation_results.get("content_consumption_habits", None),
                "vendor_relationship_preferences": simulation_results.get("vendor_relationship_preferences", None)
            }
            
            # Execute the insert with retries
            await self.supabase.execute_with_retry(
                operation="insert",
                table="survey_simulation",
                data=data_to_insert
            )
            
            logger.info(f"Successfully stored survey simulation for {simulation_results['brand_name']} (run: {run_id})")
            
        except Exception as e:
            error_msg = f"Error storing survey simulation in Supabase: {str(e)}"
            logger.error(error_msg)
            # Log error but don't re-raise to prevent workflow disruption 