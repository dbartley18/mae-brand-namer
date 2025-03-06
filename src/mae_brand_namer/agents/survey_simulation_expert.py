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
            ResponseSchema(
                name="individual_personas", 
                description="""Array of EXACTLY 100 individual persona responses, each with demographic details and feedback. 
                    Format: [
                        {"persona_id": "1", "segment": "...", "age_group": "...", "gender": "...", "income_level": "...", ...},
                        {"persona_id": "2", "segment": "...", "age_group": "...", "gender": "...", "income_level": "...", ...},
                        ...
                    ]
                    IMPORTANT: You MUST generate EXACTLY 100 unique, non-random personas. Each one will be saved in the database as a separate row.
                """
            ),
            
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
            self.single_persona_prompt = load_prompt(str(prompt_dir / "single_persona.yaml"))
            
            # Get format instructions from the output parser
            self.format_instructions = self.output_parser.get_format_instructions()
            
            # Create the main prompt template for full survey simulation
            system_message = SystemMessage(content=self.system_prompt.template)
            human_template = self.simulation_prompt.template
            
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_message.content),
                ("human", human_template)
            ])
            
            # Create the single persona template using the loaded prompt
            self.single_persona_template = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt.template),
                ("human", self.single_persona_prompt.template)
            ])
            
            logger.info("Successfully loaded all prompt templates")
            
        except Exception as e:
            error_msg = f"Error loading prompt templates: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def simulate_survey(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Dict[str, Any] = None,
        target_audience: List[str] = None,
        brand_values: List[str] = None,
        competitive_analysis: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Simulate a survey for a brand name, generating individual persona responses one at a time.
        
        Args:
            run_id: The run ID for the workflow
            brand_name: The brand name to evaluate
            brand_context: Context about the brand
            target_audience: Target audience segments
            brand_values: Brand values
            competitive_analysis: Competitive analysis data
            
        Returns:
            Dict containing survey simulation results with individual persona responses
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
            
            # First, get the overall survey results
            logger.info(f"Generating overall survey results for {brand_name}")
            overall_results = await self._generate_overall_survey_results(
                brand_name,
                formatted_brand_context,
                formatted_target_audience,
                formatted_brand_values,
                formatted_competitive_analysis
            )
            
            # Normalize the overall results
            normalized_results = self._normalize_survey_output(overall_results)
            
            # Now generate individual personas one by one
            logger.info(f"Generating individual personas for {brand_name}")
            individual_personas = []
            num_personas = 50  # Reduced from 100 to 50 as requested
            
            # Create the prompt with all inputs for individual persona generation
            # Use the existing prompt templates from _load_prompts()
            prompt_inputs = {
                "format_instructions": self.format_instructions,
                "brand_name": brand_name,
                "brand_context": formatted_brand_context,
                "target_audience": formatted_target_audience,
                "brand_values": formatted_brand_values,
                "competitive_analysis": formatted_competitive_analysis,
                "generate_individual_personas": True,
                "num_personas": 1  # Generate one persona at a time
            }
            
            # Generate personas one by one
            success_count = 0
            for i in range(num_personas):
                try:
                    persona_id = f"{brand_name.replace(' ', '')}-{i+1:03d}"
                    prompt_inputs["persona_number"] = i + 1
                    prompt_inputs["total_personas"] = num_personas
                    prompt_inputs["persona_id"] = persona_id
                    
                    # Add a note about generating just ONE persona in this call
                    # Also emphasize that the agent should first create the persona, then answer as that persona
                    prompt_note = f"""
                    IMPORTANT: In this call, generate ONLY ONE detailed persona (persona #{i+1} of {num_personas}) with ID '{persona_id}'.
                    
                    FOLLOW THIS TWO-STEP APPROACH:
                    1. First, create a detailed, realistic persona with demographic and firmographic details.
                    2. Then, answer the survey questions AS this persona would respond to the brand name.
                    
                    The persona should think about each question holistically, as if they were actually taking the survey.
                    
                    Return a complete JSON object for this one persona only.
                    """
                    
                    # Format the prompt using the template from _load_prompts()
                    single_persona_prompt = ChatPromptTemplate.from_messages([
                        ("system", self.system_prompt.template),
                        ("human", self.simulation_prompt.template + prompt_note)
                    ])
                    
                    formatted_prompt = single_persona_prompt.format_prompt(**prompt_inputs)
                    
                    # Call the LLM
                    logger.info(f"Generating persona {i+1}/{num_personas} for {brand_name}")
                    persona_response = await self.llm.ainvoke(formatted_prompt.to_messages())
                    persona_content = persona_response.content
                    
                    # Try to parse the persona
                    try:
                        # Clean up the response - look for JSON object
                        json_start = persona_content.find('{')
                        json_end = persona_content.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_content = persona_content[json_start:json_end]
                            persona = json.loads(json_content)
                        else:
                            # If no JSON found, use the whole content
                            persona = json.loads(persona_content)
                        
                        # Ensure persona_id is present
                        if "persona_id" not in persona:
                            persona["persona_id"] = persona_id
                        
                        # Store this persona
                        if self.supabase:
                            await self._store_single_persona(run_id, normalized_results, persona)
                        
                        # Add to our list
                        individual_personas.append(persona)
                        success_count += 1
                        logger.info(f"Successfully generated persona {i+1}/{num_personas} for {brand_name}")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse persona {i+1}: {str(e)}")
                        logger.debug(f"Persona content: {persona_content[:500]}...")
                        
                        # Try a fallback approach with regex
                        try:
                            # Extract key fields with regex
                            fallback_persona = {
                                "persona_id": persona_id,
                                "segment": self._extract_field(persona_content, "persona_segment"),
                                "job_title": self._extract_field(persona_content, "job_title"),
                                "industry": self._extract_field(persona_content, "industry"),
                                "qualitative_feedback_summary": self._extract_field(persona_content, "qualitative_feedback_summary")
                            }
                            
                            # Store this fallback persona
                            if self.supabase:
                                await self._store_single_persona(run_id, normalized_results, fallback_persona)
                            
                            # Add to our list
                            individual_personas.append(fallback_persona)
                            success_count += 1
                            logger.info(f"Used fallback parsing for persona {i+1}/{num_personas}")
                        
                        except Exception as fallback_error:
                            logger.error(f"Fallback parsing also failed for persona {i+1}: {str(fallback_error)}")
                except Exception as persona_error:
                    logger.error(f"Error generating persona {i+1}: {str(persona_error)}")
            
            # Add the personas to the results
            normalized_results["individual_personas"] = individual_personas
            
            # Create the final results
            simulation_results = {
                "brand_name": brand_name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                **normalized_results,
                "success_count": success_count,
                "total_personas": num_personas
            }
            
            logger.info(f"Successfully generated {success_count}/{num_personas} personas for {brand_name}")
            return simulation_results
            
        except Exception as e:
            error_msg = f"Error simulating survey for brand name '{brand_name}': {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _extract_field(self, content: str, field_name: str) -> str:
        """Extract a field from text content using regex."""
        pattern = fr'"{field_name}"\s*:\s*"(.+?)"'
        match = re.search(pattern, content)
        if match:
            return match.group(1)
        
        # Try without quotes
        pattern = fr'"{field_name}"\s*:\s*(.+?)(?:,|\n|}}'
        match = re.search(pattern, content)
        if match:
            return match.group(1).strip().strip('"')
        
        return f"Not available"
    
    async def _generate_overall_survey_results(
        self,
        brand_name: str,
        brand_context: str,
        target_audience: str,
        brand_values: str,
        competitive_analysis: str
    ) -> Dict[str, Any]:
        """Generate the overall survey results without individual personas."""
        
        # Define output schemas for the overall results
        overall_schemas = [
            ResponseSchema(name="brand_name", description="The brand name being evaluated"),
            ResponseSchema(name="persona_segment", description="Array of target audience segments analyzed"),
            ResponseSchema(name="brand_promise_perception_score", description="How well the name conveys brand promise (0-10 scale)"),
            ResponseSchema(name="personality_fit_score", description="Alignment with brand personality (0-10 scale)"),
            ResponseSchema(name="emotional_association", description="Array of key emotional responses evoked by the name"),
            ResponseSchema(name="functional_association", description="Array of practical/functional associations with the name"),
            ResponseSchema(name="competitive_differentiation_score", description="Distinctiveness from competitors (0-10 scale)"),
            ResponseSchema(name="psychometric_sentiment_mapping", description="JSON mapping emotional dimensions to scores"),
            ResponseSchema(name="competitor_benchmarking_score", description="Performance vs competitors (0-10 scale)"),
            ResponseSchema(name="simulated_market_adoption_score", description="Predicted market acceptance (0-10 scale)"),
            ResponseSchema(name="qualitative_feedback_summary", description="Summary of qualitative feedback"),
            ResponseSchema(name="final_survey_recommendation", description="Final recommendation based on survey results"),
            ResponseSchema(name="strategic_ranking", description="Overall strategic ranking (1=best)")
        ]
        
        # Create output parser for overall results
        overall_parser = StructuredOutputParser.from_response_schemas(overall_schemas)
        format_instructions = overall_parser.get_format_instructions()
        
        # Use existing prompt templates with a modification for overall results only
        prompt_inputs = {
            "format_instructions": format_instructions,
            "brand_name": brand_name,
            "brand_context": brand_context,
            "target_audience": target_audience,
            "brand_values": brand_values,
            "competitive_analysis": competitive_analysis,
            "generate_individual_personas": False,  # Don't generate individual personas in this call
            "num_personas": 0  # No personas needed
        }
        
        # Add a note about generating only overall results
        prompt_note = """
        IMPORTANT: In this call, generate ONLY the overall survey results and metrics. 
        DO NOT generate any individual personas. Focus on creating aggregate metrics and insights
        that summarize how the target audience would perceive this brand name.
        """
        
        # Use the existing templates from _load_prompts()
        overall_results_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt.template),
            ("human", self.simulation_prompt.template + prompt_note)
        ])
        
        # Format the prompt
        formatted_prompt = overall_results_prompt.format_prompt(**prompt_inputs)
        
        # Call the LLM
        logger.info(f"Generating overall survey results for {brand_name}")
        overall_response = await self.llm.ainvoke(formatted_prompt.to_messages())
        
        # Try to parse the overall results
        try:
            overall_results = overall_parser.parse(overall_response.content)
            logger.info(f"Successfully parsed overall survey results for {brand_name}")
            return overall_results
        except Exception as e:
            logger.warning(f"Error parsing overall survey results: {str(e)}")
            logger.warning("Attempting manual extraction...")
            return self._manual_extract_fields(overall_response.content)
    
    async def _store_single_persona(self, run_id: str, overall_results: Dict[str, Any], persona: Dict[str, Any]) -> None:
        """Store a single persona in Supabase.
        
        Args:
            run_id: The run ID for the workflow
            overall_results: The overall survey results
            persona: A single persona to store
        """
        if not self.supabase:
            logger.warning("Supabase not available, skipping storage")
            return
        
        try:
            brand_name = overall_results.get("brand_name", "Unknown")
            
            # Create the base data with overall results and ensure NOT NULL fields have non-null values
            # Updated to match the new schema
            base_data = {
                "run_id": run_id,
                "brand_name": brand_name,
                
                # Required fields (NOT NULL)
                "persona_segment": "General Consumer",  # Changed from array to string
                "emotional_association": "Neutral",  # Changed from array to string
                "psychometric_sentiment_mapping": overall_results.get("psychometric_sentiment_mapping", {"general": 5.0}),
                "raw_qualitative_feedback": overall_results.get("raw_qualitative_feedback", {"feedback": "Not provided"}),
                "qualitative_feedback_summary": overall_results.get("qualitative_feedback_summary", "Not provided"),
                "final_survey_recommendation": overall_results.get("final_survey_recommendation", "Not provided"),
                "strategic_ranking": overall_results.get("strategic_ranking", 5),
                
                # Nullable numeric fields
                "brand_promise_perception_score": overall_results.get("brand_promise_perception_score", None),
                "personality_fit_score": overall_results.get("personality_fit_score", None),
                "competitive_differentiation_score": overall_results.get("competitive_differentiation_score", None),
                "competitor_benchmarking_score": overall_results.get("competitor_benchmarking_score", None),
                "simulated_market_adoption_score": overall_results.get("simulated_market_adoption_score", None),
                
                # New fields from the updated schema
                "industry": None,
                "company_size_employees": None,
                "company_revenue": None,
                "market_share": None,
                "company_structure": None,
                "geographic_location": None,
                "technology_stack": None,
                "company_growth_stage": None,
                "job_title": None,
                "seniority": None,
                "years_of_experience": None,
                "department": None,
                "education_level": None,
                "goals_and_challenges": None,
                "values_and_priorities": None,
                "decision_making_style": None,
                "information_sources": None,
                "pain_points": None,
                "technological_literacy": None,
                "attitude_towards_risk": None,
                "purchasing_behavior": None,
                "online_behavior": None,
                "interaction_with_brand": None,
                "professional_associations": None,
                "technical_skills": None,
                "networking_habits": None,
                "professional_aspirations": None,
                "influence_within_company": None,
                "event_attendance": None,
                "content_consumption_habits": None,
                "vendor_relationship_preferences": None,
                
                # Additional new fields from the updated schema
                "business_chemistry": None,
                "reports_to": None,
                "buying_group_structure": None,
                "decision_maker": None,
                "company_focus": None,
                "company_maturity": None,
                "budget_authority": None,
                "preferred_vendor_size": None,
                "innovation_adoption": None,
                "key_performance_indicators": None,
                "professional_development_interests": None,
                "social_media_usage": None,
                "work_life_balance_priorities": None,
                "frustrations_annoyances": None,
                "personal_aspirations_life_goals": None,
                "motivations": None,
                "current_brand_relationships": None,
                "product_adoption_lifecycle_stage": None,
                "purchase_triggers_events": None,
                "success_metrics_product_service": None,
                "channel_preferences_brand_interaction": None,
                "barriers_to_adoption": None,
                "generation_age_range": None,
                "company_culture_values": None,
                "industry_sub_vertical": None,
                "confidence_score_persona_accuracy": None,
                "data_sources_persona_creation": None,
                "persona_archetype_type": None,
                "behavioral_tags_keywords": None
            }
            
            # Update with persona-specific fields
            for key, value in persona.items():
                # Handle field name mappings and type conversions
                if key == "segment":
                    # Convert from segment to persona_segment (string, not array)
                    if isinstance(value, list):
                        base_data["persona_segment"] = value[0] if value else "General Consumer"
                    else:
                        base_data["persona_segment"] = str(value)
                
                elif key == "emotional_associations":
                    # Convert from array to string
                    if isinstance(value, list):
                        base_data["emotional_association"] = ", ".join(value)
                    else:
                        base_data["emotional_association"] = str(value)
                
                elif key == "functional_associations":
                    # No longer in schema as array, convert to string if needed for qualitative feedback
                    if isinstance(value, list) and value:
                        functional_str = ", ".join(value)
                        if "qualitative_feedback_summary" in base_data:
                            base_data["qualitative_feedback_summary"] += f" Functional associations: {functional_str}"
                
                elif key == "qualitative_feedback":
                    # Map to qualitative_feedback_summary
                    base_data["qualitative_feedback_summary"] = str(value)
                
                elif key == "scores" and isinstance(value, dict):
                    # Handle individual scores
                    for score_key, score_value in value.items():
                        mapped_key = None
                        
                        # Map score keys to database fields
                        if score_key == "brand_perception":
                            mapped_key = "brand_promise_perception_score"
                        elif score_key == "personality_fit":
                            mapped_key = "personality_fit_score"
                        elif score_key == "competitive_position":
                            mapped_key = "competitive_differentiation_score"
                        elif score_key == "market_potential":
                            mapped_key = "simulated_market_adoption_score"
                        
                        if mapped_key and mapped_key in base_data:
                            try:
                                base_data[mapped_key] = float(score_value)
                            except (ValueError, TypeError):
                                # Keep the default
                                pass
                
                # Handle behavioral tags if present
                elif key == "behavioral_tags" or key == "behavioral_tags_keywords":
                    if isinstance(value, list):
                        # Keep as a list for text[] PostgreSQL array type
                        base_data["behavioral_tags_keywords"] = value
                    elif isinstance(value, str):
                        # Convert comma-separated string to list for text[] PostgreSQL array type
                        base_data["behavioral_tags_keywords"] = [tag.strip() for tag in value.split(",")]
                
                # Direct mapping for other fields
                elif key in base_data:
                    base_data[key] = value
            
            # Ensure all NOT NULL fields have valid values
            if not base_data["persona_segment"]:
                base_data["persona_segment"] = "General Consumer"
                
            if not base_data["emotional_association"]:
                base_data["emotional_association"] = "Neutral"
                
            # Ensure psychometric_sentiment_mapping is a valid JSON object
            if not base_data["psychometric_sentiment_mapping"] or not isinstance(base_data["psychometric_sentiment_mapping"], dict):
                base_data["psychometric_sentiment_mapping"] = {"general": 5.0}
                
            # Ensure raw_qualitative_feedback is a valid JSON object
            if not base_data["raw_qualitative_feedback"] or not isinstance(base_data["raw_qualitative_feedback"], dict):
                base_data["raw_qualitative_feedback"] = {"feedback": base_data["qualitative_feedback_summary"]}
                
            # Convert any None values in NOT NULL fields to appropriate defaults
            if base_data["qualitative_feedback_summary"] is None:
                base_data["qualitative_feedback_summary"] = "Not provided"
                
            if base_data["final_survey_recommendation"] is None:
                base_data["final_survey_recommendation"] = "Not provided"
                
            if base_data["strategic_ranking"] is None:
                base_data["strategic_ranking"] = 5
                
            # Convert numeric fields to correct types
            try:
                if base_data["strategic_ranking"] is not None:
                    base_data["strategic_ranking"] = int(float(base_data["strategic_ranking"]))
            except (ValueError, TypeError):
                base_data["strategic_ranking"] = 5
                
            # Convert numeric score fields to floats
            numeric_fields = [
                "brand_promise_perception_score", 
                "personality_fit_score",
                "competitive_differentiation_score", 
                "competitor_benchmarking_score",
                "simulated_market_adoption_score",
                "years_of_experience",
                "company_revenue",
                "confidence_score_persona_accuracy"
            ]
            
            for field in numeric_fields:
                if field in base_data and base_data[field] is not None:
                    try:
                        base_data[field] = float(base_data[field])
                    except (ValueError, TypeError):
                        # Set to None since these fields can be NULL
                        base_data[field] = None
            
            # Convert boolean fields
            boolean_fields = ["decision_maker"]
            for field in boolean_fields:
                if field in base_data and base_data[field] is not None:
                    if isinstance(base_data[field], str):
                        base_data[field] = base_data[field].lower() in ("yes", "true", "1")
            
            # Ensure JSON fields are proper JSON objects
            json_fields = [
                "psychometric_sentiment_mapping", 
                "raw_qualitative_feedback",
                "current_brand_relationships",
                "purchase_triggers_events",
                "data_sources_persona_creation"
            ]
            
            for field in json_fields:
                if field in base_data and base_data[field] is not None:
                    if isinstance(base_data[field], str):
                        try:
                            base_data[field] = json.loads(base_data[field])
                        except json.JSONDecodeError:
                            if field == "psychometric_sentiment_mapping" or field == "raw_qualitative_feedback":
                                # These are required fields
                                base_data[field] = {"value": base_data[field]}
                            else:
                                # These can be null
                                base_data[field] = None
                    elif not isinstance(base_data[field], dict) and not isinstance(base_data[field], list):
                        if field == "psychometric_sentiment_mapping" or field == "raw_qualitative_feedback":
                            base_data[field] = {"value": str(base_data[field])}
                        else:
                            base_data[field] = None
            
            # Log the data being sent
            logger.debug(f"Storing persona with keys: {list(base_data.keys())}")
            
            # Execute the insert with retries
            await self.supabase.execute_with_retry(
                operation="insert",
                table="survey_simulation",
                data=base_data
            )
            
            logger.info(f"Successfully stored persona {persona.get('persona_id', 'unknown')} for {brand_name}")
            
        except Exception as e:
            error_msg = f"Error storing persona in Supabase: {str(e)}"
            logger.error(error_msg)
            # Include more information about the error
            if hasattr(e, 'code'):
                logger.error(f"Error code: {e.code}")
            if hasattr(e, 'details'):
                logger.error(f"Error details: {e.details}")
            
            # Log partial data for debugging
            try:
                logger.debug(f"Persona data causing error: run_id={run_id}, brand_name={overall_results.get('brand_name', 'Unknown')}")
            except:
                pass
            # Log error but don't re-raise to prevent workflow disruption

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
        
        # Try multiple approaches to extract individual_personas
        try:
            # First approach: try to find the entire individual_personas array and parse it
            personas_match = re.search(r'"individual_personas"\s*:\s*(\[.*\])', content, re.DOTALL)
            if personas_match:
                personas_text = personas_match.group(1)
                # Clean up the JSON before parsing
                cleaned_json = personas_text.replace("'", '"')
                # Sometimes multi-line strings cause issues, attempt to normalize newlines in strings
                cleaned_json = re.sub(r'"\s*\n\s*"', ' ', cleaned_json)
                try:
                    result["individual_personas"] = json.loads(cleaned_json)
                    logger.info(f"Successfully extracted {len(result['individual_personas'])} personas using regex + json parsing")
                    return result
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing error for individual_personas: {e}")
                    # Continue to next approach if this fails
            
            # Second approach: Look for valid JSON objects in the text
            logger.info("Attempting to find individual persona objects in the text")
            # Look for JSON-like objects that match our persona pattern
            persona_pattern = r'\{\s*"persona_id"\s*:\s*"[^"]+"\s*,.*?\}'
            persona_matches = re.findall(persona_pattern, content, re.DOTALL)
            
            if persona_matches:
                personas = []
                for match in persona_matches:
                    try:
                        # Replace single quotes with double quotes for JSON compatibility
                        cleaned_match = match.replace("'", '"')
                        # Try to parse each persona object
                        persona = json.loads(cleaned_match)
                        personas.append(persona)
                    except json.JSONDecodeError:
                        # Skip invalid JSON objects
                        pass
                
                if personas:
                    logger.info(f"Extracted {len(personas)} individual personas using pattern matching")
                    result["individual_personas"] = personas
                    return result
            
            # Third approach: If the content is already valid JSON, try parsing it directly
            try:
                # If the entire response is valid JSON, parse it
                full_json = json.loads(content)
                if "individual_personas" in full_json and isinstance(full_json["individual_personas"], list):
                    logger.info(f"Extracted {len(full_json['individual_personas'])} personas from full JSON")
                    result["individual_personas"] = full_json["individual_personas"]
                    return result
            except json.JSONDecodeError:
                # Not a valid JSON, continue to fallback
                pass
            
            # If we get here, we couldn't extract individual_personas
            logger.warning("Could not extract individual_personas using any method, using empty list")
            result["individual_personas"] = []
            
        except Exception as e:
            logger.error(f"Error while trying to extract individual_personas: {str(e)}")
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
                    value = float(output[field])
                    normalized[field] = max(0, min(10, value))  # Clamp to 0-10 range
                except (ValueError, TypeError):
                    normalized[field] = 5.0  # Default mid-range value
            else:
                normalized[field] = 5.0  # Default mid-range value
        
        # Ensure additional numeric fields (may be None)
        additional_numeric_fields = [
            "years_of_experience", "company_size_employees", "company_size_revenue",
            "market_share"
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
        
        # Handle individual_personas field - no parsing needed in one-by-one mode
        # Just initialize to empty list - we'll add personas later
        normalized["individual_personas"] = []
        
        return normalized 