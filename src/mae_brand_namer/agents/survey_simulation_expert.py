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
            # Required NOT NULL fields
            ResponseSchema(name="run_id", description="Unique identifier for this survey run"),
            ResponseSchema(name="brand_name", description="The brand name being evaluated"),
            ResponseSchema(name="persona_segment", description="Target audience segment as a single detailed text description"),
            ResponseSchema(name="emotional_association", description="Emotional responses as a single concatenated text description"),
            ResponseSchema(name="psychometric_sentiment_mapping", description="JSON mapping emotional dimensions to scores"),
            ResponseSchema(name="qualitative_feedback_summary", description="Detailed qualitative feedback summary"),
            ResponseSchema(name="raw_qualitative_feedback", description="JSON object with structured feedback data"),
            ResponseSchema(name="final_survey_recommendation", description="Final recommendation with reasoning"),
            ResponseSchema(name="strategic_ranking", description="Integer ranking from 1-10"),

            # Numeric scores (technically nullable but required for process)
            ResponseSchema(name="brand_promise_perception_score", description="Score for brand promise perception (0-10)"),
            ResponseSchema(name="personality_fit_score", description="Score for brand personality fit (0-10)"),
            ResponseSchema(name="competitive_differentiation_score", description="Score for competitive differentiation (0-10)"),
            ResponseSchema(name="competitor_benchmarking_score", description="Score for competitor benchmarking (0-10)"),
            ResponseSchema(name="simulated_market_adoption_score", description="Score for market adoption potential (0-10)"),
            ResponseSchema(name="years_of_experience", description="Years of professional experience"),
            ResponseSchema(name="company_revenue", description="Annual company revenue"),
            ResponseSchema(name="confidence_score_persona_accuracy", description="Confidence score for persona accuracy (1-10)"),

            # JSONB fields (technically nullable but required for process)
            ResponseSchema(name="current_brand_relationships", description="JSON object describing current brand relationships and preferences"),
            ResponseSchema(name="purchase_triggers_events", description="JSON object detailing purchase triggers and events"),
            ResponseSchema(name="data_sources_persona_creation", description="JSON object listing data sources used for persona creation"),

            # Text array field
            ResponseSchema(name="behavioral_tags_keywords", description="Array of behavioral keywords and tags"),

            # Boolean field
            ResponseSchema(name="decision_maker", description="Whether the persona is a decision maker"),

            # Text fields (all required for process)
            ResponseSchema(name="industry", description="Primary industry sector"),
            ResponseSchema(name="company_size_employees", description="Company size by employee count"),
            ResponseSchema(name="market_share", description="Market share description"),
            ResponseSchema(name="company_structure", description="Organizational structure"),
            ResponseSchema(name="geographic_location", description="Primary location"),
            ResponseSchema(name="technology_stack", description="Technology stack description"),
            ResponseSchema(name="company_growth_stage", description="Company growth stage"),
            ResponseSchema(name="job_title", description="Job title"),
            ResponseSchema(name="seniority", description="Seniority level"),
            ResponseSchema(name="department", description="Department"),
            ResponseSchema(name="education_level", description="Education level"),
            ResponseSchema(name="goals_and_challenges", description="Professional goals and challenges"),
            ResponseSchema(name="values_and_priorities", description="Core values and priorities"),
            ResponseSchema(name="decision_making_style", description="Decision making approach"),
            ResponseSchema(name="information_sources", description="Information sources"),
            ResponseSchema(name="pain_points", description="Key pain points"),
            ResponseSchema(name="technological_literacy", description="Technology literacy level"),
            ResponseSchema(name="attitude_towards_risk", description="Risk attitude"),
            ResponseSchema(name="purchasing_behavior", description="Purchasing behavior patterns"),
            ResponseSchema(name="online_behavior", description="Online behavior patterns"),
            ResponseSchema(name="interaction_with_brand", description="Brand interaction style"),
            ResponseSchema(name="professional_associations", description="Professional associations"),
            ResponseSchema(name="technical_skills", description="Technical skills"),
            ResponseSchema(name="networking_habits", description="Networking habits"),
            ResponseSchema(name="professional_aspirations", description="Professional aspirations"),
            ResponseSchema(name="influence_within_company", description="Level of influence, max 255 chars"),
            ResponseSchema(name="event_attendance", description="Event attendance patterns"),
            ResponseSchema(name="content_consumption_habits", description="Content consumption habits"),
            ResponseSchema(name="vendor_relationship_preferences", description="Vendor relationship preferences"),
            ResponseSchema(name="business_chemistry", description="Business personality type"),
            ResponseSchema(name="reports_to", description="Reporting structure"),
            ResponseSchema(name="buying_group_structure", description="Buying group structure"),
            ResponseSchema(name="company_focus", description="Company focus areas"),
            ResponseSchema(name="company_maturity", description="Company maturity level"),
            ResponseSchema(name="budget_authority", description="Budget authority level"),
            ResponseSchema(name="preferred_vendor_size", description="Preferred vendor size"),
            ResponseSchema(name="innovation_adoption", description="Innovation adoption approach"),
            ResponseSchema(name="key_performance_indicators", description="Key performance indicators"),
            ResponseSchema(name="professional_development_interests", description="Professional development interests"),
            ResponseSchema(name="social_media_usage", description="Social media usage patterns"),
            ResponseSchema(name="work_life_balance_priorities", description="Work-life balance priorities"),
            ResponseSchema(name="frustrations_annoyances", description="Key frustrations and annoyances"),
            ResponseSchema(name="personal_aspirations_life_goals", description="Personal aspirations and life goals"),
            ResponseSchema(name="motivations", description="Key motivations"),
            ResponseSchema(name="product_adoption_lifecycle_stage", description="Product adoption lifecycle stage"),
            ResponseSchema(name="success_metrics_product_service", description="Success metrics for products/services"),
            ResponseSchema(name="channel_preferences_brand_interaction", description="Channel preferences for brand interaction"),
            ResponseSchema(name="barriers_to_adoption", description="Barriers to adoption"),
            ResponseSchema(name="generation_age_range", description="Generation and age range"),
            ResponseSchema(name="company_culture_values", description="Company culture and values"),
            ResponseSchema(name="industry_sub_vertical", description="Industry sub-vertical"),
            ResponseSchema(name="persona_archetype_type", description="Persona archetype classification")
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
            
            # Generate individual personas one by one
            logger.info(f"Generating individual personas for {brand_name}")
            individual_personas = []
            num_personas = 50  # Generate 50 personas per brand name
            success_count = 0
            
            for i in range(num_personas):
                try:
                    # Generate a single persona response
                    persona = await self._generate_persona_response(
                        brand_name=brand_name,
                        brand_context=formatted_brand_context,
                        target_audience=formatted_target_audience,
                        brand_values=formatted_brand_values,
                        competitive_analysis=formatted_competitive_analysis,
                        persona_number=i + 1,
                        total_personas=num_personas
                    )
                    
                    # Store this persona
                    if self.supabase:
                        await self._store_single_persona(run_id, {"brand_name": brand_name}, persona)
                    
                    # Add to our list
                    individual_personas.append(persona)
                    success_count += 1
                    logger.info(f"Successfully generated persona {i+1}/{num_personas} for {brand_name}")
                    
                except Exception as persona_error:
                    logger.error(f"Error generating persona {i+1}: {str(persona_error)}")
            
            # Create the final results
            simulation_results = {
                "brand_name": brand_name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "individual_personas": individual_personas,
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
        """Generate a single persona's survey response.
        
        This method creates a realistic synthetic persona and generates their response
        to the brand name evaluation. Each persona is based on real-world data and
        responds from their specific perspective.
        """
        # Use the complete schema from __init__ that matches our database
        format_instructions = self.output_parser.get_format_instructions()
        
        # Create the prompt for generating a single persona's response
        prompt_note = """
        IMPORTANT: Generate a SINGLE, REALISTIC persona who will evaluate this brand name.
        The persona must be based on real-world data. For example:
        - If they work at a specific company, use real company data (revenue, size, etc.)
        - Their role, department, and seniority should match real organizational structures
        - Their responses should reflect their specific position and perspective
        
        CRITICAL: Ensure all fields match the required format and database schema:
        - All responses must be from this specific persona's perspective
        - persona_segment must be a detailed TEXT description of their segment
        - emotional_association must be TEXT describing their emotional response
        - psychometric_sentiment_mapping must be a valid JSON object
        - raw_qualitative_feedback must be a valid JSON object
        - strategic_ranking must be an INTEGER between 1 and 10
        - All other fields must be provided and match the database schema types
        """
        
        # Use the existing templates from _load_prompts()
        persona_response_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt.template),
            ("human", self.simulation_prompt.template + prompt_note)
        ])
        
        # Format the prompt with inputs
        prompt_inputs = {
            "format_instructions": format_instructions,
            "brand_name": brand_name,
            "brand_context": brand_context,
            "target_audience": target_audience,
            "brand_values": brand_values,
            "competitive_analysis": competitive_analysis
        }
        
        formatted_prompt = persona_response_prompt.format_prompt(**prompt_inputs)
        
        # Generate the persona's response
        logger.info(f"Generating persona response for {brand_name}")
        response = await self.llm.ainvoke(formatted_prompt.to_messages())
        
        try:
            # Parse the response using our complete schema parser
            persona_response = self.output_parser.parse(response.content)
            logger.info(f"Successfully parsed persona response for {brand_name}")
            return persona_response
            
        except Exception as e:
            logger.warning(f"Error parsing persona response: {str(e)}")
            logger.warning("Attempting manual extraction...")
            return self._manual_extract_fields(response.content)
    
    async def _store_single_persona(self, run_id: str, overall_results: Dict[str, Any], persona: Dict[str, Any]) -> None:
        """Store a single persona with complete field validation."""
        if not self.supabase:
            logger.warning("Supabase not available, skipping storage")
            return
        
        try:
            # Create base data structure
            base_data = {
                "run_id": run_id,
                "brand_name": overall_results.get("brand_name", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Normalize all fields
            normalized_data = self._normalize_survey_output({**overall_results, **persona})
            base_data.update(normalized_data)
            
            # Final validation before storage
            required_fields = [
                "persona_segment", "emotional_association", "psychometric_sentiment_mapping",
                "qualitative_feedback_summary", "raw_qualitative_feedback",
                "final_survey_recommendation", "strategic_ranking"
            ]
            
            missing_fields = [field for field in required_fields if not base_data.get(field)]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Store in database
            await self.supabase.execute_with_retry(
                operation="insert",
                table="survey_simulation",
                data=base_data
            )
            
            logger.info(f"Successfully stored complete persona for {base_data['brand_name']}")
            
        except Exception as e:
            error_msg = f"Error storing persona: {str(e)}"
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
        """Normalize and validate all survey output fields."""
        normalized = {}
        
        # Helper function to ensure text field has a value
        def ensure_text_field(field: str) -> str:
            if field not in output or not output[field]:
                return "Not specified"  # Default value for missing text fields
            return str(output[field])
        
        # Helper function to ensure numeric field
        def ensure_numeric_field(field: str, min_val: float = None, max_val: float = None) -> float:
            try:
                value = float(output.get(field, 0))
                if min_val is not None:
                    value = max(min_val, value)
                if max_val is not None:
                    value = min(max_val, value)
                return value
            except (ValueError, TypeError):
                return 0.0  # Default value for numeric fields
        
        # Helper function to ensure JSONB field
        def ensure_jsonb_field(field: str) -> Dict:
            if field not in output or not output[field]:
                return {"status": "Not specified"}  # Default value for JSON fields
            if isinstance(output[field], str):
                try:
                    return json.loads(output[field])
                except json.JSONDecodeError:
                    return {"value": output[field]}
            return output[field] if isinstance(output[field], dict) else {"value": str(output[field])}

        # Required NOT NULL fields
        normalized["run_id"] = output["run_id"]
        normalized["brand_name"] = output["brand_name"]
        normalized["persona_segment"] = ensure_text_field("persona_segment")
        normalized["emotional_association"] = ensure_text_field("emotional_association")
        normalized["psychometric_sentiment_mapping"] = ensure_jsonb_field("psychometric_sentiment_mapping")
        normalized["qualitative_feedback_summary"] = ensure_text_field("qualitative_feedback_summary")
        normalized["raw_qualitative_feedback"] = ensure_jsonb_field("raw_qualitative_feedback")
        normalized["final_survey_recommendation"] = ensure_text_field("final_survey_recommendation")
        normalized["strategic_ranking"] = int(ensure_numeric_field("strategic_ranking", 1, 10))

        # Numeric fields (0-10 scale)
        for field in ["brand_promise_perception_score", "personality_fit_score", 
                     "competitive_differentiation_score", "competitor_benchmarking_score",
                     "simulated_market_adoption_score"]:
            normalized[field] = ensure_numeric_field(field, 0, 10)

        # Other numeric fields
        normalized["years_of_experience"] = ensure_numeric_field("years_of_experience")
        normalized["company_revenue"] = ensure_numeric_field("company_revenue")
        normalized["confidence_score_persona_accuracy"] = ensure_numeric_field("confidence_score_persona_accuracy", 1, 10)

        # JSONB fields
        normalized["current_brand_relationships"] = ensure_jsonb_field("current_brand_relationships")
        normalized["purchase_triggers_events"] = ensure_jsonb_field("purchase_triggers_events")
        normalized["data_sources_persona_creation"] = ensure_jsonb_field("data_sources_persona_creation")

        # Boolean field
        normalized["decision_maker"] = bool(output.get("decision_maker", False))

        # Text array field
        if "behavioral_tags_keywords" in output and isinstance(output["behavioral_tags_keywords"], list):
            normalized["behavioral_tags_keywords"] = [str(tag) for tag in output["behavioral_tags_keywords"]]
        else:
            normalized["behavioral_tags_keywords"] = []

        # All other text fields
        text_fields = [
            "industry", "company_size_employees", "market_share", "company_structure",
            "geographic_location", "technology_stack", "company_growth_stage", "job_title",
            "seniority", "department", "education_level", "goals_and_challenges",
            "values_and_priorities", "decision_making_style", "information_sources",
            "pain_points", "technological_literacy", "attitude_towards_risk",
            "purchasing_behavior", "online_behavior", "interaction_with_brand",
            "professional_associations", "technical_skills", "networking_habits",
            "professional_aspirations", "event_attendance", "content_consumption_habits",
            "vendor_relationship_preferences", "business_chemistry", "reports_to",
            "buying_group_structure", "company_focus", "company_maturity",
            "budget_authority", "preferred_vendor_size", "innovation_adoption",
            "key_performance_indicators", "professional_development_interests",
            "social_media_usage", "work_life_balance_priorities", "frustrations_annoyances",
            "personal_aspirations_life_goals", "motivations", "product_adoption_lifecycle_stage",
            "success_metrics_product_service", "channel_preferences_brand_interaction",
            "barriers_to_adoption", "generation_age_range", "company_culture_values",
            "industry_sub_vertical", "persona_archetype_type"
        ]
        
        for field in text_fields:
            normalized[field] = ensure_text_field(field)

        # Special handling for varchar(255) field
        influence = ensure_text_field("influence_within_company")
        normalized["influence_within_company"] = influence[:255] if len(influence) > 255 else influence

        return normalized 

    async def _generate_persona_response(
        self,
        brand_name: str,
        brand_context: str,
        target_audience: str,
        brand_values: str,
        competitive_analysis: str,
        persona_number: int = 1,
        total_personas: int = 50
    ) -> Dict[str, Any]:
        """Generate a single persona's survey response.
        
        This method creates a realistic synthetic persona and generates their response
        to the brand name evaluation. Each persona is based on real-world data and
        responds from their specific perspective.
        
        Args:
            brand_name: The brand name to evaluate
            brand_context: Context about the brand
            target_audience: Target audience description
            brand_values: Brand values
            competitive_analysis: Competitive analysis data
            persona_number: Current persona number being generated
            total_personas: Total number of personas to generate
        """
        # Use the complete schema from __init__ that matches our database
        format_instructions = self.output_parser.get_format_instructions()
        
        # Create the prompt for generating a single persona's response
        prompt_note = """
        IMPORTANT: Generate a SINGLE, REALISTIC persona who will evaluate this brand name.
        The persona must be based on real-world data. For example:
        - If they work at a specific company, use real company data (revenue, size, etc.)
        - Their role, department, and seniority should match real organizational structures
        - Their responses should reflect their specific position and perspective
        
        CRITICAL: Ensure all fields match the required format and database schema:
        - All responses must be from this specific persona's perspective
        - persona_segment must be a detailed TEXT description of their segment
        - emotional_association must be TEXT describing their emotional response
        - psychometric_sentiment_mapping must be a valid JSON object
        - raw_qualitative_feedback must be a valid JSON object
        - strategic_ranking must be an INTEGER between 1 and 10
        - All other fields must be provided and match the database schema types
        """
        
        # Format the prompt with inputs
        prompt_inputs = {
            "format_instructions": format_instructions,
            "brand_name": brand_name,
            "brand_context": brand_context,
            "target_audience": target_audience,
            "brand_values": brand_values,
            "competitive_analysis": competitive_analysis,
            "persona_number": persona_number,
            "total_personas": total_personas,
            "persona_id": f"{brand_name.replace(' ', '')}-{persona_number:03d}"
        }
        
        # Use the single persona template
        formatted_prompt = self.single_persona_template.format_prompt(**prompt_inputs)
        
        # Generate the persona's response
        logger.info(f"Generating persona {persona_number}/{total_personas} for {brand_name}")
        response = await self.llm.ainvoke(formatted_prompt.to_messages())
        
        try:
            # Parse the response using our complete schema parser
            persona_response = self.output_parser.parse(response.content)
            logger.info(f"Successfully parsed persona response for {brand_name}")
            return persona_response
            
        except Exception as e:
            logger.warning(f"Error parsing persona response: {str(e)}")
            logger.warning("Attempting manual extraction...")
            return self._manual_extract_fields(response.content) 