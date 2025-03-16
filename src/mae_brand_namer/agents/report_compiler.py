"""Report Compiler for processing and storing brand name analysis data."""

# Standard library imports
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
import asyncio

# Third-party imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from postgrest import APIError

# Local application imports
from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)


class ReportCompiler:
    """Expert in processing and storing brand name analysis data.
    
    This class follows a three-step process:
    1. Pull filtered data from Supabase tables
    2. Process data using LLM to generate structured insights
    3. Store processed data in the report_raw_data table
    """

    def __init__(
        self,
        dependencies: Optional[Dependencies] = None,
        supabase: Optional[SupabaseManager] = None,
    ):
        """Initialize the ReportCompiler with dependencies."""
        if dependencies:
            self.supabase = dependencies.supabase
        else:
            logger.info("Initializing ReportCompiler with direct Supabase connection")
            self.supabase = supabase or SupabaseManager()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=settings.model_temperature,
            convert_system_message_to_human=True,
            google_api_key=settings.gemini_api_key,  # Use the correct API key from settings
            top_k=40,
            top_p=0.95
        )
        
        # Load prompt templates
        try:
            self.system_prompt = load_prompt("src/mae_brand_namer/agents/prompts/report_compiler/system.yaml")
            self.compilation_prompt = load_prompt("src/mae_brand_namer/agents/prompts/report_compiler/compilation.yaml")
            logger.info("Successfully loaded report compiler prompt templates")
        except Exception as e:
            logger.error(f"Error loading prompt templates: {str(e)}")
            # Create fallback prompts
            self.system_prompt = ChatPromptTemplate.from_template(
                "You are a Report Compilation Expert specializing in brand naming analysis reports."
            )
            self.compilation_prompt = ChatPromptTemplate.from_template(
                "Compile a comprehensive brand naming report using the following workflow data:\n\n"
                "State Data:\n{state_data}\n\nFormat your report according to this schema:\n{format_instructions}"
            )

    async def compile_report(
        self,
        run_id: str,
        state: Dict[str, Any] = None,  # Make state optional
        user_prompt: Optional[str] = None,
        store_raw_only: bool = False
    ) -> Dict[str, Any]:
        """Main method to compile report data.
        
        Args:
            run_id: Unique identifier for the report generation run
            state: Optional workflow state (not used for data retrieval)
            user_prompt: Optional user prompt for customization
            store_raw_only: If True, only store raw data without processing
            
        Returns:
            Dict containing processing status and results
        """
        try:
            # Start logging the process
            await self._log_overall_process(run_id, "started")
            
            # Process each section
            sections_processed = []
            for section_name in self._get_required_sections():
                try:
                    logger.info(f"Processing section: {section_name}")
                    
                    # Fetch raw data
                    raw_data = await self._fetch_section_data(run_id, section_name)
                    
                    if not raw_data:
                        logger.warning(f"No data found for section {section_name}")
                        continue
                        
                    if store_raw_only:
                        # Store raw data directly
                        success = await self._store_raw_section_data(run_id, section_name, raw_data)
                        if success:
                            sections_processed.append(section_name)
                            logger.info(f"Successfully stored raw data for section {section_name}")
                    else:
                        # Process and store data
                        processed_data = await self._generate_section(run_id, section_name, raw_data, user_prompt)
                        success = await self._store_raw_section_data(run_id, section_name, processed_data)
                        if success:
                            sections_processed.append(section_name)
                            logger.info(f"Successfully processed and stored data for section {section_name}")
                
                except Exception as e:
                    logger.error(f"Error processing section {section_name}: {str(e)}")
                    continue
            
            # Log completion
            await self._log_overall_process(run_id, "completed")
            
            # Verify all required sections were processed
            required_sections = set(self._get_required_sections())
            processed_sections = set(sections_processed)
            missing_sections = required_sections - processed_sections
            
            status = "success" if not missing_sections else "partial"
            result = {
                "status": status,
                "sections_processed": sections_processed,
                "run_id": run_id
            }
            
            if missing_sections:
                result["missing_sections"] = list(missing_sections)
                logger.warning(f"Some sections were not processed: {', '.join(missing_sections)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in compile_report: {str(e)}")
            await self._log_overall_process(run_id, "failed", error_message=str(e))
            return {
                "status": "error",
                "error": str(e),
                "run_id": run_id
            }

    def _get_required_sections(self) -> List[str]:
        """Return list of required report sections in the specified order."""
        return [
            "brand_context",
            "brand_name_generation",
            "semantic_analysis",
            "linguistic_analysis",
            "cultural_sensitivity_analysis",
            "brand_name_evaluation",
            "translation_analysis",
            "market_research",
            "competitor_analysis",
            "domain_analysis",
            "survey_simulation",
            "seo_online_discoverability"
        ]

    async def _fetch_section_data(self, run_id: str, section_name: str) -> Dict[str, Any]:
        """Fetch raw data for a specific section from Supabase.
        
        Args:
            run_id: Unique identifier for the report generation run
            section_name: Name of the section to fetch data for
            
        Returns:
            Dict containing the raw data for the section
        """
        try:
            if section_name == "brand_context":
                # Fetch brand context data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "brand_context",
                    {
                        "run_id": run_id,
                        "select": "brand_promise,brand_personality,brand_tone_of_voice,brand_values,brand_purpose,brand_mission,target_audience,customer_needs,market_positioning,competitive_landscape,industry_focus,industry_trends,brand_identity_brief",
                        "limit": 1
                    }
                )
                
                if not data:
                    logger.warning(f"No brand context data found for run_id {run_id}")
                    return {}
                
                return {"brand_context": data[0] if data else {}}
                
            elif section_name == "brand_name_generation":
                # Fetch brand name generation data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "brand_name_generation",
                    {
                        "run_id": run_id,
                        "select": "brand_name,naming_category,brand_personality_alignment,brand_promise_alignment,name_generation_methodology,memorability_score_details,pronounceability_score_details,visual_branding_potential_details,target_audience_relevance_details,market_differentiation_details"
                    }
                )
                
                if not data:
                    logger.warning(f"No brand name generation data found for run_id {run_id}")
                    return {}
                
                # Organize by naming category as required in the notepad
                by_category = {}
                for item in data:
                    category = item.get("naming_category", "Uncategorized")
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(item)
                
                return {"brand_name_generation": by_category}
                
            elif section_name == "semantic_analysis":
                # Fetch semantic analysis data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "semantic_analysis",
                    {
                        "run_id": run_id,
                        "select": "brand_name,denotative_meaning,etymology,emotional_valence,brand_personality,sensory_associations,figurative_language,phoneme_combinations,sound_symbolism,alliteration_assonance,word_length_syllables,compounding_derivation,semantic_trademark_risk"
                    }
                )
                
                if not data:
                    logger.warning(f"No semantic analysis data found for run_id {run_id}")
                    return {}
                
                # Structure data with brand names as keys while preserving brand_name
                formatted_data = {}
                for item in data:
                    brand_name = item["brand_name"]
                    formatted_data[brand_name] = item
                
                return {"semantic_analysis": formatted_data}
                
            elif section_name == "linguistic_analysis":
                # Fetch linguistic analysis data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "linguistic_analysis",
                    {
                        "run_id": run_id,
                        "select": "brand_name,pronunciation_ease,euphony_vs_cacophony,rhythm_and_meter,phoneme_frequency_distribution,sound_symbolism,word_class,morphological_transparency,inflectional_properties,ease_of_marketing_integration,naturalness_in_collocations,semantic_distance_from_competitors,neologism_appropriateness,overall_readability_score,notes"
                    }
                )
                
                if not data:
                    logger.warning(f"No linguistic analysis data found for run_id {run_id}")
                    return {}
                
                # Structure data with brand names as keys while preserving brand_name
                formatted_data = {}
                for item in data:
                    brand_name = item["brand_name"]
                    formatted_data[brand_name] = item
                
                return {"linguistic_analysis": formatted_data}
                
            elif section_name == "cultural_sensitivity_analysis":
                # Fetch cultural sensitivity analysis data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "cultural_sensitivity_analysis",
                    {
                        "run_id": run_id,
                        "select": "brand_name,cultural_connotations,symbolic_meanings,alignment_with_cultural_values,religious_sensitivities,social_political_taboos,age_related_connotations,regional_variations,historical_meaning,current_event_relevance,overall_risk_rating,notes"
                    }
                )
                
                if not data:
                    logger.warning(f"No cultural sensitivity analysis data found for run_id {run_id}")
                    return {}
                
                # Structure data with brand names as keys while preserving brand_name
                formatted_data = {}
                for item in data:
                    brand_name = item["brand_name"]
                    formatted_data[brand_name] = item
                
                return {"cultural_sensitivity_analysis": formatted_data}
                
            elif section_name == "brand_name_evaluation":
                # Fetch brand name evaluation data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "brand_name_evaluation",
                    {
                        "run_id": run_id,
                        "select": "brand_name,overall_score,shortlist_status,evaluation_comments",
                        "order": "overall_score.desc"
                    }
                )
                
                if not data:
                    logger.warning(f"No brand name evaluation data found for run_id {run_id}")
                    return {}
                
                # Split into shortlisted and other names, preserving brand_name in each
                shortlisted = []
                other_names = []
                for item in data:
                    if item.get("shortlist_status"):
                        shortlisted.append(item)
                    else:
                        other_names.append(item)
                
                return {
                    "brand_name_evaluation": {
                        "shortlisted_names": shortlisted,
                        "other_names": other_names
                    }
                }
                
            elif section_name == "translation_analysis":
                # Fetch translation analysis data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "translation_analysis",
                    {
                        "run_id": run_id,
                        "select": "brand_name,target_language,direct_translation,semantic_shift,pronunciation_difficulty,phonetic_retention,cultural_acceptability,adaptation_needed,proposed_adaptation,brand_essence_preserved,global_consistency_vs_localization,notes"
                    }
                )
                
                if not data:
                    logger.warning(f"No translation analysis data found for run_id {run_id}")
                    return {}
                
                # Structure data with brand names as keys and languages as a dictionary
                formatted_data = {}
                for item in data:
                    brand_name = item["brand_name"]
                    target_lang = item["target_language"]
                    if brand_name not in formatted_data:
                        formatted_data[brand_name] = {}
                    # Keep brand_name in the item for reference
                    formatted_data[brand_name][target_lang] = item
                
                # Wrap in translation_analysis key
                return {"translation_analysis": formatted_data}
                
            elif section_name == "market_research":
                # Fetch market research data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "market_research",
                    {
                        "run_id": run_id,
                        "select": "brand_name,market_opportunity,target_audience_fit,competitive_analysis,market_viability,potential_risks,recommendations,industry_name,market_size,market_growth_rate,key_competitors,customer_pain_points,market_entry_barriers,emerging_trends"
                    }
                )
                
                if not data:
                    logger.warning(f"No market research data found for run_id {run_id}")
                    return {}
                
                # Structure data as array while preserving brand_name in each entry
                formatted_data = []
                for item in data:
                    formatted_data.append(item)
                
                return {"market_research": formatted_data}
                
            elif section_name == "competitor_analysis":
                # Fetch competitor analysis data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "competitor_analysis",
                    {
                        "run_id": run_id,
                        "select": "brand_name,competitor_name,competitor_positioning,competitor_strengths,competitor_weaknesses,competitor_differentiation_opportunity,risk_of_confusion,target_audience_perception,trademark_conflict_risk"
                    }
                )
                
                if not data:
                    logger.warning(f"No competitor analysis data found for run_id {run_id}")
                    return {}
                
                # Group by brand name while preserving brand_name at top level
                by_brand = {}
                for item in data:
                    brand_name = item["brand_name"]
                    if brand_name not in by_brand:
                        by_brand[brand_name] = {
                            "brand_name": brand_name,  # Keep brand_name at top level
                            "competitors": []
                        }
                    # Remove brand_name from competitor data since it's at top level
                    competitor_data = {k: v for k, v in item.items() if k != "brand_name"}
                    by_brand[brand_name]["competitors"].append(competitor_data)
                
                # Convert to list preserving the structure
                formatted_data = list(by_brand.values())
                
                return {"competitor_analysis": formatted_data}
                
            elif section_name == "domain_analysis":
                # Fetch domain analysis data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "domain_analysis",
                    {
                        "run_id": run_id,
                        "select": "brand_name,domain_exact_match,alternative_tlds,misspellings_variations_available,acquisition_cost,domain_length_readability,hyphens_numbers_present,brand_name_clarity_in_url,social_media_availability,scalability_future_proofing,notes"
                    }
                )
                
                if not data:
                    logger.warning(f"No domain analysis data found for run_id {run_id}")
                    return {}
                
                # Structure data with brand names as keys while preserving brand_name
                formatted_data = {}
                for item in data:
                    brand_name = item["brand_name"]
                    formatted_data[brand_name] = item
                
                return {"domain_analysis": formatted_data}
                
            elif section_name == "survey_simulation":
                # Fetch survey simulation data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "survey_simulation",
                    {
                        "run_id": run_id,
                        "select": "brand_name,brand_promise_perception_score,personality_fit_score,emotional_association,competitive_differentiation_score,competitor_benchmarking_score,simulated_market_adoption_score,qualitative_feedback_summary,raw_qualitative_feedback,final_survey_recommendation"
                    }
                )
                
                if not data:
                    logger.warning(f"No survey simulation data found for run_id {run_id}")
                    return {}
                
                # Structure data as array while preserving brand_name in each entry
                formatted_data = []
                for item in data:
                    formatted_data.append(item)
                
                return {"survey_simulation": formatted_data}
                
            elif section_name == "seo_online_discoverability":
                # Fetch SEO online discoverability data
                data = await self.supabase.execute_with_retry(
                    "select",
                    "seo_online_discoverability",
                    {
                        "run_id": run_id,
                        "select": "brand_name,keyword_alignment,search_volume,keyword_competition,branded_keyword_potential,non_branded_keyword_potential,exact_match_search_results,competitor_domain_strength,name_length_searchability,unusual_spelling_impact,content_marketing_opportunities,social_media_availability,social_media_discoverability,negative_keyword_associations,negative_search_results,seo_viability_score,seo_recommendations"
                    }
                )
                
                if not data:
                    logger.warning(f"No SEO online discoverability data found for run_id {run_id}")
                    return {}
                
                # Structure data as array while preserving brand_name in each entry
                formatted_data = []
                for item in data:
                    formatted_data.append(item)
                
                return {"seo_online_discoverability": formatted_data}
            
            else:
                raise ValueError(f"Unknown section: {section_name}")
                
        except Exception as e:
            logger.error(f"Error fetching data for section {section_name}: {str(e)}")
            raise

    def _get_structure_template(self, section_name: str) -> str:
        """Get the JSON structure template and field descriptions for a specific section.
        
        Args:
            section_name: Name of the section to get template for
            
        Returns:
            String containing the structure template and field descriptions
        """
        try:
            # Get the JSON template
            template_path = f"src/mae_brand_namer/agents/prompts/report_compiler/templates/{section_name}.json"
            json_template = ""
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    json_template = f.read()
            else:
                logger.warning(f"No template file found for section {section_name}")
                return ""

            # Get field descriptions from README.md
            readme_path = "src/mae_brand_namer/agents/prompts/report_compiler/templates/README.md"
            field_descriptions = ""
            if os.path.exists(readme_path):
                with open(readme_path, 'r') as f:
                    content = f.read()
                    # Find the section for this template
                    section_start = content.find(f"## {section_name.replace('_', ' ').title()} Template")
                    if section_start != -1:
                        section_end = content.find("##", section_start + 1)
                        if section_end == -1:  # If it's the last section
                            section_end = len(content)
                        field_descriptions = content[section_start:section_end].strip()

            # Combine template and descriptions
            return f"""Structure Template:
{json_template}

Field Descriptions:
{field_descriptions}"""

        except Exception as e:
            logger.error(f"Error loading template for section {section_name}: {str(e)}")
            return ""

    async def _generate_section(
        self,
        run_id: str,
        section_name: str,
        section_data: Dict[str, Any],
        user_prompt: Optional[str] = None  # Keep this for potential future use
    ) -> Dict[str, Any]:
        """Process raw data to ensure consistent structure for the formatter.
        
        Args:
            run_id: Unique identifier for the report generation run
            section_name: Name of the section being processed
            section_data: Raw data for the section organized by table names
            user_prompt: Optional user prompt for customization (not used for structure template)
            
        Returns:
            Dict containing the structured data in a format ready for the formatter
        """
        try:
            logger.info(f"Processing section {section_name} for consistent structure")
            
            # Get the template
            structure_template = self._get_structure_template(section_name)
            
            # If data structure is already well-formed or no template exists, skip LLM processing
            if not structure_template or self._is_data_well_structured(section_data, section_name):
                logger.info(f"Data for {section_name} is already well-structured, skipping LLM processing")
                return section_data
                
            # Use the LLM to ensure data follows the proper structure
            # Create messages for the LLM
            system_message = SystemMessage(content=self.system_prompt.format())
            
            # Use the compilation prompt to format the section data
            compilation_messages = self.compilation_prompt.format_messages(
                section_name=section_name,
                section_data=section_data,
                structure_template=structure_template  # Use the correct parameter name
            )
            
            # Combine messages
            messages = [system_message] + compilation_messages
            
            # Invoke LLM
            response = await self.llm.ainvoke(messages)
            
            try:
                # Parse the response as JSON
                structured_data = json.loads(response.content)
                logger.info(f"Successfully structured data for section {section_name}")
                return structured_data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON for {section_name}: {e}")
                # If parsing fails, just return the original data
                return section_data
                
        except Exception as e:
            logger.error(f"Error in _generate_section for {section_name}: {str(e)}")
            # Return the original data if processing fails
            return section_data
            
    def _is_data_well_structured(self, data: Dict[str, Any], section_name: str) -> bool:
        """Check if the data is already properly structured.
        
        Args:
            data: The data to check
            section_name: The name of the section
            
        Returns:
            bool indicating if the data is already well-structured
        """
        # Check if data is empty
        if not data:
            return False
            
        # Check if the data has the expected top-level key
        if section_name not in data:
            return False
            
        # For brand_name_generation, check if it has the expected structure
        if section_name == "brand_name_generation":
            if not isinstance(data.get(section_name), dict):
                return False
                
            # Check if we have categories with lists of names
            for category, names in data.get(section_name, {}).items():
                if not isinstance(names, list):
                    return False
                    
        # For brand_name_evaluation, check for shortlisted and other names arrays
        elif section_name == "brand_name_evaluation":
            eval_data = data.get(section_name)
            if not isinstance(eval_data, dict):
                return False
            if "shortlisted_names" not in eval_data or "other_names" not in eval_data:
                return False
            if not isinstance(eval_data["shortlisted_names"], list) or not isinstance(eval_data["other_names"], list):
                return False
                
        # For translation_analysis and competitor_analysis, check nested array structure
        elif section_name in ["translation_analysis", "competitor_analysis"]:
            section_data = data.get(section_name)
            if not isinstance(section_data, list):
                return False
                
            # Check each brand entry has the correct structure
            for brand_entry in section_data:
                if not isinstance(brand_entry, dict):
                    return False
                if "brand_name" not in brand_entry:
                    return False
                    
                # Check nested arrays
                if section_name == "translation_analysis":
                    if "languages" not in brand_entry or not isinstance(brand_entry["languages"], list):
                        return False
                elif section_name == "competitor_analysis":
                    if "competitors" not in brand_entry or not isinstance(brand_entry["competitors"], list):
                        return False
                        
        # For all other sections, check if it's an array of objects with brand_name field
        else:
            section_data = data.get(section_name)
            if not isinstance(section_data, list):
                return False
                
            # Check each entry has brand_name field
            for entry in section_data:
                if not isinstance(entry, dict) or "brand_name" not in entry:
                    return False
                    
        # Data passes all checks
        return True

    async def _store_raw_section_data(
        self,
        run_id: str,
        section_name: str,
        section_data: Dict[str, Any]
    ) -> bool:
        """Store section data in the report_raw_data table.
        
        Args:
            run_id: Unique identifier for the report generation run
            section_name: Name of the section being stored
            section_data: Data to store
            
        Returns:
            bool indicating success or failure
        """
        try:
            logger.info(f"Storing data for section {section_name}")
            
            # Check if entry exists
            existing = await self.supabase.execute_with_retry(
                "select",
                "report_raw_data",
                {
                    "run_id": f"eq.{run_id}",
                    "section_name": f"eq.{section_name}",
                    "select": "id"
                }
            )
            
            if existing:
                # Update existing entry
                logger.info(f"Updating existing entry for section {section_name}")
                await self.supabase.execute_with_retry(
                    "update",
                    "report_raw_data",
                    {
                        "raw_data": section_data,  # Correct column name is raw_data, not section_data
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    },
                    {
                        "run_id": f"eq.{run_id}",
                        "section_name": f"eq.{section_name}"
                    }
                )
                logger.info(f"Updated existing data for section {section_name}")
            else:
                # Insert new entry
                logger.info(f"Inserting new entry for section {section_name}")
                await self.supabase.execute_with_retry(
                    "insert",
                    "report_raw_data",
                    {
                        "run_id": run_id,
                        "section_name": section_name,
                        "raw_data": section_data  # Correct column name is raw_data, not section_data
                    }
                )
                logger.info(f"Inserted new data for section {section_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing data for section {section_name}: {str(e)}")
            return False

    async def _log_overall_process(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Log the overall process status in the process_logs table.
        
        Args:
            run_id: Unique identifier for the report generation run
            status: Current status of the process
            error_message: Optional error message if status is 'failed'
        """
        try:
            now = datetime.now(timezone.utc)
            
            # Check if entry exists
            existing = await self.supabase.execute_with_retry(
                "select",
                "process_logs",
                {
                    "run_id": f"eq.{run_id}",
                    "process_type": "eq.report_compilation",
                    "select": "id"
                }
            )
            
            data = {
                "run_id": run_id,
                "status": status,
                "process_type": "report_compilation",
                "error_message": error_message
            }
            
            # Add timestamps based on status
            if status == "started":
                data["start_time"] = now.isoformat()
            elif status in ["completed", "failed"]:
                data["end_time"] = now.isoformat()
            
            if existing:
                # Update existing entry
                await self.supabase.execute_with_retry(
                    "update",
                    "process_logs",
                    data,
                    {
                        "run_id": f"eq.{run_id}",
                        "process_type": "eq.report_compilation"
                    }
                )
            else:
                # Insert new entry
                await self.supabase.execute_with_retry(
                    "insert",
                    "process_logs",
                    data
                )
            
        except Exception as e:
            logger.error(f"Error logging process status: {str(e)}")
