"""Market research expert for evaluating brand names."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import asyncio
import re

from langchain.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.messages import HumanMessage, SystemMessage
from postgrest.exceptions import APIError

from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.settings import settings

logger = get_logger(__name__)

class MarketResearchExpert:
    """Expert in market research and brand name evaluation."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the MarketResearchExpert with necessary configurations."""
        # Initialize Supabase client
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "market_research"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.research_prompt = load_prompt(str(prompt_dir / "research.yaml"))
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
        # Initialize Gemini model with tracing
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.model_name,  # Use the correct model name from settings
                temperature=0.3,  # Lower temperature for more consistent analysis
                google_api_key=settings.gemini_api_key,  # Use the correct API key from settings
                convert_system_message_to_human=True,
                top_k=40,
                top_p=0.95
            )
            logger.info("Successfully initialized ChatGoogleGenerativeAI for Market Research Expert")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}")
            raise ValueError(f"Could not initialize LLM: {str(e)}")
            
        # Initialize output parser for structured data
        try:
            self.output_parser = self._create_output_parser()
            logger.info("Successfully initialized output parser for market research")
        except Exception as e:
            logger.error(f"Failed to initialize output parser: {str(e)}")
            raise

    async def analyze_market_potential(
        self,
        run_id: str,
        brand_names: List[str],
        brand_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze the market potential of shortlisted brand names.
        
        Args:
            run_id: Unique run identifier
            brand_names: List of shortlisted brand names
            brand_context: Dictionary containing brand context information
            
        Returns:
            List of dictionaries with market research analysis results
        """
        # Log the start of analysis
        logger.info(f"Analyzing market potential for {len(brand_names)} shortlisted brand names")
        
        # Check if we have brand names to analyze
        if not brand_names:
            logger.warning("No brand names provided for market research analysis")
            return []
        
        try:
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format())
            
            # Log the names being analyzed
            logger.info(f"Shortlisted names for analysis: {brand_names}")
            
            # For 3 or fewer names, analyze each individually for deeper analysis
            if len(brand_names) <= 3:
                logger.info(f"Performing individual deep analysis for {len(brand_names)} shortlisted names")
                all_results = []
                
                for brand_name in brand_names:
                    # Format the research prompt for a single name
                    formatted_prompt = self.research_prompt.format(
                        format_instructions=self.output_parser.get_format_instructions(),
                        brand_names=[brand_name],  # Analyze one shortlisted name at a time
                        brand_context=brand_context
                    )
                    
                    # Create human message
                    human_message = HumanMessage(content=formatted_prompt)
                    
                    # Get response from LLM
                    try:
                        response = await self.llm.ainvoke([system_message, human_message])
                        logger.info(f"Got response for brand name: {brand_name}")
                        
                        # Parse the structured output
                        try:
                            parsed_output = self.output_parser.parse(response.content)
                            parsed_output["brand_name"] = brand_name
                            parsed_output["run_id"] = run_id
                            parsed_output["timestamp"] = datetime.now().isoformat()
                            parsed_output["version"] = "1.0"  # Hardcoded version since it's not in settings
                            
                            # Store in Supabase
                            if self.supabase:
                                await self._store_in_supabase(run_id, parsed_output)
                                
                            all_results.append(parsed_output)
                            logger.info(f"Successfully processed market research for {brand_name}")
                        except Exception as parse_error:
                            logger.error(f"Error parsing market research results for {brand_name}: {str(parse_error)}")
                            # Create a minimal result for this name
                            minimal_result = {
                                "brand_name": brand_name,
                                "run_id": run_id,
                                "timestamp": datetime.now().isoformat(),
                                "version": "1.0",  # Hardcoded version
                                "market_opportunity": "Error parsing analysis",
                                "target_audience_fit": "Error parsing analysis",
                                "competitive_analysis": "Error parsing analysis",
                                "market_viability": "Error parsing analysis",
                                "overall_market_score": 5.0
                            }
                            all_results.append(minimal_result)
                    except Exception as llm_error:
                        logger.error(f"LLM invocation error for {brand_name}: {str(llm_error)}")
                        # Add a placeholder result
                        all_results.append({
                            "brand_name": brand_name,
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
                            "version": "1.0",  # Hardcoded version
                            "market_opportunity": "Unable to analyze due to LLM error",
                            "target_audience_fit": "Unable to analyze due to LLM error",
                            "competitive_analysis": "Unable to analyze due to LLM error",
                            "market_viability": "Unable to analyze due to LLM error",
                            "overall_market_score": 5.0
                        })
                    
                    # Brief pause between requests to avoid rate limiting
                    await asyncio.sleep(1.0)
                
                return all_results
            else:
                # For more names, analyze them as a batch
                logger.info(f"Performing batch analysis for {len(brand_names)} shortlisted names")
                
                # Format the research prompt for all names at once
                formatted_prompt = self.research_prompt.format(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_names=brand_names,
                    brand_context=brand_context
                )
                
                # Create human message
                human_message = HumanMessage(content=formatted_prompt)
                
                # Get response from LLM
                response = await self.llm.ainvoke([system_message, human_message])
                
                # Parse the response - since we may get multiple analyses
                content = response.content
                
                # Handle cases where we get a single analysis vs multiple
                results = []
                
                try:
                    # Extract JSON content if wrapped in markdown code blocks
                    if "```json" in content:
                        json_matches = re.findall(r'```json\n(.*?)\n```', content, re.DOTALL)
                        if json_matches:
                            content = json_matches[0]
                            
                    # Try parsing as a JSON array directly
                    try:
                        json_data = json.loads(content)
                        if isinstance(json_data, list):
                            results = json_data
                        else:
                            # Single result
                            results = [json_data]
                    except json.JSONDecodeError:
                        # If not a JSON array, try parsing as individual JSON objects
                        # This might happen if the model returns multiple JSON objects
                        if len(brand_names) > 1:
                            # Try to split by brand name and parse each section
                            for brand_name in brand_names:
                                brand_pattern = re.compile(
                                    rf'(?:Analysis for|Brand Name:|\*\*)\s*{re.escape(brand_name)}.*?'
                                    r'(.*?)(?=(?:Analysis for|Brand Name:|\*\*)\s*\w+|\Z)',
                                    re.DOTALL | re.IGNORECASE
                                )
                                matches = brand_pattern.findall(content)
                                if matches:
                                    for match in matches:
                                        # Try to extract JSON from this section
                                        json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
                                        json_matches = json_pattern.findall(match)
                                        if json_matches:
                                            for json_str in json_matches:
                                                try:
                                                    parsed = json.loads(json_str)
                                                    parsed["brand_name"] = brand_name
                                                    results.append(parsed)
                                                    break  # Take the first valid JSON for this brand
                                                except json.JSONDecodeError:
                                                    continue
                        
                        # If we still don't have parsed outputs, fall back to the output parser
                        if not results:
                            # Use the output parser as fallback for a single analysis
                            parsed = self.output_parser.parse(content)
                            if brand_names and len(brand_names) == 1:
                                parsed["brand_name"] = brand_names[0]
                            results = [parsed]
                    
                    # Process each parsed output
                    for idx, parsed_output in enumerate(results):
                        # Make sure we have a brand name
                        if "brand_name" not in parsed_output and brand_names:
                            if len(results) == len(brand_names):
                                # Assign brand names in order
                                parsed_output["brand_name"] = brand_names[idx]
                            elif idx < len(brand_names):
                                # Assign by index if possible
                                parsed_output["brand_name"] = brand_names[idx]
                            else:
                                # Just use the first name as fallback
                                parsed_output["brand_name"] = brand_names[0]
                        
                        # Add common fields
                        parsed_output["run_id"] = run_id
                        parsed_output["timestamp"] = datetime.now().isoformat()
                        parsed_output["version"] = "1.0"  # Hardcoded version since it's not in settings
                        
                        # Store in Supabase
                        if self.supabase and "brand_name" in parsed_output:
                            await self._store_in_supabase(run_id, parsed_output)
                    
                    return results
                except Exception as parsing_error:
                    logger.error(f"Error parsing multiple analyses: {str(parsing_error)}")
                    # Fall back to creating minimal results for each name
                    return [{
                        "brand_name": name,
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(),
                        "version": "1.0",  # Hardcoded version
                        "market_opportunity": "Error parsing analysis",
                        "target_audience_fit": "Error parsing analysis",
                        "competitive_analysis": "Error parsing analysis",
                        "market_viability": "Error parsing analysis",
                        "overall_market_score": 5.0
                    } for name in brand_names]
        except Exception as e:
            logger.error(f"Failed to analyze market potential: {str(e)}")
            # Return minimal placeholder results
            return [{
                "brand_name": name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",  # Hardcoded version
                "market_opportunity": "Failed to analyze due to error",
                "target_audience_fit": "Failed to analyze due to error",
                "competitive_analysis": "Failed to analyze due to error",
                "market_viability": "Failed to analyze due to error",
                "overall_market_score": 5.0
            } for name in brand_names]

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
                "timestamp": analysis_data.get("timestamp", datetime.now().isoformat()),
                "version": analysis_data.get("version", "1.0"),
                
                # Core analysis fields
                "market_opportunity": analysis_data.get("market_opportunity"),
                "target_audience_fit": analysis_data.get("target_audience_fit"),
                "competitive_analysis": analysis_data.get("competitive_analysis"),
                "market_viability": analysis_data.get("market_viability"),
                
                # Numeric scores
                "market_opportunity_score": float(analysis_data.get("market_opportunity_score", 5.0)),
                "target_audience_score": float(analysis_data.get("target_audience_score", 5.0)),
                "competitive_advantage_score": float(analysis_data.get("competitive_advantage_score", 5.0)),
                "scalability_score": float(analysis_data.get("scalability_score", 5.0)),
                "overall_market_score": float(analysis_data.get("overall_market_score", 5.0)),
                
                # Industry information
                "industry_name": analysis_data.get("industry_name"),
                "market_size": analysis_data.get("market_size"),
                "market_growth_rate": analysis_data.get("market_growth_rate"),
                
                # Competitive landscape
                "key_competitors": self._ensure_array(analysis_data.get("key_competitors", [])),
                
                # Customer information
                "target_customer_segments": self._ensure_array(analysis_data.get("target_customer_segments", [])),
                "customer_pain_points": self._ensure_array(analysis_data.get("customer_pain_points", [])),
                
                # Market entry considerations
                "market_entry_barriers": analysis_data.get("market_entry_barriers"),
                "regulatory_considerations": analysis_data.get("regulatory_considerations"),
                
                # Future outlook
                "emerging_trends": analysis_data.get("emerging_trends"),
                
                # Additional analysis
                "potential_risks": analysis_data.get("potential_risks"),
                "recommendations": analysis_data.get("recommendations")
            }
            
            # Remove None values to prevent database errors
            supabase_data = {k: v for k, v in supabase_data.items() if v is not None}
            
            # Store in Supabase using the client
            await self.supabase.execute_with_retry(
                operation="insert",
                table="market_research",
                data=supabase_data
            )
            
            logger.info(f"Stored market research analysis for '{analysis_data['brand_name']}' with run_id '{run_id}'")
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                "Error preparing market research data for Supabase",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "brand_name": analysis_data.get("brand_name", "unknown")
                }
            )
            raise ValueError(f"Error preparing market research data: {str(e)}")
            
        except (APIError) as e:
            logger.error(
                "Error storing market research in Supabase",
                extra={
                    "run_id": run_id,
                    "error": str(e),
                    "brand_name": supabase_data.get("brand_name", "unknown")
                }
            )
            raise
    
    def _ensure_array(self, value):
        """Safely convert different data types to arrays for PostgreSQL."""
        # If already a list, return as is
        if isinstance(value, list):
            return value
            
        # If None, return an empty list
        if value is None:
            return []
            
        # If a string, try to parse as JSON first
        if isinstance(value, str):
            try:
                # Try parsing as JSON
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
                else:
                    # If it parsed but not as a list, treat it as a single-item list
                    return [value]
            except:
                # If not valid JSON, split by commas for comma-separated values
                if "," in value:
                    return [item.strip() for item in value.split(",") if item.strip()]
                else:
                    # Single value
                    return [value]
                
        # Fallback for any other type
        return [str(value)] 

    def _create_output_parser(self):
        """Create the output parser for structured data."""
        # Define the schemas for the output parser
        schemas = [
            ResponseSchema(
                name="brand_name",
                description="The brand name being analyzed"
            ),
            ResponseSchema(
                name="industry_name",
                description="Specific industry category the brand name is best suited for"
            ),
            ResponseSchema(
                name="market_size",
                description="Estimated size of the market in specific terms (e.g. '$X billion TAM')"
            ),
            ResponseSchema(
                name="market_growth_rate", 
                description="Growth rate of the market (e.g. '15% CAGR through 2026')"
            ),
            ResponseSchema(
                name="key_competitors",
                description="List of 3-5 key competitors in the market space"
            ),
            ResponseSchema(
                name="target_customer_segments",
                description="List of 3-5 primary customer segments for this brand"
            ),
            ResponseSchema(
                name="customer_pain_points",
                description="List of 3-5 specific customer pain points this brand could address"
            ),
            ResponseSchema(
                name="market_entry_barriers",
                description="Description of barriers to market entry and possible mitigation strategies"
            ),
            ResponseSchema(
                name="regulatory_considerations",
                description="Any regulatory issues or compliance requirements to consider"
            ),
            ResponseSchema(
                name="emerging_trends",
                description="2-3 key emerging trends in this market space"
            ),
            ResponseSchema(
                name="overall_market_score",
                description="Overall market viability score on scale of 1-10"
            ),
            ResponseSchema(
                name="market_opportunity_score",
                description="Market opportunity score on scale of 1-10"
            ),
            ResponseSchema(
                name="target_audience_score",
                description="Target audience fit score on scale of 1-10"
            ),
            ResponseSchema(
                name="competitive_advantage_score",
                description="Competitive advantage score on scale of 1-10"
            ),
            ResponseSchema(
                name="scalability_score",
                description="Business scalability score on scale of 1-10"
            ),
            ResponseSchema(
                name="potential_risks",
                description="Specific potential risks or challenges for this brand in the market"
            ),
            ResponseSchema(
                name="recommendations",
                description="Strategic recommendations for market entry and positioning"
            ),
            ResponseSchema(
                name="market_opportunity",
                description="Analysis of the market opportunity for this brand name"
            ),
            ResponseSchema(
                name="target_audience_fit",
                description="Analysis of how well the brand fits the target audience"
            ),
            ResponseSchema(
                name="competitive_analysis",
                description="Analysis of the competitive landscape and the brand's position within it"
            ),
            ResponseSchema(
                name="market_viability",
                description="Assessment of the brand's long-term market viability and potential"
            )
        ]
        
        return StructuredOutputParser.from_response_schemas(schemas) 