"""Expert in analyzing SEO potential and online discoverability of brand names."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio
import requests
import aiohttp
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)


class SEOOnlineDiscoveryExpert:
    """Expert in analyzing SEO potential and online discoverability of brand names.
    
    This expert evaluates brand names for their search engine optimization potential,
    online visibility, and digital marketing effectiveness across various platforms.
    
    Attributes:
        supabase: Supabase client for data storage
        langsmith: LangSmith client for tracing (optional)
        role (str): The expert's role identifier
        goal (str): The expert's primary objective
        system_prompt: Loaded system prompt template
        output_schemas (List[ResponseSchema]): Schemas for structured output parsing
        prompt (ChatPromptTemplate): Configured prompt template
    """
    
    def __init__(self, dependencies: Dependencies) -> None:
        """Initialize the SEOOnlineDiscoveryExpert with dependencies.
        
        Args:
            dependencies: Container for application dependencies
        """
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        # Agent identity
        self.role = "SEO & Online Discovery Expert"
        self.goal = (
            "Evaluate brand names for their search engine optimization potential "
            "and online discoverability across multiple search contexts."
        )
        
        # Load prompts from YAML files
        prompt_dir = Path(__file__).parent / "prompts" / "seo_discovery"
        self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
        self.human_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(
                name="keyword_alignment",
                description="How well the name aligns with target keywords",
                type="string"
            ),
            ResponseSchema(
                name="search_volume",
                description="Estimated monthly search volume",
                type="float"
            ),
            ResponseSchema(
                name="keyword_competition",
                description="Level of competition for keywords",
                type="string"
            ),
            ResponseSchema(
                name="branded_keyword_potential",
                description="Potential for branded keyword success",
                type="string"
            ),
            ResponseSchema(
                name="non_branded_keyword_potential",
                description="Potential for non-branded keyword success",
                type="string"
            ),
            ResponseSchema(
                name="exact_match_search_results",
                description="Number of exact match search results",
                type="integer"
            ),
            ResponseSchema(
                name="competitor_domain_strength",
                description="Analysis of competitor domain strength",
                type="string"
            ),
            ResponseSchema(
                name="name_length_searchability",
                description="Impact of name length on search",
                type="string"
            ),
            ResponseSchema(
                name="unusual_spelling_impact",
                description="Impact of unusual spelling on search",
                type="boolean"
            ),
            ResponseSchema(
                name="negative_keyword_associations",
                description="Potential negative keyword associations",
                type="string"
            ),
            ResponseSchema(
                name="negative_search_results",
                description="Whether negative search results exist for the name",
                type="boolean"
            ),
            ResponseSchema(
                name="content_marketing_opportunities",
                description="Content marketing opportunities related to the name",
                type="string"
            ),
            ResponseSchema(
                name="social_media_availability",
                description="Social media handle availability",
                type="object"
            ),
            ResponseSchema(
                name="social_media_discoverability",
                description="How easily findable the brand will be on social platforms",
                type="string"
            ),
            ResponseSchema(
                name="seo_recommendations",
                description="SEO strategy recommendations",
                type="array"
            ),
            ResponseSchema(
                name="seo_viability_score",
                description="Overall SEO viability score (1-10)",
                type="float"
            )
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.output_schemas
        )
        
        # Create Google Search tool
        search_tool = {
            "type": "google_search",
            "google_search": {}
        }
        
        # Initialize Gemini model with tracing and Google Search tool
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None,
            tools=[search_tool]
        )

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the SEO potential and online discoverability of a shortlisted brand name.
        
        This method focuses specifically on online discovery aspects for names that
        have already passed initial screening:
        - Search engine findability and keyword potential
        - Social media availability (verified by API) and discoverability
        - Search intent alignment and content opportunities
        - Broader online visibility beyond domain aspects
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The shortlisted brand name to analyze
            brand_context: Additional brand context information
            
        Returns:
            Dictionary containing the SEO analysis results
            
        Raises:
            ValueError: If the analysis fails
        """
        try:
            with tracing_v2_enabled():
                # Format prompt with parser instructions
                system_message = SystemMessage(
                    content=self.system_prompt.format(),
                    additional_kwargs={
                        "metadata": {
                            "agent_type": "seo_analyzer",
                            "methodology": "Digital Brand Optimization Framework"
                        }
                    }
                )
                
                # The domain_analysis variable is expected by the template but may not be available
                domain_analysis = brand_context.get("domain_analysis", {})
                
                # Ensure brand_context is properly formatted for the prompt
                formatted_brand_context = {
                    "industry_focus": brand_context.get("industry_focus", ""),
                    "target_audience": brand_context.get("target_audience", ""),
                    "brand_personality": brand_context.get("brand_personality", ""),
                    "brand_values": brand_context.get("brand_values", "")
                }
                
                human_message = HumanMessage(
                    content=self.human_prompt.format(
                        format_instructions=self.output_parser.get_format_instructions(),
                        brand_name=brand_name,
                        brand_context=json.dumps(formatted_brand_context, ensure_ascii=False),
                        domain_analysis=json.dumps(domain_analysis, ensure_ascii=False)
                    )
                )
                
                formatted_messages = [system_message, human_message]
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_messages)
                
                try:
                    # Parse and validate response
                    analysis = self.output_parser.parse(response.content)
                except Exception as parse_error:
                    logger.error(
                        "Failed to parse LLM response",
                        extra={
                            "error": str(parse_error),
                            "response_content": response.content
                        }
                    )
                    # Provide default values for required fields
                    analysis = {
                        "keyword_alignment": "Moderate",
                        "search_volume": 0.0,
                        "keyword_competition": "Moderate",
                        "branded_keyword_potential": "Moderate",
                        "non_branded_keyword_potential": "Moderate",
                        "exact_match_search_results": 0,
                        "competitor_domain_strength": "Moderate",
                        "name_length_searchability": "Medium",
                        "unusual_spelling_impact": False,
                        "negative_keyword_associations": "None identified",
                        "negative_search_results": False,
                        "content_marketing_opportunities": "Standard opportunities available",
                        "social_media_availability": {},
                        "social_media_discoverability": "Moderate",
                        "seo_recommendations": ["Implement standard SEO best practices"],
                        "seo_viability_score": 5.0
                    }
                
                # Ensure social_media_availability is a dict
                if not isinstance(analysis.get("social_media_availability"), dict):
                    analysis["social_media_availability"] = {}
                
                # Ensure seo_recommendations is a list
                if not isinstance(analysis.get("seo_recommendations"), list):
                    try:
                        recommendations = json.loads(analysis.get("seo_recommendations", "[]"))
                        if isinstance(recommendations, list):
                            analysis["seo_recommendations"] = recommendations
                        else:
                            analysis["seo_recommendations"] = [str(recommendations)]
                    except json.JSONDecodeError:
                        analysis["seo_recommendations"] = [str(analysis.get("seo_recommendations", ""))]
                
                # Check social media availability using Username Hunter API
                social_media_available, platform_details, discoverability_rating = await self.check_social_media_availability(brand_name)
                
                # Update analysis with social media availability data
                analysis["social_media_availability"] = platform_details
                analysis["social_media_discoverability"] = discoverability_rating
                
                # Store results
                await self._store_analysis(run_id, brand_name, analysis)
                
                return analysis
                
        except Exception as e:
            logger.error(
                "SEO analysis failed",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise ValueError(f"Failed to analyze brand name: {str(e)}")

    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """
        Store the SEO analysis in Supabase.
        
        This method specifically focuses on online discovery aspects:
        - Search engine optimization metrics and potential
        - Actual social media availability (verified by API)
        - Discoverability factors across digital platforms
        - Search intent and keyword behavior analysis
        """
        try:
            # Validate that analysis stays within proper boundaries
            
            # Ensure branded_keyword_potential focuses on search behavior, not domain aspects
            branded_keyword = analysis.get("branded_keyword_potential", "Moderate")
            if any(term in branded_keyword.lower() for term in ["domain", "url", "tld"]):
                logger.warning("branded_keyword_potential contains domain aspects - focusing on search aspects only")
                branded_keyword = "Moderate"
                
            # Ensure non_branded_keyword_potential focuses on search behavior
            non_branded_keyword = analysis.get("non_branded_keyword_potential", "Moderate")
            if any(term in non_branded_keyword.lower() for term in ["domain", "url", "tld"]):
                logger.warning("non_branded_keyword_potential contains domain aspects - focusing on search aspects only")
                non_branded_keyword = "Moderate"
                
            # Ensure social_media_discoverability focuses on findability, not just availability
            social_discoverability = analysis.get("social_media_discoverability", "Moderate")
            if social_discoverability in ["Available", "Not Available", "Partially Available"]:
                logger.warning("social_media_discoverability contains availability info - changing to discoverability rating")
                social_discoverability = "Moderate"
            
            # Convert data types where needed
            try:
                search_volume = float(analysis.get("search_volume", 0))
            except (ValueError, TypeError):
                search_volume = 0.0
                logger.warning(f"Invalid search_volume value: {analysis.get('search_volume')}, using default")
                
            try:
                seo_viability_score = float(analysis.get("seo_viability_score", 5.0))
                # Ensure score is within 1-10 range
                seo_viability_score = max(1.0, min(10.0, seo_viability_score))
            except (ValueError, TypeError):
                seo_viability_score = 5.0
                logger.warning(f"Invalid seo_viability_score value: {analysis.get('seo_viability_score')}, using default")
                
            # Handle boolean fields
            try:
                unusual_spelling_impact = bool(analysis.get("unusual_spelling_impact", False))
                if isinstance(unusual_spelling_impact, str):
                    unusual_spelling_impact = unusual_spelling_impact.lower() in ["true", "yes", "1"]
            except (ValueError, TypeError):
                unusual_spelling_impact = False
                
            try:
                negative_search_results = bool(analysis.get("negative_search_results", False))
                if isinstance(negative_search_results, str):
                    negative_search_results = negative_search_results.lower() in ["true", "yes", "1"]
            except (ValueError, TypeError):
                negative_search_results = False
                
            # Handle arrays
            seo_recommendations = analysis.get("seo_recommendations", [])
            if isinstance(seo_recommendations, str):
                try:
                    # Try to convert string to array if it's in JSON format
                    seo_recommendations = json.loads(seo_recommendations)
                except json.JSONDecodeError:
                    # If not JSON, create a single-item array
                    seo_recommendations = [seo_recommendations]
            
            # Convert social media availability dictionary to boolean
            social_media_data = analysis.get("social_media_availability", {})
            if isinstance(social_media_data, dict):
                # Consider it available if any platform is marked as available
                social_media_available = any(
                    platform.get("available", False) if isinstance(platform, dict) else 
                    (str(platform).lower() == "available")
                    for platform in social_media_data.values()
                )
            else:
                social_media_available = False
            
            # Prepare data for storage based on the database schema
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "timestamp": datetime.now().isoformat(),
                
                # Core SEO metrics - focused on search behavior, not domain aspects
                "branded_keyword_potential": branded_keyword,
                "non_branded_keyword_potential": non_branded_keyword,
                "keyword_competition": analysis.get("keyword_competition", "Moderate"),
                "competitor_domain_strength": analysis.get("competitor_domain_strength", "Moderate"),
                
                # Search characteristics
                "keyword_alignment": analysis.get("keyword_alignment", ""),
                "name_length_searchability": analysis.get("name_length_searchability", "Medium"),
                "search_volume": search_volume,
                "exact_match_search_results": analysis.get("exact_match_search_results", 0),
                
                # Potential issues
                "negative_keyword_associations": analysis.get("negative_keyword_associations", ""),
                "negative_search_results": negative_search_results,
                "unusual_spelling_impact": unusual_spelling_impact,
                
                # Social media aspects (actual availability verified by API)
                "social_media_availability": social_media_available,
                "social_media_discoverability": social_discoverability,
                
                # Content strategy
                "content_marketing_opportunities": analysis.get("content_marketing_opportunities", ""),
                "seo_recommendations": seo_recommendations,
                
                # Overall score
                "seo_viability_score": seo_viability_score
            }
            
            # Log what we're storing
            logger.info(f"Storing SEO analysis for '{brand_name}' with fields: {list(data.keys())}")
            
            # Use the async execute_with_retry method instead of direct table operations
            await self.supabase.execute_with_retry("insert", "seo_online_discoverability", data)
            logger.info(f"Stored SEO analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing SEO analysis in Supabase",
                extra={
                    "error": str(e),
                    "run_id": run_id,
                    "brand_name": brand_name
                }
            )
            raise

    async def check_social_media_availability(self, brand_name: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Check if a brand name is available on key social media platforms using Username Hunter API.
        
        Args:
            brand_name (str): The brand name to check
            
        Returns:
            Tuple containing:
            - Boolean indicating overall availability
            - Dictionary with platform details
            - Discoverability rating (Easy, Moderate, Difficult)
        """
        try:
            # Strip spaces and special characters for username check
            username = brand_name.lower().replace(" ", "").replace("-", "")
            
            # Setup API request
            url = "https://username-hunter-api.p.rapidapi.com/hunt/username"
            
            querystring = {
                "username": username,
                "platforms": "facebook,threads,tiktok,instagram"
            }
            
            headers = {
                "x-rapidapi-key": settings.rapid_api_key,
                "x-rapidapi-host": settings.username_hunter_host
            }
            
            # Make async API request
            logger.info(f"Checking social media availability for: {username}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=querystring) as response:
                    results = await response.json()
            
            # Process results
            available_platforms = []
            platform_details = {}
            
            if "availability" in results:
                availability_data = results["availability"]
                for platform, data in availability_data.items():
                    is_available = data.get("is_available", False)
                    platform_details[platform] = {
                        "available": is_available,
                        "url": data.get("url", "")
                    }
                    
                    if is_available:
                        available_platforms.append(platform)
            
            # Determine overall availability and discoverability rating
            total_platforms = len(platform_details)
            available_count = len(available_platforms)
            
            # Overall availability is true if at least half the platforms are available
            overall_available = available_count >= (total_platforms / 2)
            
            # Determine discoverability rating
            if available_count >= total_platforms * 0.75:
                discoverability = "Easy"
            elif available_count >= total_platforms * 0.5:
                discoverability = "Moderate"
            else:
                discoverability = "Difficult"
                
            logger.info(f"Social media availability results for '{username}': Available platforms: {available_platforms}, Discoverability: {discoverability}")
            
            return overall_available, platform_details, discoverability
            
        except Exception as e:
            logger.error(f"Error checking social media availability for '{brand_name}': {str(e)}")
            # Return default values in case of error
            return False, {}, "Difficult" 