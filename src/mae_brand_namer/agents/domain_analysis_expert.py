"""Domain Analysis Expert for evaluating domain name availability and strategy."""

import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import asyncio
import requests
import aiohttp

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pathlib import Path
from langchain_core.tracers.context import tracing_v2_enabled

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)

class DomainAnalysisExpert:
    """Expert in analyzing domain name availability and digital presence strategy."""
    
    def __init__(self, dependencies: Dependencies):
        """Initialize the DomainAnalysisExpert with dependencies."""
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        # Agent identity
        self.role = "Domain Strategy & Digital Presence Expert"
        self.goal = """Evaluate brand names for domain availability, digital presence strategy, 
        and overall web identity effectiveness."""
        
        # Load prompts from YAML files
        prompt_dir = Path(__file__).parent / "prompts" / "domain_analysis"
        self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
        self.human_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="domain_exact_match", description="Availability of exact match domain", type="boolean"),
            ResponseSchema(name="alternative_tlds", description="List of alternative top-level domains", type="array"),
            ResponseSchema(name="domain_length_readability", description="Domain length assessment (Short, Medium, Long)", type="string"),
            ResponseSchema(name="domain_history_reputation", description="Domain history assessment (Clean, Neutral, Problematic)", type="string"),
            ResponseSchema(name="acquisition_cost", description="Cost estimate (Low, Moderate, High, Not for Sale)", type="string"),
            ResponseSchema(name="brand_name_clarity_in_url", description="Clarity of brand name in URL (High, Moderate, Low)", type="string"),
            ResponseSchema(name="seo_keyword_relevance", description="SEO keyword relevance (High, Moderate, Low)", type="string"),
            ResponseSchema(name="hyphens_numbers_present", description="Presence of hyphens or numbers in domain", type="boolean"),
            ResponseSchema(name="misspellings_variations_available", description="Availability of common misspellings or variations", type="boolean"),
            ResponseSchema(name="idn_support_needed", description="Whether international domain name support is required", type="boolean"),
            ResponseSchema(name="social_media_availability", description="Suggested handle formats for social media", type="array"),
            ResponseSchema(name="scalability_future_proofing", description="How well the domain accommodates future expansion", type="string"),
            ResponseSchema(name="security_privacy_recommendations", description="Security and privacy recommendations", type="string"),
            ResponseSchema(name="notes", description="Additional observations or considerations", type="string"),
            ResponseSchema(name="rank", description="Overall domain quality rating (1.0-10.0)", type="number")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.2,  # Balanced temperature for analysis
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )

    async def analyze_domain(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze domain name availability and strategy for a shortlisted brand name.
        
        This method focuses specifically on domain-related aspects for names that
        have already passed initial screening:
        - Domain availability and alternatives
        - Domain technical characteristics
        - Domain-based social media handle formats
        - Keywords as they appear in the domain name itself
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_name (str): The shortlisted brand name to analyze
            brand_context (Dict[str, Any], optional): Additional brand context
            
        Returns:
            Dict[str, Any]: Domain analysis results
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            with tracing_v2_enabled():
                # First, check real domain availability using Domainr API
                domain_suggestions = await self.search_domains(brand_name)
                
                # Extract domain information
                available_domains = []
                primary_domain_available = False
                domain_tlds = []
                
                # Process domain suggestions
                primary_domain = f"{brand_name.lower().replace(' ', '')}.com"
                
                for domain in domain_suggestions:
                    domain_name = domain.get("domain", "")
                    if domain_name:
                        # Check domain status
                        status = await self.check_domain_status(domain_name)
                        availability = status.get("status", "")
                        
                        # Check if domain is available
                        if "inactive" in availability or "available" in availability:
                            available_domains.append(domain_name)
                            
                            # Extract TLD
                            parts = domain_name.split(".")
                            if len(parts) > 1:
                                tld = parts[-1]
                                if tld not in domain_tlds:
                                    domain_tlds.append(tld)
                                    
                        # Check if primary domain is available
                        if domain_name.lower() == primary_domain:
                            primary_domain_available = "inactive" in availability or "available" in availability
                
                logger.info(f"Domain API check results: Primary domain available: {primary_domain_available}, Available domains: {len(available_domains)}, TLDs: {domain_tlds}")
                
                # Format prompt with parser instructions and include real domain data
                prompt_context = {
                    "brand_context": json.dumps(brand_context) if brand_context else "{}",
                    "domain_data": {
                        "primary_domain_available": primary_domain_available,
                        "available_domains": available_domains,
                        "domain_tlds": domain_tlds
                    }
                }
                
                # Create formatted messages using both system prompt and human prompt
                system_message = SystemMessage(
                    content=self.system_prompt.format(),
                    additional_kwargs={
                        "metadata": {
                            "agent_type": "domain_analyzer",
                            "methodology": "Digital Brand Identity Framework"
                        }
                    }
                )
                
                human_message = HumanMessage(
                    content=self.human_prompt.format(
                        format_instructions=self.output_parser.get_format_instructions(),
                        brand_name=brand_name,
                        brand_context=json.dumps(prompt_context)
                    )
                )
                
                formatted_messages = [system_message, human_message]
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_messages)
                
                # Parse the response according to the defined schema
                content = response.content if hasattr(response, 'content') else str(response)
                analysis = self.output_parser.parse(content)
                
                # Add real domain data from API to analysis
                analysis["domain_exact_match"] = primary_domain_available
                
                # Process alternative TLDs
                alternative_tlds = []
                if available_domains:
                    # Extract unique TLDs from alternative domains
                    for domain in available_domains:
                        parts = domain.split(".")
                        if len(parts) > 1:
                            tld = parts[-1]
                            if tld not in alternative_tlds:
                                alternative_tlds.append(tld)
                
                analysis["alternative_tlds"] = alternative_tlds
                
                # Generate domain-based social media handles
                sanitized_name = brand_name.lower().replace(" ", "")
                social_handles = [
                    f"{sanitized_name}",
                    f"{sanitized_name}_official",
                    f"the{sanitized_name}",
                    f"{sanitized_name}brand"
                ]
                analysis["social_media_availability"] = social_handles
                
                # Enforce field boundaries to ensure experts stay in their lanes
                
                # Ensure SEO keyword relevance is about the domain name only
                if "seo_keyword_relevance" in analysis:
                    # Check if the analysis goes beyond domain-specific keyword presence
                    seo_text = analysis["seo_keyword_relevance"]
                    if len(seo_text) > 100 or any(term in seo_text.lower() for term in ["serp", "google", "ranking", "search engine"]):
                        # Simplify to focus only on domain keyword presence
                        keywords = [word for word in brand_name.lower().split() if len(word) > 3]
                        if keywords:
                            analysis["seo_keyword_relevance"] = f"Domain contains brand keywords: {', '.join(keywords)}"
                        else:
                            analysis["seo_keyword_relevance"] = "Domain directly represents the brand name"
                
                # Ensure scalability analysis is about domain infrastructure
                if "scalability_future_proofing" in analysis:
                    scalability = analysis["scalability_future_proofing"]
                    if any(term in scalability.lower() for term in ["market", "audience", "revenue", "business model"]):
                        analysis["scalability_future_proofing"] = "Domain structure allows for future expansion via subdomains if needed"
                
                # Store results in Supabase
                await self._store_analysis(run_id, brand_name, analysis)
                
                return analysis
                
        except Exception as e:
            logger.error(
                "Error analyzing domain strategy",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise ValueError(f"Domain analysis failed: {str(e)}")
    
    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """
        Store domain analysis results in Supabase.
        
        This method specifically focuses on domain-related aspects:
        - Domain availability and registration
        - Domain characteristics and technical aspects
        - Domain-based handle suggestions (NOT actual social media availability)
        - Domain-specific SEO keyword presence (NOT broader SEO potential)
        """
        try:
            # Validate that analysis stays within proper boundaries
            # For SEO keyword relevance, restrict to how keywords appear in domain name only
            seo_keyword_relevance = analysis.get("seo_keyword_relevance", "Moderate")
            
            # Ensure SEO keyword relevance is one of the allowed values
            allowed_seo_values = ["High", "Moderate", "Low"]
            if seo_keyword_relevance not in allowed_seo_values:
                logger.warning(f"Invalid seo_keyword_relevance value: {seo_keyword_relevance}. Setting to 'Moderate'")
                seo_keyword_relevance = "Moderate"
                
            # For domain_length_readability, ensure it's one of the allowed values
            domain_length = analysis.get("domain_length_readability", "Medium")
            allowed_length_values = ["Short", "Medium", "Long"]
            if domain_length not in allowed_length_values:
                logger.warning(f"Invalid domain_length_readability value: {domain_length}. Setting to 'Medium'")
                domain_length = "Medium"
                
            # For domain_history_reputation, ensure it's one of the allowed values
            domain_history = analysis.get("domain_history_reputation", "Neutral")
            allowed_history_values = ["Clean", "Neutral", "Problematic"]
            if domain_history not in allowed_history_values:
                logger.warning(f"Invalid domain_history_reputation value: {domain_history}. Setting to 'Neutral'")
                domain_history = "Neutral"
                
            # For brand_name_clarity_in_url, ensure it's one of the allowed values
            brand_clarity = analysis.get("brand_name_clarity_in_url", "Moderate")
            allowed_clarity_values = ["High", "Moderate", "Low"]
            if brand_clarity not in allowed_clarity_values:
                logger.warning(f"Invalid brand_name_clarity_in_url value: {brand_clarity}. Setting to 'Moderate'")
                brand_clarity = "Moderate"
                
            # For acquisition_cost, ensure it's one of the allowed values
            acquisition_cost = analysis.get("acquisition_cost", "Moderate")
            allowed_cost_values = ["Low", "Moderate", "High", "Not for Sale"]
            if acquisition_cost not in allowed_cost_values:
                logger.warning(f"Invalid acquisition_cost value: {acquisition_cost}. Setting to 'Moderate'")
                acquisition_cost = "Moderate"
            
            # For scalability_future_proofing, ensure focus on domain expansion, not market scalability
            scalability = analysis.get("scalability_future_proofing", "")
            if scalability and any(term in scalability.lower() for term in ["market", "audience", "revenue", "profit"]):
                logger.warning("scalability_future_proofing contains market aspects - restricting to domain aspects only")
                scalability = "Domain provides room for expansion via subdomains if needed."
            
            # Handle suggestions should be domain-based formats, not platform availability analysis
            social_handles = analysis.get("social_media_availability", [])
            if not social_handles:
                # Generate domain-based social media handles if not provided
                sanitized_name = brand_name.lower().replace(" ", "")
                social_handles = [
                    f"{sanitized_name}",
                    f"{sanitized_name}_official",
                    f"the{sanitized_name}",
                    f"{sanitized_name}brand"
                ]
            
            # Handle the rank value
            rank = analysis.get("rank")
            try:
                if rank is not None:
                    rank = float(rank)
                    # Ensure the rank is within the valid range
                    if rank < 1.0 or rank > 10.0:
                        logger.warning(f"Invalid rank value {rank}. Setting to 5.0")
                        rank = 5.0
                else:
                    # Use digital_presence_score as fallback for rank if available
                    digital_presence_score = analysis.get("digital_presence_score")
                    if digital_presence_score is not None:
                        rank = float(digital_presence_score)
                    else:
                        rank = 5.0
            except (ValueError, TypeError):
                logger.warning(f"Could not convert rank value to float: {rank}. Setting to 5.0")
                rank = 5.0
            
            # Prepare data for storage
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                
                # Domain registration aspects
                "domain_exact_match": analysis.get("domain_exact_match", False),
                "acquisition_cost": acquisition_cost,
                "alternative_tlds": analysis.get("alternative_tlds", []),
                
                # Domain characteristics
                "domain_length_readability": domain_length,
                "brand_name_clarity_in_url": brand_clarity,
                "hyphens_numbers_present": analysis.get("hyphens_numbers_present", False),
                
                # Technical aspects
                "domain_history_reputation": domain_history,
                "misspellings_variations_available": analysis.get("misspellings_variations_available", False),
                "idn_support_needed": analysis.get("idn_support_needed", False),
                
                # Domain-based handle suggestions
                "social_media_availability": social_handles,
                
                # Future considerations - focused on domain aspects only
                "scalability_future_proofing": analysis.get("scalability_future_proofing", scalability),
                "security_privacy_recommendations": analysis.get("security_privacy_recommendations", ""),
                
                # Domain SEO characteristics only
                "seo_keyword_relevance": seo_keyword_relevance,
                
                # Additional information
                "notes": analysis.get("notes", ""),
                "rank": rank
            }
            
            # Log what we're storing
            logger.info(f"Storing domain analysis for '{brand_name}' with fields: {list(data.keys())}")
            
            # Use the async execute_with_retry method instead of direct table operations
            await self.supabase.execute_with_retry("insert", "domain_analysis", data)
            logger.info(f"Stored domain analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing domain analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise
    
    async def search_domains(self, brand_name: str) -> List[Dict[str, Any]]:
        """
        Search for available domains for a brand name using Domainr API.
        
        Args:
            brand_name: The brand name to search domains for
            
        Returns:
            List of domain suggestions with availability information
        """
        try:
            # Setup API request
            url = "https://domainr.p.rapidapi.com/v2/search"
            
            # Strip spaces and sanitize the brand name for domain search
            sanitized_name = brand_name.lower().replace(" ", "")
            
            # Common TLDs to search
            defaults = "com,net,org,io,co"
            
            # Set up query parameters
            querystring = {
                "defaults": defaults,
                "query": sanitized_name,
            }
            
            headers = {
                "x-rapidapi-key": settings.rapid_api_key,
                "x-rapidapi-host": settings.domainr_host
            }
            
            # Make async API request
            logger.info(f"Searching domains for brand name: {brand_name}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=querystring) as response:
                    # Properly handle the response
                    if response.status == 200:
                        results = await response.json()
                    else:
                        logger.error(f"API request failed with status code {response.status}")
                        return []
            
            # Extract domain results
            domains = []
            if "results" in results:
                domains = results["results"]
                logger.info(f"Found {len(domains)} domain suggestions for '{brand_name}'")
            else:
                logger.warning(f"No domain results found for '{brand_name}'")
            
            return domains
            
        except Exception as e:
            logger.error(f"Error searching domains for '{brand_name}': {str(e)}")
            return []
    
    async def check_domain_status(self, domain: str) -> Dict[str, Any]:
        """
        Check availability status of a specific domain using Domainr API.
        
        Args:
            domain: The domain to check (e.g., "example.com")
            
        Returns:
            Domain status information
        """
        try:
            # Setup API request
            url = "https://domainr.p.rapidapi.com/v2/status"
            
            # Set up query parameters
            querystring = {"domain": domain}
            
            headers = {
                "x-rapidapi-key": settings.rapid_api_key,
                "x-rapidapi-host": settings.domainr_host
            }
            
            # Make async API request
            logger.info(f"Checking status for domain: {domain}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=querystring) as response:
                    # Properly handle the response
                    if response.status == 200:
                        result = await response.json()
                    else:
                        logger.error(f"API request failed with status code {response.status}")
                        return {"domain": domain, "status": "error", "error": f"API error: {response.status}"}
            
            # Extract status information
            status = {}
            if "status" in result:
                status = result["status"][0] if result["status"] else {}
                logger.info(f"Domain '{domain}' status: {status.get('status', 'unknown')}")
            else:
                logger.warning(f"No status information found for domain '{domain}'")
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking status for domain '{domain}': {str(e)}")
            return {"domain": domain, "status": "error", "error": str(e)} 