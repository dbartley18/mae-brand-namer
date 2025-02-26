"""Domain Analysis Expert for evaluating domain name availability and strategy."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pathlib import Path

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
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="exact_match_domain", description="Availability of exact match domain", type="boolean"),
            ResponseSchema(name="alternative_domains", description="List of viable domain alternatives", type="array"),
            ResponseSchema(name="domain_length", description="Analysis of domain length impact", type="string"),
            ResponseSchema(name="domain_memorability", description="Assessment of domain memorability", type="string"),
            ResponseSchema(name="domain_brandability", description="Potential for domain branding", type="string"),
            ResponseSchema(name="domain_pronunciation", description="Ease of verbal communication", type="string"),
            ResponseSchema(name="domain_spelling", description="Risk of misspelling", type="string"),
            ResponseSchema(name="tld_strategy", description="Top-level domain recommendations", type="string"),
            ResponseSchema(name="domain_availability", description="Overall domain availability assessment", type="string"),
            ResponseSchema(name="domain_cost_estimate", description="Estimated cost range for domains", type="string"),
            ResponseSchema(name="domain_history", description="Analysis of domain history if previously used", type="string"),
            ResponseSchema(name="digital_presence_score", description="Overall digital presence score (1-10)", type="float"),
            ResponseSchema(name="recommendations", description="Strategic domain recommendations", type="array")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None
        )
        
        # Create the prompt template with metadata
        system_message = SystemMessage(
            content=self.system_prompt.format(),
            additional_kwargs={
                "metadata": {
                    "agent_type": "domain_analyzer",
                    "methodology": "Digital Brand Identity Framework"
                }
            }
        )
        human_template = """Analyze the domain strategy for the following brand name:
        Brand Name: {brand_name}
        Brand Context: {brand_context}
        
        Format your analysis according to this schema:
        {format_instructions}
        """
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

    async def analyze_domain(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze domain name availability and strategy.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_name (str): The brand name to analyze
            brand_context (Dict[str, Any], optional): Additional brand context
            
        Returns:
            Dict[str, Any]: Domain analysis results
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            with tracing_enabled(tags={"agent": "DomainAnalysisExpert", "run_id": run_id}):
                # Format prompt with parser instructions
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=json.dumps(brand_context) if brand_context else "{}"
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse structured response
                analysis = self.output_parser.parse(response.content)
                
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
        """Store domain analysis results in Supabase."""
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "timestamp": datetime.now().isoformat(),
                **analysis
            }
            
            await self.supabase.table("domain_analysis").insert(data).execute()
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