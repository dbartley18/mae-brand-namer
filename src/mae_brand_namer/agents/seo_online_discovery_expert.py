"""Expert in analyzing SEO potential and online discoverability of brand names."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.messages import HumanMessage, SystemMessage

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
                name="social_media_availability",
                description="Social media handle availability",
                type="object"
            ),
            ResponseSchema(
                name="seo_recommendations",
                description="SEO strategy recommendations",
                type="array"
            ),
            ResponseSchema(
                name="digital_presence_score",
                description="Overall digital presence score (1-10)",
                type="number"
            )
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.output_schemas
        )
        
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
                    "agent_type": "seo_analyzer",
                    "methodology": "Digital Brand Optimization Framework"
                }
            }
        )
        human_template = (
            "Analyze the SEO potential and online discoverability of the "
            "following brand name:\n"
            "Brand Name: {brand_name}\n"
            "Brand Context: {brand_context}\n"
            "\nFormat your analysis according to this schema:\n"
            "{format_instructions}"
        )
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the SEO potential and online discoverability of a brand name.
        
        Evaluates the brand name's potential for search engine optimization,
        digital marketing effectiveness, and online visibility.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Additional brand context information
            
        Returns:
            Dictionary containing the SEO analysis results
            
        Raises:
            ValueError: If the analysis fails
        """
        try:
            with tracing_enabled(
                tags={
                    "agent": "SEOOnlineDiscoveryExpert",
                    "run_id": run_id
                }
            ):
                # Format prompt with parser instructions
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=brand_context
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse and validate response
                analysis = self.output_parser.parse(response.content)
                
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
        """Store SEO analysis results in Supabase.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The analyzed brand name
            analysis: Analysis results to store
            
        Raises:
            Exception: If storage fails
        """
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
            
            await self.supabase.table("seo_analysis").insert(data).execute()
            logger.info(f"Stored SEO analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing SEO analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise 