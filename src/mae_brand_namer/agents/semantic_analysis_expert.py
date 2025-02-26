"""Semantic Analysis Expert for analyzing brand name meanings and associations."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError
from storage3.exceptions import StorageException
from gotrue.errors import AuthError as AuthException

from ..config.settings import settings
from ..utils.logging import get_logger
from ..models.state import SemanticAnalysisResult
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class SemanticAnalysisExpert:
    """Expert in analyzing semantic meaning and brand associations."""
    
    def __init__(self, supabase: SupabaseManager = None):
        """Initialize the SemanticAnalysisExpert with dependencies."""
        self.supabase = supabase or SupabaseManager()
        
        self.role = "Semantic Analysis Expert"
        self.goal = "Analyze brand names for semantic meaning, associations, and brand fit"
        self.backstory = """You are a distinguished expert in semantic analysis with deep expertise in 
        linguistics, brand psychology, and consumer behavior. Your analyses help companies understand the 
        full semantic impact of their brand names."""
        
        # Initialize retry configuration
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            self.tracer = LangChainTracer(
                project_name=os.getenv("LANGCHAIN_PROJECT", "mae-brand-namer")
            )
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            convert_system_message_to_human=True,
            callbacks=[self.tracer] if self.tracer else None
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(
                name="denotative_meaning",
                description="Direct, literal meaning of the name",
                type="string"
            ),
            ResponseSchema(
                name="connotative_associations",
                description="Indirect associations and implications",
                type="list[string]"
            ),
            ResponseSchema(
                name="brand_personality_alignment",
                description="How well the name aligns with desired brand personality (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="emotional_resonance",
                description="Emotional impact and memorability score (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="conceptual_clarity",
                description="How clearly the name communicates intended concepts (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="semantic_distinctiveness",
                description="How semantically unique the name is in the market (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="analysis_notes",
                description="Detailed semantic analysis notes",
                type="string"
            )
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Create the prompt template with metadata for LangGraph Studio
        system_message = SystemMessage(
            content=f"""You are a Semantic Analysis Expert with the following profile:
            Role: {self.role}
            Goal: {self.goal}
            Backstory: {self.backstory}
            
            Analyze the provided brand name based on its semantic meaning and brand context.
            Consider:
            1. Denotative & Connotative Meaning
            2. Brand Personality Alignment
            3. Emotional Impact & Memorability
            4. Conceptual Clarity & Communication
            5. Semantic Distinctiveness
            6. Market Context & Relevance
            
            Format your response according to the following schema:
            {{format_instructions}}
            """,
            additional_kwargs={
                "metadata": {
                    "agent_type": "semantic_analyzer",
                    "methodology": "Alina Wheeler's Designing Brand Identity"
                }
            }
        )
        human_template = """Analyze the following brand name in its context:
        Brand Name: {brand_name}
        Brand Context: {brand_context}
        """
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
        """
        Analyze a brand name's semantic meaning and store results.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_name (str): The brand name to analyze
            brand_context (Dict[str, Any]): Brand context information
            
        Returns:
            Dict[str, Any]: Semantic analysis results
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            # Setup event loop if not available
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            with tracing_enabled(tags={"agent": "SemanticAnalysisExpert", "run_id": run_id}):
                # Format prompt with parser instructions
                formatted_prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=json.dumps(brand_context, indent=2)
                )
                
                # Get response from LLM
                response = await self.llm.ainvoke(formatted_prompt)
                
                # Parse structured response
                analysis_result = self.output_parser.parse(response.content)
                
                # Store results in Supabase
                await self._store_analysis(run_id, brand_name, analysis_result)
                
                # Create a result dictionary with brand_name and run_id
                result = {
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "task_name": "semantic_analysis",
                }
                
                # Add all analysis results, avoiding duplicate keys
                for key, value in analysis_result.items():
                    if key not in result:  # Avoid duplicating keys
                        result[key] = value
                
                return result
        
        except Exception as e:
            logger.error(
                "Error in semantic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise ValueError(f"Semantic analysis failed: {str(e)}")
    
    async def _store_analysis(
        self,
        run_id: str,
        brand_name: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Store semantic analysis results in Supabase."""
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
                "task_name": "semantic_analysis",
                "timestamp": datetime.now().isoformat(),
                **analysis
            }
            
            await self.supabase.table("semantic_analysis").insert(data).execute()
            logger.info(f"Stored semantic analysis for brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing semantic analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "error": str(e)
                }
            )
            raise 