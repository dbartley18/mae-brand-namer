"""Brand Name Evaluator for final assessment and shortlisting of brand names."""

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

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class BrandNameEvaluator:
    """Expert in evaluating and shortlisting brand names based on comprehensive analysis."""
    
    def __init__(self, supabase: SupabaseManager = None):
        """Initialize the BrandNameEvaluator with necessary configurations."""
        # Agent identity from agents.yaml
        self.role = "Brand Name Evaluation Expert"
        self.goal = """Assess and score generated brand names based on linguistic qualities, cultural considerations, 
        strategic fit, competitive differentiation, and overall alignment with the Brand Identity Brief."""
        self.backstory = """You are the final decision-maker in the brand naming process, applying an expert-level, 
        holistic evaluation process that ensures only the most strategically sound names advance. Your expertise spans 
        brand strategy, linguistics, consumer psychology, and competitive intelligence, making you the last line of 
        defense before a name is finalized.

        With a deep understanding of Alina Wheeler's branding principles and real-world naming best practices, you 
        critically assess how well each name aligns with the Brand Identity Brief, stands out from competitors, and 
        resonates with target audiences. You don't just evaluate names—you predict their long-term impact, scalability, 
        visibility, and memorability in the market.

        Your work ensures that the chosen name is not just creative, but commercially viable, legally defensible, 
        and globally adaptable—the foundation of a successful, future-proof brand."""
        
        # Initialize retry configuration
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        
        # Initialize Supabase client
        self.supabase = supabase or SupabaseManager()
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if settings.langchain_tracing_v2:
            self.tracer = LangChainTracer(project_name=settings.langsmith_project)
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=0.7,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.tracer] if self.tracer else None
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="strategic_alignment_score", description="How well the name aligns with the Brand Identity Brief (1-10)", type="number"),
            ResponseSchema(name="distinctiveness_score", description="How unique the name is compared to competitors (1-10)", type="number"),
            ResponseSchema(name="competitive_advantage", description="Analysis of competitive differentiation"),
            ResponseSchema(name="brand_fit_score", description="How well the name aligns with brand strategy (1-10)", type="number"),
            ResponseSchema(name="positioning_strength", description="Effectiveness in market positioning"),
            ResponseSchema(name="memorability_score", description="How easy the name is to recall (1-10)", type="number"),
            ResponseSchema(name="pronounceability_score", description="How easily the name is spoken (1-10)", type="number"),
            ResponseSchema(name="meaningfulness_score", description="Clarity and positive connotation (1-10)", type="number"),
            ResponseSchema(name="phonetic_harmony", description="Analysis of sound patterns and flow"),
            ResponseSchema(name="visual_branding_potential", description="Potential for visual identity development"),
            ResponseSchema(name="storytelling_potential", description="Capacity for brand narrative development"),
            ResponseSchema(name="domain_viability_score", description="Initial domain name availability assessment (1-10)", type="number"),
            ResponseSchema(name="overall_score", description="Total weighted evaluation score (1-10)", type="number"),
            ResponseSchema(name="shortlist_status", description="Whether selected for final round (Yes/No)"),
            ResponseSchema(name="evaluation_comments", description="Detailed rationale for evaluation"),
            ResponseSchema(name="rank", description="Final ranking among all candidates (1-N)", type="number")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Create the prompt template with metadata for LangGraph Studio
        system_message = SystemMessage(
            content=f"""You are a Brand Name Evaluator with the following profile:
            Role: {self.role}
            Goal: {self.goal}
            Backstory: {self.backstory}
            
            Evaluate the provided brand name based on comprehensive analysis results.
            Consider:
            1. Strategic Alignment & Brand Fit
            2. Distinctiveness & Competitive Advantage
            3. Memorability & Pronounceability
            4. Linguistic & Cultural Considerations
            5. Global Market Potential
            6. Digital Viability
            
            Format your response according to the following schema:
            {{format_instructions}}
            """,
            additional_kwargs={
                "metadata": {
                    "agent_type": "brand_name_evaluator",
                    "methodology": "Alina Wheeler's Designing Brand Identity"
                }
            }
        )
        human_template = """Evaluate the following brand name based on all analysis results:
        Brand Name: {brand_name}
        Brand Context: {brand_context}
        Semantic Analysis: {semantic_analysis}
        Linguistic Analysis: {linguistic_analysis}
        Cultural Analysis: {cultural_analysis}
        Translation Analysis: {translation_analysis}
        """
        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            HumanMessage(content=human_template)
        ])

    async def evaluate_brand_names(
        self,
        brand_names: List[str],
        semantic_analyses: List[Dict[str, Any]],
        linguistic_analyses: List[Dict[str, Any]],
        cultural_analyses: List[Dict[str, Any]],
        run_id: str,
        brand_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple brand names based on their analyses.
        
        Args:
            brand_names: List of brand names to evaluate
            semantic_analyses: List of semantic analysis results
            linguistic_analyses: List of linguistic analysis results  
            cultural_analyses: List of cultural analysis results
            run_id: Unique identifier for this workflow run
            brand_context: Brand context information
            
        Returns:
            List of evaluation results for each brand name
        """
        evaluation_results = []
        
        # Process each brand name
        for i, brand_name in enumerate(brand_names):
            try:
                # Get corresponding analyses (if available)
                semantic_analysis = semantic_analyses[i] if i < len(semantic_analyses) else {}
                linguistic_analysis = linguistic_analyses[i] if i < len(linguistic_analyses) else {}
                cultural_analysis = cultural_analyses[i] if i < len(cultural_analyses) else {}
                
                # Format prompt with all available information
                prompt = self.prompt.format_messages(
                    format_instructions=self.output_parser.get_format_instructions(),
                    brand_name=brand_name,
                    brand_context=json.dumps(brand_context, indent=2),
                    semantic_analysis=json.dumps(semantic_analysis, indent=2),
                    linguistic_analysis=json.dumps(linguistic_analysis, indent=2),
                    cultural_analysis=json.dumps(cultural_analysis, indent=2),
                    translation_analysis=json.dumps({})  # Empty for now as we're not passing translation analysis
                )
                
                # Get evaluation for this name
                evaluation = await self._evaluate_name(prompt)
                
                # Add brand name and run_id to the result
                evaluation["brand_name"] = brand_name
                evaluation["run_id"] = run_id
                
                # Store evaluation in results list
                evaluation_results.append(evaluation)
                
                # Also store in Supabase for persistence
                await self._store_in_supabase(run_id, evaluation)
                
            except Exception as e:
                logger.error(f"Error evaluating brand name '{brand_name}': {str(e)}")
                # Add minimal error result
                evaluation_results.append({
                    "brand_name": brand_name,
                    "run_id": run_id,
                    "error": str(e),
                    "overall_score": 0.0,
                    "shortlist_status": False
                })
        
        return evaluation_results

    async def _evaluate_name(self, prompt) -> Dict[str, Any]:
        """
        Private method to evaluate a single brand name.
        
        Args:
            prompt: Formatted prompt for the evaluation
            
        Returns:
            Dict[str, Any]: Structured evaluation results
        """
        try:
            # Setup event loop if not available
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Call the LLM with the formatted prompt
            response = await self.llm.ainvoke(prompt)
            
            # Parse the response according to the defined schema
            evaluation_result = self.output_parser.parse(response.content)
            
            # Ensure numeric values are properly typed
            numeric_fields = [
                "strategic_alignment_score", "distinctiveness_score", "brand_fit_score",
                "memorability_score", "pronounceability_score", "meaningfulness_score",
                "domain_viability_score", "overall_score", "rank"
            ]
            
            for field in numeric_fields:
                if field in evaluation_result:
                    try:
                        evaluation_result[field] = float(evaluation_result[field])
                    except (ValueError, TypeError):
                        # Default to middle score if conversion fails
                        evaluation_result[field] = 5.0
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating brand name: {str(e)}")
            # Return minimal default evaluation in case of error
            return {
                "strategic_alignment_score": 0.0,
                "distinctiveness_score": 0.0,
                "competitive_advantage": "Error in evaluation",
                "brand_fit_score": 0.0,
                "overall_score": 0.0,
                "shortlist_status": "No",
                "evaluation_comments": f"Error during evaluation: {str(e)}",
                "rank": 0.0
            }

    async def _store_in_supabase(self, run_id: str, evaluation_results: Dict[str, Any]) -> None:
        """
        Store the evaluation results in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            evaluation_results (Dict[str, Any]): The evaluation results to store
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Ensure numeric fields are properly typed
            numeric_fields = [
                "strategic_alignment_score", "distinctiveness_score", "brand_fit_score",
                "memorability_score", "pronounceability_score", "meaningfulness_score",
                "domain_viability_score", "overall_score", "rank"
            ]
            for field in numeric_fields:
                if field in evaluation_results:
                    evaluation_results[field] = float(evaluation_results[field])
            
            # Insert into brand_name_evaluation table
            await self.supabase.table("brand_name_evaluation").insert(evaluation_results).execute()
            logger.info(f"Stored brand name evaluation for '{evaluation_results.get('brand_name', 'unknown')}' with run_id '{run_id}'")
            
        except Exception as e:
            error_msg = f"Error storing evaluation in Supabase: {str(e)}"
            logger.error(error_msg)
            raise  # Re-raise to handle in calling function 