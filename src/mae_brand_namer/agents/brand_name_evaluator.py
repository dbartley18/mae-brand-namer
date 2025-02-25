"""Brand Name Evaluator for final assessment and shortlisting of brand names."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

class BrandNameEvaluator:
    """Expert in evaluating and shortlisting brand names based on comprehensive analysis."""
    
    def __init__(self):
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
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Initialize Supabase client
        self.supabase: Client = create_client(settings.supabase_url, settings.supabase_key)
        
        # Initialize LangSmith tracer if enabled
        self.tracer = None
        if settings.tracing_enabled:
            self.tracer = LangChainTracer(project_name=settings.langsmith_project)
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
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

    def evaluate_brand_names(
        self,
        run_id: str,
        brand_names: List[Dict[str, Any]],
        brand_context: Dict[str, Any],
        semantic_analysis: Dict[str, Any],
        linguistic_analysis: Dict[str, Any],
        cultural_analysis: Dict[str, Any],
        translation_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate and shortlist brand names based on comprehensive analysis results.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_names (List[Dict[str, Any]]): List of brand names to evaluate
            brand_context (Dict[str, Any]): Brand context information
            semantic_analysis (Dict[str, Any]): Results of semantic analysis
            linguistic_analysis (Dict[str, Any]): Results of linguistic analysis
            cultural_analysis (Dict[str, Any]): Results of cultural sensitivity analysis
            translation_analysis (Dict[str, Any]): Results of translation analysis
            
        Returns:
            Dict[str, Any]: Evaluation results for each brand name
        """
        with tracing_enabled(tags={"agent": "BrandNameEvaluator", "run_id": run_id}):
            try:
                evaluated_names = {}
                
                # Evaluate each brand name
                for name_data in brand_names:
                    brand_name = name_data["brand_name"]
                    
                    # Format the prompt with all analysis results
                    formatted_prompt = self.prompt.format(
                        format_instructions=self.output_parser.get_format_instructions(),
                        brand_name=brand_name,
                        brand_context=json.dumps(brand_context),
                        semantic_analysis=json.dumps(semantic_analysis.get(brand_name, {})),
                        linguistic_analysis=json.dumps(linguistic_analysis.get(brand_name, {})),
                        cultural_analysis=json.dumps(cultural_analysis.get(brand_name, {})),
                        translation_analysis=json.dumps(translation_analysis.get(brand_name, {}))
                    )
                    
                    # Get response from LLM
                    response = self.llm.invoke(formatted_prompt)
                    
                    # Parse the structured output
                    parsed_output = self.output_parser.parse(response.content)
                    
                    # Add required fields
                    evaluation_results = {
                        "run_id": run_id,
                        "brand_name": brand_name,
                        **parsed_output
                    }
                    
                    # Store in Supabase
                    self._store_in_supabase(run_id, evaluation_results)
                    
                    evaluated_names[brand_name] = evaluation_results
                
                # Select top 3 names based on overall_score
                shortlisted_names = sorted(
                    evaluated_names.values(),
                    key=lambda x: x["overall_score"],
                    reverse=True
                )[:3]
                
                # Update shortlist status
                for name in evaluated_names.values():
                    name["shortlist_status"] = "Yes" if name in shortlisted_names else "No"
                    self._store_in_supabase(run_id, name)  # Update in Supabase
                
                return evaluated_names
                
            except Exception as e:
                error_msg = f"Error evaluating brand names: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _store_in_supabase(self, run_id: str, evaluation_results: Dict[str, Any]) -> None:
        """
        Store the evaluation results in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            evaluation_results (Dict[str, Any]): The evaluation results to store
        """
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
            self.supabase.table("brand_name_evaluation").insert(evaluation_results).execute()
            
        except Exception as e:
            error_msg = f"Error storing evaluation in Supabase: {str(e)}"
            logger.error(error_msg)
            raise  # Re-raise to handle in calling function 