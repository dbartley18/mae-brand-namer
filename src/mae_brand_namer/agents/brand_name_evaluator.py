"""Brand Name Evaluator for final assessment and shortlisting of brand names."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio
from pathlib import Path

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages.base import BaseMessage

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager

# Configure logging
logger = get_logger(__name__)

class BrandNameEvaluator:
    """
    Expert agent responsible for evaluating brand name candidates.
    
    This agent applies comprehensive evaluation methodologies to assess
    brand names across multiple dimensions based on all available analysis.
    """
    
    def __init__(
        self,
        supabase,
        dependencies=None,
        role: str = "Brand Name Evaluation Expert",
        goal: str = "Comprehensively evaluate brand name candidates based on all analysis",
        backstory: str = "Expert in brand evaluation with cross-disciplinary expertise in linguistics, marketing, and brand strategy.",
        **kwargs
    ):
        """
        Initialize the BrandNameEvaluator expert agent.
        
        Args:
            supabase: Supabase client for data persistence
            dependencies: Optional dependencies
            role (str): Agent role identity
            goal (str): Agent goal description
            backstory (str): Agent backstory for context
        """
        # Initialize dependencies
        self.supabase = supabase
        if dependencies:
            self.langsmith = dependencies.langsmith
        else:
            self.langsmith = None
            
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=1.0,  # Balanced temperature for analysis
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )
        
        # Agent identity
        self.role = role
        self.goal = goal
        self.backstory = backstory
        
        logger.info(f"Initialized {self.role} with goal: {self.goal}")
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="strategic_alignment_score", description="How well the name aligns with the Brand Identity Brief (1-10, integer)", type="integer"),
            ResponseSchema(name="distinctiveness_score", description="How unique the name is compared to competitors (1-10, integer)", type="integer"),
            ResponseSchema(name="competitive_advantage", description="Analysis of competitive differentiation"),
            ResponseSchema(name="brand_fit_score", description="How well the name aligns with brand strategy (1-10, integer)", type="integer"),
            ResponseSchema(name="positioning_strength", description="Effectiveness in market positioning"),
            ResponseSchema(name="memorability_score", description="How easy the name is to recall (1-10, integer)", type="integer"),
            ResponseSchema(name="pronounceability_score", description="How easily the name is spoken (1-10, integer)", type="integer"),
            ResponseSchema(name="meaningfulness_score", description="Clarity and positive connotation (1-10, integer)", type="integer"),
            ResponseSchema(name="phonetic_harmony", description="Analysis of sound patterns and flow"),
            ResponseSchema(name="visual_branding_potential", description="Potential for visual identity development"),
            ResponseSchema(name="storytelling_potential", description="Capacity for brand narrative development"),
            ResponseSchema(name="domain_viability_score", description="Initial domain name availability assessment (1-10, integer)", type="integer"),
            ResponseSchema(name="overall_score", description="Total weighted evaluation score (1-10, integer)", type="integer"),
            ResponseSchema(name="shortlist_status", description="Whether selected for final round (true/false)", type="boolean"),
            ResponseSchema(name="evaluation_comments", description="Detailed rationale for evaluation"),
            ResponseSchema(name="rank", description="Final ranking among all candidates (1-N)", type="integer")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)
        
        # Load prompts from YAML files
        self._load_prompts()
        
        logger.info(f"Initialized BrandNameEvaluator with output parser")

    def _load_prompts(self):
        """Load prompts from YAML files."""
        try:
            # Load system and evaluation prompts from YAML files
            prompt_dir = Path(__file__).parent / "prompts" / "brand_name_evaluator"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.evaluation_prompt = load_prompt(str(prompt_dir / "evaluation.yaml"))
            
            # Get format instructions from the output parser
            format_instructions = self.output_parser.get_format_instructions()
            
            logger.info(f"Loaded brand name evaluator prompt templates")
            logger.debug(f"Evaluation prompt variables: {self.evaluation_prompt.input_variables}")
            
            # Create the prompt template with loaded YAML files
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt.template),
                ("human", self.evaluation_prompt.template)
            ])
            
        except Exception as e:
            logger.error(f"Error loading prompt templates: {str(e)}")
            # Fallback to hardcoded prompts if loading fails
            system_message = SystemMessage(
                content="You are a Brand Name Evaluation Expert specializing in holistic brand analysis and strategic assessment. Your evaluation synthesizes linguistic, semantic, cultural, and market dimensions to provide comprehensive brand name assessments."
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
            logger.warning(f"Using fallback prompts due to template loading error")

    async def evaluate_brand_names(
        self,
        brand_names: List[str],
        semantic_analyses: Dict[str, Dict[str, Any]],
        linguistic_analyses: Dict[str, Dict[str, Any]],
        cultural_analyses: Dict[str, Dict[str, Any]],
        run_id: str,
        brand_context: str,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple brand names based on their analyses.
        
        Args:
            brand_names: List of brand names to evaluate
            semantic_analyses: Dictionary mapping brand names to semantic analysis
            linguistic_analyses: Dictionary mapping brand names to linguistic analysis
            cultural_analyses: Dictionary mapping brand names to cultural analysis
            run_id: Unique identifier for this workflow run
            brand_context: Brand context information
            
        Returns:
            List[Dict[str, Any]]: List of evaluation results for each brand name
        """
        logger.info(f"Evaluating {len(brand_names)} brand names")
        
        # List to store evaluation results
        evaluations = []
        
        for brand_name in brand_names:
            # Get the analyses for this brand name
            semantic_analysis = semantic_analyses.get(brand_name, {}).get("analysis", "No semantic analysis available")
            linguistic_analysis = linguistic_analyses.get(brand_name, {}).get("analysis", "No linguistic analysis available")
            cultural_analysis = cultural_analyses.get(brand_name, {}).get("analysis", "No cultural analysis available")
            
            # Create a combined translation analysis text from all available languages
            translation_analysis = "No translation analysis available"
            
            # Format prompt with all available information
            messages = self.prompt.format_messages(
                format_instructions=self.output_parser.get_format_instructions(),
                brand_name=brand_name,
                brand_context=brand_context,
                semantic_analysis=semantic_analysis,
                linguistic_analysis=linguistic_analysis,
                cultural_analysis=cultural_analysis,
                translation_analysis=translation_analysis
            )
            
            try:
                # Evaluate the brand name
                evaluation = await self._evaluate_name(messages)
                
                # Add the brand name and run_id to the evaluation for Supabase
                evaluation["brand_name"] = brand_name
                evaluation["run_id"] = run_id
                
                # Store in Supabase for persistence
                await self._store_in_supabase(run_id, evaluation)
                
                # Add to results list
                evaluations.append(evaluation)
                
            except Exception as e:
                logger.error(f"Error evaluating {brand_name}: {str(e)}")
                # Continue with other names if one fails
        
        logger.info(f"Completed evaluation of {len(evaluations)} brand names")
        return evaluations

    async def _evaluate_name(self, prompt) -> Dict[str, Any]:
        """
        Private method to evaluate a single brand name.
        
        Args:
            prompt: Formatted prompt messages for the evaluation
            
        Returns:
            Dict[str, Any]: Structured evaluation results
        """
        try:
            # Generate completion
            response = await self.llm.invoke(prompt)
            
            # Extract content from AIMessage if needed
            content = response.content if hasattr(response, "content") else str(response)
            
            # Parse the response according to the defined schema
            evaluation_result = self.output_parser.parse(content)
            
            # Ensure numeric values are properly typed and constrained
            numeric_fields = [
                "strategic_alignment_score", "distinctiveness_score", "brand_fit_score",
                "memorability_score", "pronounceability_score", "meaningfulness_score",
                "domain_viability_score", "overall_score", "rank"
            ]
            
            for field in numeric_fields:
                if field in evaluation_result:
                    try:
                        # Convert to int and constrain between 1-10
                        value = int(float(evaluation_result[field]))
                        evaluation_result[field] = max(1, min(10, value))  # Clamp between 1-10
                    except (ValueError, TypeError):
                        # Default to middle score if conversion fails
                        evaluation_result[field] = 5
                else:
                    # Add missing numeric fields with default value
                    evaluation_result[field] = 5
            
            # Ensure shortlist_status is a boolean
            if "shortlist_status" in evaluation_result:
                status = evaluation_result["shortlist_status"]
                if isinstance(status, str):
                    evaluation_result["shortlist_status"] = status.lower() in ["true", "yes", "1", "t", "y"]
                else:
                    evaluation_result["shortlist_status"] = bool(status)
            else:
                # Default to False if missing
                evaluation_result["shortlist_status"] = False
            
            # Ensure required string fields are present
            string_fields = [
                "competitive_advantage", "positioning_strength", "phonetic_harmony",
                "visual_branding_potential", "storytelling_potential", "evaluation_comments"
            ]
            
            for field in string_fields:
                if field not in evaluation_result or not evaluation_result[field]:
                    evaluation_result[field] = "Not evaluated"
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating brand name: {str(e)}")
            # Return minimal default evaluation in case of error
            return {
                "strategic_alignment_score": 5,
                "distinctiveness_score": 5,
                "competitive_advantage": "Error in evaluation",
                "brand_fit_score": 5,
                "positioning_strength": "Not evaluated",
                "memorability_score": 5,
                "pronounceability_score": 5,
                "meaningfulness_score": 5,
                "phonetic_harmony": "Not evaluated",
                "visual_branding_potential": "Not evaluated",
                "storytelling_potential": "Not evaluated",
                "domain_viability_score": 5,
                "overall_score": 5,
                "shortlist_status": False,
                "evaluation_comments": f"Error during evaluation: {str(e)}",
                "rank": 5
            }

    async def _store_in_supabase(self, run_id: str, evaluation: Dict[str, Any]) -> None:
        """
        Store evaluation results in Supabase.
        
        Args:
            run_id: Unique identifier for this workflow run
            evaluation: Evaluation results to store
        """
        try:
            # Ensure required fields are present
            if "brand_name" not in evaluation:
                logger.error("Cannot store evaluation: missing brand_name field")
                return
                
            # Ensure run_id is set
            evaluation["run_id"] = run_id
            
            # Create a copy of the evaluation to modify for Supabase
            supabase_data = evaluation.copy()
            
            # Convert shortlist_status from string to boolean for Supabase
            if "shortlist_status" in supabase_data:
                if isinstance(supabase_data["shortlist_status"], str):
                    supabase_data["shortlist_status"] = supabase_data["shortlist_status"].lower() in ["yes", "true", "1", "t", "y"]
            else:
                supabase_data["shortlist_status"] = False
            
            # Convert numeric scores to integers and ensure they're within range 1-10
            integer_score_fields = [
                "strategic_alignment_score", "distinctiveness_score", "brand_fit_score",
                "memorability_score", "pronounceability_score", "meaningfulness_score",
                "domain_viability_score", "overall_score"
            ]
            
            for field in integer_score_fields:
                if field in supabase_data:
                    try:
                        # Convert to integer
                        value = int(float(supabase_data[field]))
                        # Constrain to range 1-10
                        supabase_data[field] = max(1, min(10, value))
                    except (ValueError, TypeError):
                        # Default to middle value if conversion fails
                        supabase_data[field] = 5
                else:
                    # Set default value if missing
                    supabase_data[field] = 5
            
            # Ensure rank is numeric but doesn't need to be an integer
            if "rank" in supabase_data and not isinstance(supabase_data["rank"], (int, float)):
                try:
                    supabase_data["rank"] = float(supabase_data["rank"])
                except (ValueError, TypeError):
                    supabase_data["rank"] = 5.0
            elif "rank" not in supabase_data:
                supabase_data["rank"] = 5.0
            
            # Ensure required string fields are present
            string_fields = [
                "competitive_advantage", "positioning_strength", "phonetic_harmony",
                "visual_branding_potential", "storytelling_potential", "evaluation_comments"
            ]
            
            for field in string_fields:
                if field not in supabase_data or not supabase_data[field]:
                    supabase_data[field] = "Not evaluated"
            
            # Add created_at timestamp if not present
            if "created_at" not in supabase_data:
                supabase_data["created_at"] = datetime.now().isoformat()
            
            # Log the data being inserted
            logger.info(f"Storing evaluation for brand name '{supabase_data['brand_name']}' with run_id '{run_id}'")
            
            # Insert into Supabase
            result = await self.supabase.table("brand_name_evaluation").insert(supabase_data).execute()
            logger.info(f"Successfully stored evaluation in Supabase")
            
        except Exception as e:
            logger.error(f"Error storing evaluation in Supabase: {str(e)}")
            # Continue execution even if storage fails 