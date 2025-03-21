"""Brand Name Evaluator for final assessment and shortlisting of brand names."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio
from pathlib import Path
import re

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_enabled
from langchain_core.tracers import LangChainTracer
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
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
                
                # Add to results list without storing in Supabase yet
                evaluations.append(evaluation)
                
            except Exception as e:
                logger.error(f"Error evaluating {brand_name}: {str(e)}")
                # Continue with other names if one fails
        
        if evaluations:
            # Determine shortlist status based on comparative analysis and full brand context
            evaluations = await self._determine_shortlist_status(
                evaluations,
                semantic_analyses,
                linguistic_analyses,
                cultural_analyses,
                brand_context  # Pass the full brand context for in-depth analysis
            )
            
            # Now store the updated evaluations in Supabase
            for evaluation in evaluations:
                try:
                    await self._store_in_supabase(run_id, evaluation)
                except Exception as e:
                    logger.error(f"Error storing evaluation for {evaluation.get('brand_name', 'unknown')} in Supabase: {str(e)}")
        
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
            # Generate completion - ensure we're using the correct invoke pattern
            # prompt may already be a list of message objects
            response = await self.llm.ainvoke(prompt)
            
            # Parse the response according to the defined schema
            # Ensure response.content is accessed properly (response might be AIMessage)
            content = response.content if hasattr(response, 'content') else str(response)
            evaluation_result = self.output_parser.parse(content)
            
            # Ensure numeric values are properly typed and constrained
            numeric_fields = [
                "strategic_alignment_score",
                "distinctiveness_score",
                "brand_fit_score",
                "memorability_score",
                "pronounceability_score",
                "meaningfulness_score", 
                "domain_viability_score",
                "overall_score",
                "rank"
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
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating brand name: {str(e)}")
            # Return minimal default evaluation in case of error
            return {
                "strategic_alignment_score": 5.0,
                "distinctiveness_score": 5.0,
                "brand_fit_score": 5.0,
                "memorability_score": 5.0,
                "pronounceability_score": 5.0,
                "meaningfulness_score": 5.0,
                "phonetic_harmony": "Not evaluated due to error",
                "visual_branding_potential": "Not evaluated due to error",
                "storytelling_potential": "Not evaluated due to error",
                "competitive_advantage": "Error in evaluation",
                "positioning_strength": "Error in evaluation",
                "domain_viability_score": 5,
                "overall_score": 5,
                "shortlist_status": False,
                "evaluation_comments": f"Error during evaluation: {str(e)}",
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
            if "brand_name" not in evaluation or not evaluation["brand_name"]:
                logger.error("Cannot store evaluation: missing brand_name field")
                return
                
            # Ensure run_id is set
            evaluation["run_id"] = run_id
            
            # Ensure shortlist_status is a boolean (not nullable)
            if "shortlist_status" not in evaluation or evaluation["shortlist_status"] is None:
                evaluation["shortlist_status"] = False
            elif isinstance(evaluation["shortlist_status"], str):
                evaluation["shortlist_status"] = evaluation["shortlist_status"].lower() in ["true", "yes", "1", "t", "y"]
            else:
                evaluation["shortlist_status"] = bool(evaluation["shortlist_status"])
            
            # Convert numeric scores to integers and ensure they're within range 1-10
            integer_score_fields = [
                "strategic_alignment_score", "distinctiveness_score", "brand_fit_score",
                "memorability_score", "pronounceability_score", "meaningfulness_score",
                "domain_viability_score", "overall_score"
            ]
            
            for field in integer_score_fields:
                if field in evaluation and evaluation[field] is not None:
                    try:
                        # Convert to integer
                        value = int(float(evaluation[field]))
                        # Constrain to range 1-10
                        evaluation[field] = max(1, min(10, value))
                    except (ValueError, TypeError):
                        # Default to middle value if conversion fails
                        evaluation[field] = 5
                else:
                    # Set default value if missing
                    evaluation[field] = 5
            
            # Ensure rank is numeric but doesn't need to be an integer
            if "rank" in evaluation and evaluation["rank"] is not None:
                if not isinstance(evaluation["rank"], (int, float)):
                    try:
                        evaluation["rank"] = float(evaluation["rank"])
                    except (ValueError, TypeError):
                        evaluation["rank"] = 0.0
            
            # Filter evaluation data to include only fields present in the Supabase schema
            # This prevents errors when trying to insert fields that don't exist in the table
            allowed_fields = [
                "brand_name", "run_id", "strategic_alignment_score", "distinctiveness_score", 
                "brand_fit_score", "memorability_score", "pronounceability_score", 
                "meaningfulness_score", "domain_viability_score", "overall_score", 
                "shortlist_status", "evaluation_comments", "rank"
            ]
            
            filtered_evaluation = {k: v for k, v in evaluation.items() if k in allowed_fields}
            
            # Log the data being inserted
            logger.info(f"Storing evaluation for brand name '{filtered_evaluation['brand_name']}' with run_id '{run_id}'")
            logger.debug(f"Evaluation data: {json.dumps(filtered_evaluation, default=str)}")
            
            # Use the async execute_with_retry method instead of direct table operations
            await self.supabase.execute_with_retry("insert", "brand_name_evaluation", filtered_evaluation)
            logger.info(f"Successfully stored evaluation in Supabase")
            
        except Exception as e:
            logger.error(f"Error storing evaluation in Supabase: {str(e)}")
            # Continue execution even if storage fails 

    async def _determine_shortlist_status(self, 
                                        evaluations: List[Dict[str, Any]], 
                                        semantic_analyses: Dict[str, Dict[str, Any]], 
                                        linguistic_analyses: Dict[str, Dict[str, Any]],
                                        cultural_analyses: Dict[str, Dict[str, Any]],
                                        brand_context: str = None) -> List[Dict[str, Any]]:
        """
        Determine which brand names should be shortlisted based on comprehensive analysis
        and relative comparison. Only shortlists the top 3 names.
        
        Args:
            evaluations: List of evaluation results for each brand name
            semantic_analyses: Dictionary of semantic analysis results for each brand name
            linguistic_analyses: Dictionary of linguistic analysis results for each brand name
            cultural_analyses: Dictionary of cultural analysis results for each brand name
            brand_context: The full brand context information used to make more informed decisions
            
        Returns:
            Updated list of evaluations with revised shortlist status
        """
        if not evaluations or len(evaluations) <= 3:
            # If there are 3 or fewer names, all are shortlisted by default
            for eval_data in evaluations:
                eval_data['shortlist_status'] = True
            return evaluations
        
        # If we have the brand context, use LLM-based approach for deeper contextual analysis
        if brand_context:
            logger.info(f"Using full brand context to determine shortlist status for {len(evaluations)} brand names")
            return await self._context_based_shortlisting(evaluations, brand_context)
        else:
            logger.warning("Brand context not provided for shortlisting. Using score-based approach.")
            return await self._score_based_shortlisting(evaluations, semantic_analyses, linguistic_analyses, cultural_analyses)
            
    async def _context_based_shortlisting(self, evaluations: List[Dict[str, Any]], brand_context: str) -> List[Dict[str, Any]]:
        """
        Uses the full brand context with LLM to make a more informed decision about which names to shortlist.
        This goes beyond simple score-based metrics and considers how well each name aligns with the brand context.
        
        Args:
            evaluations: List of evaluation results for each brand name
            brand_context: The full brand context information
            
        Returns:
            Updated list of evaluations with shortlist status determined via brand context analysis
        """
        # Extract the most relevant information for each brand name
        brand_summaries = []
        for eval_data in evaluations:
            brand_name = eval_data.get('brand_name')
            if not brand_name:
                continue
                
            summary = {
                'brand_name': brand_name,
                'evaluation_summary': eval_data.get('evaluation_comments', ''),
                'strategic_alignment_score': eval_data.get('strategic_alignment_score', 5),
                'brand_fit_score': eval_data.get('brand_fit_score', 5),
                'distinctiveness_score': eval_data.get('distinctiveness_score', 5),
                'memorability_score': eval_data.get('memorability_score', 5),
                'overall_score': eval_data.get('overall_score', 5)
            }
            brand_summaries.append(summary)
        
        # Prepare the prompt for contextual shortlisting
        shortlist_system = """You are a Strategic Brand Naming Expert responsible for selecting the final shortlist of brand names.
Your task is to evaluate a set of brand names against the provided brand context and select ONLY the top 3 
that best align with the brand's identity, values, target audience, and strategic objectives.

Your analysis must consider:
1. How well each name embodies the brand's core values and personality
2. Alignment with target audience expectations and preferences
3. Support for the brand's strategic positioning and differentiation
4. Overall brand name quality, memorability, and distinctiveness

Be extremely selective and critical in your assessment, focusing on substantive strategic fit 
rather than just numeric scores. Your shortlist should represent truly exceptional options that 
deeply connect with the brand's essence and market position."""

        # Create a concise version of the brand context if it's too long
        brand_context_summary = brand_context
        if len(brand_context) > 2000:
            brand_context_summary = brand_context[:2000] + "... [truncated for length]"
        
        shortlist_human = f"""BRAND CONTEXT:
{brand_context_summary}

BRAND NAME EVALUATIONS:
{json.dumps(brand_summaries, indent=2)}

Based on the above BRAND CONTEXT and evaluations, determine which 3 brand names should be shortlisted.
Your response must be in JSON format with the following structure:
{{
  "shortlisted_names": ["Name1", "Name2", "Name3"],
  "rationale": "Your detailed explanation of why these 3 names were selected based on the brand context",
  "individual_assessments": {{
    "Name1": "Specific reasons this name aligns with the brand context",
    "Name2": "Specific reasons this name aligns with the brand context",
    "Name3": "Specific reasons this name aligns with the brand context"
  }}
}}

Focus specifically on how each name aligns with the BRAND CONTEXT, not just general branding principles.
Ensure your rationale references specific elements from the brand context document.
"""

        messages = [
            SystemMessage(content=shortlist_system),
            HumanMessage(content=shortlist_human)
        ]
        
        try:
            # Get the shortlist decision from the LLM
            shortlist_response = await self.llm.ainvoke(messages)
            content = shortlist_response.content if hasattr(shortlist_response, 'content') else str(shortlist_response)
            
            # Parse the JSON response
            try:
                # Extract JSON from the response (in case there's other text)
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)
                
                shortlist_data = json.loads(content)
                shortlisted_names = shortlist_data.get('shortlisted_names', [])
                rationale = shortlist_data.get('rationale', '')
                individual_assessments = shortlist_data.get('individual_assessments', {})
                
                if not shortlisted_names:
                    logger.warning("LLM did not return any shortlisted names. Falling back to score-based shortlisting.")
                    return await self._score_based_shortlisting(evaluations, {}, {}, {})
                
                logger.info(f"Context-based shortlisting selected names: {', '.join(shortlisted_names)}")
                
                # Update evaluations with shortlist status and contextual rationale
                for eval_data in evaluations:
                    brand_name = eval_data.get('brand_name')
                    if not brand_name:
                        continue
                        
                    is_shortlisted = brand_name in shortlisted_names
                    eval_data['shortlist_status'] = is_shortlisted
                    
                    # Add detailed contextual reasoning to the evaluation comments
                    if 'evaluation_comments' in eval_data:
                        if is_shortlisted:
                            assessment = individual_assessments.get(brand_name, "Selected based on strong alignment with brand context.")
                            eval_data['evaluation_comments'] += f"\n\nSHORTLISTED (CONTEXT ANALYSIS): {assessment}"
                        else:
                            eval_data['evaluation_comments'] += f"\n\nNOT SHORTLISTED (CONTEXT ANALYSIS): This name was not selected when evaluated against the full brand context."
                
                # Add the overall rationale to the first shortlisted name's comments
                shortlist_found = False
                for eval_data in evaluations:
                    if eval_data.get('shortlist_status', False) and 'evaluation_comments' in eval_data:
                        eval_data['evaluation_comments'] += f"\n\nOVERALL SHORTLISTING RATIONALE: {rationale}"
                        shortlist_found = True
                        break
                
                # If no shortlisted names were found in our evaluations (unlikely but possible),
                # add the rationale to the first evaluation
                if not shortlist_found and evaluations and 'evaluation_comments' in evaluations[0]:
                    evaluations[0]['evaluation_comments'] += f"\n\nNOTE: LLM SHORTLISTING RATIONALE: {rationale}"
                
                return evaluations
                
            except Exception as e:
                logger.error(f"Error parsing LLM shortlist response: {str(e)}. Response: {content[:200]}...")
                logger.warning("Falling back to score-based shortlisting.")
                return await self._score_based_shortlisting(evaluations, {}, {}, {})
                
        except Exception as e:
            logger.error(f"Error in context-based shortlisting: {str(e)}")
            logger.warning("Falling back to score-based shortlisting.")
            return await self._score_based_shortlisting(evaluations, {}, {}, {})

    async def _score_based_shortlisting(self, 
                                     evaluations: List[Dict[str, Any]], 
                                     semantic_analyses: Dict[str, Dict[str, Any]], 
                                     linguistic_analyses: Dict[str, Dict[str, Any]],
                                     cultural_analyses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method that determines shortlist status based on scores.
        """
        logger.info(f"Using score-based approach to shortlist from {len(evaluations)} brand names")
            
        # Calculate composite scores that consider all analyses
        for eval_data in evaluations:
            brand_name = eval_data.get('brand_name')
            if not brand_name:
                continue
                
            # Collect metrics from all analyses
            semantic_metrics = semantic_analyses.get(brand_name, {})
            linguistic_metrics = linguistic_analyses.get(brand_name, {})
            cultural_metrics = cultural_analyses.get(brand_name, {})
            
            # Define weights for different scores with increased emphasis on brand context alignment
            weights = {
                'strategic_alignment': 0.20,  # Increased weight - this directly relates to brand context
                'brand_fit': 0.20,           # Increased weight - this directly relates to brand context
                'distinctiveness': 0.12,
                'memorability': 0.10,
                'pronounceability': 0.08,
                'meaningfulness': 0.08,
                'domain_viability': 0.07,
                'semantic_quality': 0.05,
                'linguistic_quality': 0.05,
                'cultural_sensitivity': 0.05
            }
            
            # Get scores from evaluation
            eval_scores = {
                'strategic_alignment': eval_data.get('strategic_alignment_score', 5),
                'distinctiveness': eval_data.get('distinctiveness_score', 5),
                'brand_fit': eval_data.get('brand_fit_score', 5),
                'memorability': eval_data.get('memorability_score', 5),
                'pronounceability': eval_data.get('pronounceability_score', 5),
                'meaningfulness': eval_data.get('meaningfulness_score', 5),
                'domain_viability': eval_data.get('domain_viability_score', 5)
            }
            
            # Get scores from other analyses
            semantic_rank = semantic_metrics.get('rank', 5) if isinstance(semantic_metrics, dict) else 5
            linguistic_rank = linguistic_metrics.get('rank', 5) if isinstance(linguistic_metrics, dict) else 5
            
            # For cultural risk, lower is better (inverse the score)
            cultural_risk_str = str(cultural_metrics.get('overall_risk_rating', '5/10')).split('/')[0] if isinstance(cultural_metrics, dict) else '5'
            try:
                cultural_risk = float(cultural_risk_str)
                cultural_sensitivity_score = 10 - cultural_risk  # Invert so higher is better
            except (ValueError, TypeError):
                cultural_sensitivity_score = 5
                
            other_scores = {
                'semantic_quality': semantic_rank,
                'linguistic_quality': linguistic_rank,
                'cultural_sensitivity': cultural_sensitivity_score
            }
            
            # Combine scores
            all_scores = {**eval_scores, **other_scores}
            
            # Calculate weighted composite score
            composite_score = sum(all_scores[key] * weights[key] for key in weights)
            
            # Add to evaluation data
            eval_data['composite_score'] = composite_score
            
            # Apply a brand context alignment bonus/penalty based on strategic alignment and brand fit
            # This emphasizes names that specifically align well with the brand context
            brand_context_alignment_score = (eval_scores['strategic_alignment'] + eval_scores['brand_fit']) / 2
            if brand_context_alignment_score >= 8:
                # Apply a bonus for exceptional brand context alignment
                context_multiplier = 1.1  # 10% bonus
            elif brand_context_alignment_score <= 4:
                # Apply a penalty for poor brand context alignment
                context_multiplier = 0.9  # 10% penalty
            else:
                context_multiplier = 1.0
                
            # Apply the context multiplier to the composite score
            eval_data['composite_score'] *= context_multiplier
            
            # Add an explanation about brand context consideration
            eval_data['context_alignment_note'] = f"Brand context alignment {'bonus' if context_multiplier > 1 else 'penalty' if context_multiplier < 1 else 'neutral'} applied: {context_multiplier:.2f}x multiplier based on strategic alignment and brand fit scores."
            
        # Sort by composite score (descending)
        sorted_evals = sorted(evaluations, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Get the threshold for significant difference (average score drop after top 3)
        if len(sorted_evals) > 4:
            top_3_avg = sum(e.get('composite_score', 0) for e in sorted_evals[:3]) / 3
            next_group_avg = sum(e.get('composite_score', 0) for e in sorted_evals[3:min(6, len(sorted_evals))]) / min(3, len(sorted_evals) - 3)
            score_threshold = top_3_avg - ((top_3_avg - next_group_avg) * 0.5)  # Midpoint between top 3 and next group
        else:
            score_threshold = 0
            
        # Mark only top 3 as shortlisted, and only if their score is above the threshold
        for i, eval_data in enumerate(sorted_evals):
            should_shortlist = i < 3 and eval_data.get('composite_score', 0) > score_threshold
            sorted_evals[i]['shortlist_status'] = should_shortlist
            
            # Update evaluation comments to explain shortlist decision
            comment_addition = ""
            if should_shortlist:
                comment_addition = f"\n\nSHORTLISTED (SCORE-BASED): This name ranked #{i+1} in the comprehensive evaluation with a composite score of {eval_data.get('composite_score', 0):.2f}."
            else:
                reason = "below quality threshold" if i < 3 else f"ranked #{i+1}"
                comment_addition = f"\n\nNOT SHORTLISTED (SCORE-BASED): This name was {reason} with a composite score of {eval_data.get('composite_score', 0):.2f}."
                
            # Add context alignment note to the comments
            if 'context_alignment_note' in eval_data:
                comment_addition += f" {eval_data['context_alignment_note']}"
                del eval_data['context_alignment_note']  # Remove temporary field
                
            if 'evaluation_comments' in sorted_evals[i]:
                sorted_evals[i]['evaluation_comments'] += comment_addition
            
        # Log the shortlisting decisions
        shortlisted = [e.get('brand_name') for e in sorted_evals if e.get('shortlist_status', False)]
        logger.info(f"Score-based shortlisting selected {len(shortlisted)} brand names: {', '.join(shortlisted)}")
        
        return sorted_evals 