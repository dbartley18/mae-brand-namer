"""Expert in analyzing brand names in the context of market competition."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, load_prompt
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import tracing_v2_enabled
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool

from ..config.settings import settings
from ..utils.logging import get_logger
from ..config.dependencies import Dependencies

logger = get_logger(__name__)


class CompetitorAnalysisExpert:
    """Expert in analyzing brand names in the context of market competition.
    
    This expert evaluates brand names against market competitors, analyzing
    differentiation, positioning, and competitive advantage potential.
    
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
        """Initialize the CompetitorAnalysisExpert with dependencies.
        
        Args:
            dependencies: Container for application dependencies
        """
        self.supabase = dependencies.supabase
        self.langsmith = dependencies.langsmith
        
        # Agent identity
        self.role = "Competitor Analysis & Market Positioning Expert"
        self.goal = (
            "Evaluate brand names in the context of market competition, analyzing "
            "differentiation, positioning, and competitive advantage potential."
        )
        
        # Load prompts from YAML files
        prompt_dir = Path(__file__).parent / "prompts" / "competitor_analysis"
        self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
        self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(
                name="competitor_name",
                description="Name of the primary competitor being analyzed",
                type="string"
            ),
            ResponseSchema(
                name="competitor_naming_style",
                description="Whether competitors use descriptive, abstract, or other naming styles",
                type="string"
            ),
            ResponseSchema(
                name="competitor_keywords",
                description="Common words or themes in competitor brand names",
                type="string"
            ),
            ResponseSchema(
                name="competitor_positioning",
                description="How competitors position their brands in the market",
                type="string"
            ),
            ResponseSchema(
                name="competitor_strengths",
                description="Strengths of competitor brand names",
                type="string"
            ),
            ResponseSchema(
                name="competitor_weaknesses",
                description="Weaknesses of competitor brand names",
                type="string"
            ),
            ResponseSchema(
                name="competitor_differentiation_opportunity",
                description="How to create differentiation from competitors",
                type="string"
            ),
            ResponseSchema(
                name="differentiation_score",
                description="Quantified differentiation from competitors (1-10)",
                type="number"
            ),
            ResponseSchema(
                name="risk_of_confusion",
                description="Likelihood of brand confusion with competitors",
                type="string"
            ),
            ResponseSchema(
                name="target_audience_perception",
                description="How consumers may compare this name to competitors",
                type="string"
            ),
            ResponseSchema(
                name="competitive_advantage_notes",
                description="Any competitive advantages of the brand name",
                type="string"
            ),
            ResponseSchema(
                name="trademark_conflict_risk",
                description="Potential conflicts with existing trademarks",
                type="string"
            )
        ]
        
        # Initialize the output parser before trying to use it
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.output_schemas
        )
        
        # Get format instructions from the output parser
        self.format_instructions = self.output_parser.get_format_instructions()
        
        # Log the full prompt templates for debugging
        logger.debug(f"Analysis template content: {self.analysis_prompt.template[:200]}...")
        logger.debug(f"Analysis template has brand_name placeholder: {'{{brand_name}}' in self.analysis_prompt.template}")
        
        # Create Google Search tool
        search_tool = {
            "type": "google_search",
            "google_search": {}
        }
        
        # Initialize Gemini model with tracing and Google Search tool
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=settings.model_temperature,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks(),
            tools=[search_tool]
        )
        
        # Create the prompt template with metadata
        system_message = SystemMessage(
            content=self.system_prompt.format(),
            additional_kwargs={
                "metadata": {
                    "agent_type": "competitor_analyzer",
                    "methodology": "Market Competition Framework"
                }
            }
        )
        
        # Use the analysis prompt as the human message template
        human_template = self.analysis_prompt.template
        
        # Create proper prompt template from the loaded templates
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message.content),
            ("human", human_template)
        ])
        
        # Verify the input variables are correct and log them
        logger.info(f"Prompt expects these variables: {self.prompt.input_variables}")
        logger.debug(f"Checking if analysis template contains expected placeholders:")
        logger.debug(f"- brand_name: {'{{brand_name}}' in human_template}")
        logger.debug(f"- format_instructions: {'{{format_instructions}}' in human_template}")

    async def analyze_brand_name(
        self,
        run_id: str,
        brand_name: str,
        brand_context: Dict[str, Any],
        user_prompt: str = None
    ) -> Dict[str, Any]:
        """Analyze the competitive positioning of a brand name.
        
        Evaluates the brand name against market competitors, analyzing
        differentiation, positioning, and competitive advantage potential.
        
        Args:
            run_id: Unique identifier for this workflow run
            brand_name: The brand name to analyze
            brand_context: Additional brand context information
            user_prompt: Original user prompt for better search context (optional)
            
        Returns:
            Dictionary containing a list of competitor analyses
            
        Raises:
            ValueError: If analysis fails
        """
        try:
            with tracing_v2_enabled():
                # First ensure we have a valid user prompt
                if not user_prompt:
                    user_prompt = "Create a competitive analysis for this brand name in its market."
                    logger.warning(f"No user prompt provided, using default: {user_prompt}")
                
                # Convert brand_context to a formatted string if it's a dictionary
                formatted_context = ""
                if isinstance(brand_context, dict):
                    for key, value in brand_context.items():
                        formatted_context += f"{key}: {value}\n"
                else:
                    formatted_context = str(brand_context)
                
                logger.info(f"Analyzing competitive position for brand name: '{brand_name}'")
                logger.info(f"Using user prompt: '{user_prompt[:100]}...' (truncated)")
                
                try:
                    # Format the prompt with the required variables, including user_prompt
                    formatted_prompt = self.prompt.format_messages(
                        brand_name=brand_name,
                        brand_context=formatted_context,
                        format_instructions=self.format_instructions,
                        user_prompt=user_prompt
                    )
                    
                    # Log the formatted prompt for debugging
                    second_message_content = formatted_prompt[1].content
                    logger.debug(f"Formatted prompt contains brand name: {brand_name in second_message_content}")
                    
                    # Invoke the LLM with the formatted prompt
                    with tracing_v2_enabled():
                        response = await self.llm.ainvoke(formatted_prompt)
                    
                    # Log the response for debugging
                    logger.debug(f"Raw LLM response (excerpt): {response.content[:500]}...")
                    
                    # Process multiple competitors from the response
                    all_competitors = []
                    
                    # Try to parse the response using different approaches
                    try:
                        # Try to find JSON objects in the text
                        import re
                        import json
                        
                        # Look for JSON objects in the response
                        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                        matches = re.findall(json_pattern, response.content)
                        
                        if matches:
                            logger.info(f"Found {len(matches)} potential competitor entries using regex")
                            
                            for match in matches:
                                try:
                                    competitor = json.loads(match)
                                    # Validate that it has the required competitor_name field
                                    if "competitor_name" in competitor:
                                        all_competitors.append(competitor)
                                        logger.info(f"Added competitor: {competitor.get('competitor_name')}")
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON: {match[:50]}...")
                        
                        # If no JSON objects found, try using the output parser
                        if not all_competitors:
                            logger.info("No JSON objects found, trying output parser")
                            competitor = self.output_parser.parse(response.content)
                            if "competitor_name" in competitor:
                                all_competitors.append(competitor)
                                logger.info(f"Added competitor using output parser: {competitor.get('competitor_name')}")
                            else:
                                raise ValueError("Output parser result missing competitor_name field")
                    
                    except Exception as parse_error:
                        logger.error(f"Error parsing competitor response: {str(parse_error)}")
                    
                    # Ensure we have at least one competitor, add defaults if not
                    if not all_competitors:
                        logger.warning("No competitors found in response, using fallback approach")
                        
                        # Extract potential competitor names using regex
                        competitor_pattern = r'(?:Competitor|Company)(?:\s+\d+)?(?:\s*):?\s*([A-Za-z0-9\s\-\.&]+)(?:\n|\.|\-|$)'
                        name_matches = re.findall(competitor_pattern, response.content)
                        
                        if name_matches:
                            logger.info(f"Found {len(name_matches)} potential competitor names using regex")
                            
                            for name in name_matches[:5]:  # Limit to 5 competitors
                                competitor_name = name.strip()
                                if competitor_name and len(competitor_name) > 1:  # Ensure valid name
                                    fallback = {
                                        "competitor_name": competitor_name,
                                        "competitor_naming_style": "Unknown style",
                                        "competitor_keywords": "Unknown",
                                        "competitor_positioning": "Unknown positioning",
                                        "competitor_strengths": "Unknown strengths",
                                        "competitor_weaknesses": "Unknown weaknesses",
                                        "competitor_differentiation_opportunity": "Conduct manual research",
                                        "differentiation_score": 5.0,
                                        "risk_of_confusion": 5.0,
                                        "target_audience_perception": f"Comparison to {competitor_name} unclear",
                                        "competitive_advantage_notes": "Unable to determine",
                                        "trademark_conflict_risk": "Requires manual verification"
                                    }
                                    all_competitors.append(fallback)
                                    logger.info(f"Added fallback competitor: {competitor_name}")
                        
                        # If still no competitors, add generic ones
                        if not all_competitors:
                            industry = brand_context.get('industry_focus', 'general industry')
                            if isinstance(industry, dict):
                                industry = str(industry.get('value', 'general industry')).lower()
                            else:
                                industry = str(industry).lower()
                                
                            logger.warning(f"No competitor names found, adding generic {industry} competitors")
                            
                            # Add 3 generic competitors
                            generic_competitors = [
                                f"Leading {industry} company",
                                f"Major {industry} brand",
                                f"Popular {industry} competitor"
                            ]
                            
                            for name in generic_competitors:
                                fallback = {
                                    "competitor_name": name,
                                    "competitor_naming_style": "Unknown style",
                                    "competitor_keywords": "Unknown",
                                    "competitor_positioning": "Unknown positioning",
                                    "competitor_strengths": "Unknown strengths",
                                    "competitor_weaknesses": "Unknown weaknesses", 
                                    "competitor_differentiation_opportunity": "Conduct manual research",
                                    "differentiation_score": 5.0,
                                    "risk_of_confusion": 5.0,
                                    "target_audience_perception": f"Comparison unclear",
                                    "competitive_advantage_notes": "Unable to determine",
                                    "trademark_conflict_risk": "Requires manual verification"
                                }
                                all_competitors.append(fallback)
                                logger.info(f"Added generic competitor: {name}")
                    
                    # Normalize all competitors (ensure proper data types)
                    for competitor in all_competitors:
                        # Ensure numerical fields are floats in range 0-10
                        try:
                            competitor["differentiation_score"] = float(competitor.get("differentiation_score", 5.0))
                            competitor["differentiation_score"] = max(0.0, min(10.0, competitor["differentiation_score"]))
                        except (ValueError, TypeError):
                            competitor["differentiation_score"] = 5.0
                        
                        # Convert risk_of_confusion to float if it's a string
                        try:
                            risk = competitor.get("risk_of_confusion", "5.0")
                            # If it's a string, try to extract a number
                            if isinstance(risk, str):
                                number_match = re.search(r'(\d+(\.\d+)?)', risk)
                                if number_match:
                                    competitor["risk_of_confusion"] = float(number_match.group(1))
                                else:
                                    competitor["risk_of_confusion"] = 5.0
                            else:
                                competitor["risk_of_confusion"] = float(risk)
                        except (ValueError, TypeError):
                            competitor["risk_of_confusion"] = 5.0
                        
                        # Ensure risk is in range 0-10
                        competitor["risk_of_confusion"] = max(0.0, min(10.0, competitor["risk_of_confusion"]))
                    
                    # Store each competitor separately
                    for competitor in all_competitors:
                        await self._store_analysis(run_id, brand_name, competitor)
                    
                    # Return a consolidated analysis with all competitors
                    return {
                        "brand_name": brand_name,
                        "competitors": all_competitors,
                        "competitor_count": len(all_competitors)
                    }
                except Exception as e:
                    logger.error(f"Error analyzing competitor landscape for {brand_name}: {str(e)}")
                    return {
                        "error": str(e),
                        "brand_name": brand_name
                    }
                
        except Exception as e:
            logger.error(
                "Competitor analysis failed",
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
        """Store competitor analysis results in Supabase.
        
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
            # Map the analysis fields to the Supabase schema fields
            data = {
                "run_id": run_id,
                "brand_name": brand_name,
                "competitor_name": analysis.get("competitor_name", "Unknown Competitor"),
                "competitor_naming_style": analysis.get("competitor_naming_style", ""),
                "competitor_keywords": analysis.get("competitor_keywords", ""),
                "competitor_positioning": analysis.get("competitor_positioning", ""),
                "competitor_strengths": analysis.get("competitor_strengths", ""),
                "competitor_weaknesses": analysis.get("competitor_weaknesses", ""),
                "competitor_differentiation_opportunity": analysis.get("competitor_differentiation_opportunity", ""),
                "differentiation_score": 5.0,  # Default value, will be updated below
                "risk_of_confusion": 5.0,  # Default value, will be updated below
                "target_audience_perception": analysis.get("target_audience_perception", ""),
                "competitive_advantage_notes": analysis.get("competitive_advantage_notes", ""),
                "trademark_conflict_risk": analysis.get("trademark_conflict_risk", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # Ensure text fields are strings
            for key in ["competitor_name", "competitor_naming_style", "competitor_keywords", 
                        "competitor_positioning", "competitor_strengths", "competitor_weaknesses",
                        "competitor_differentiation_opportunity", "target_audience_perception", 
                        "competitive_advantage_notes", "trademark_conflict_risk"]:
                if key in data:
                    data[key] = str(data[key])
            
            # Ensure differentiation_score is a float between 0 and 10
            try:
                diff_score = analysis.get("differentiation_score")
                if diff_score is not None:
                    data["differentiation_score"] = float(diff_score)
                else:
                    data["differentiation_score"] = 5.0
            except (ValueError, TypeError):
                data["differentiation_score"] = 5.0
                
            # Ensure the score is within valid range
            data["differentiation_score"] = max(0.0, min(10.0, data["differentiation_score"]))
            
            # Ensure risk_of_confusion is a float between 0 and 10
            try:
                risk = analysis.get("risk_of_confusion")
                if risk is not None:
                    # If it's a string, try to extract a number
                    if isinstance(risk, str):
                        # Extract first number from string (e.g. "7 out of 10" -> 7.0)
                        import re
                        number_match = re.search(r'(\d+(\.\d+)?)', risk)
                        if number_match:
                            data["risk_of_confusion"] = float(number_match.group(1))
                        else:
                            data["risk_of_confusion"] = 5.0
                    else:
                        data["risk_of_confusion"] = float(risk)
                else:
                    data["risk_of_confusion"] = 5.0
            except (ValueError, TypeError):
                data["risk_of_confusion"] = 5.0
                
            # Ensure the risk is within valid range
            data["risk_of_confusion"] = max(0.0, min(10.0, data["risk_of_confusion"]))
            
            # Log the data being stored
            logger.info(f"Storing competitor analysis for '{data['competitor_name']}' related to brand name '{brand_name}' with run_id '{run_id}'")
            
            # Use the async execute_with_retry method instead of direct table operations
            await self.supabase.execute_with_retry("insert", "competitor_analysis", data)
            logger.info(f"Successfully stored competitor analysis for '{data['competitor_name']}' related to brand name '{brand_name}' with run_id '{run_id}'")
            
        except Exception as e:
            logger.error(
                "Error storing competitor analysis",
                extra={
                    "run_id": run_id,
                    "brand_name": brand_name,
                    "competitor_name": analysis.get("competitor_name", "Unknown"),
                    "error": str(e)
                }
            )
            # Don't raise the exception - we want to continue even if storing one competitor fails
            logger.warning(f"Continuing despite storage error for competitor: {analysis.get('competitor_name', 'Unknown')}") 