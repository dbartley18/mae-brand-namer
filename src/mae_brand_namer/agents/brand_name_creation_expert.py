"""Brand Name Creation Expert for generating strategic brand name candidates."""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import re
import asyncio

from langchain.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from postgrest.exceptions import APIError
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.settings import settings

logger = get_logger(__name__)

class BrandNameCreationExpert:
    """Expert in strategic brand name generation following Alina Wheeler's methodology."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the BrandNameCreationExpert with necessary configurations."""
        # Initialize Supabase client
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
        
        # Load prompts from YAML files
        try:
            prompt_dir = Path(__file__).parent / "prompts" / "brand_name_generator"
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.generation_prompt = load_prompt(str(prompt_dir / "generation.yaml"))
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
        # Initialize Gemini model with tracing
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash"),  # Default to gemini-2.0-flash if env var not set
            temperature=1.0,  # Higher temperature for more creative name generation
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=settings.get_langsmith_callbacks()
        )
        
        # Define output schemas for structured parsing
        self.output_schemas = [
            ResponseSchema(name="brand_name", description="The generated brand name candidate"),
            ResponseSchema(name="naming_category", description="The category or type of name (e.g., descriptive, abstract, evocative)"),
            ResponseSchema(name="brand_personality_alignment", description="How the name aligns with the defined brand personality"),
            ResponseSchema(name="brand_promise_alignment", description="The degree to which the name reflects the brand's promise and value proposition"),
            ResponseSchema(name="target_audience_relevance_score", description="Score from 1-10 indicating relevance to target audience", type="number"),
            ResponseSchema(name="target_audience_relevance_details", description="2-3 bullet points explaining the target audience relevance"),
            ResponseSchema(name="market_differentiation_score", description="Score from 1-10 indicating potential for market differentiation", type="number"),
            ResponseSchema(name="market_differentiation_details", description="2-3 bullet points explaining the market differentiation potential"),
            ResponseSchema(name="memorability_score", description="Score from 1-10 indicating how easily the name can be remembered", type="number"),
            ResponseSchema(name="memorability_score_details", description="2-3 bullet points explaining the memorability factors"),
            ResponseSchema(name="pronounceability_score", description="Score from 1-10 indicating how easily the name can be pronounced", type="number"),
            ResponseSchema(name="pronounceability_score_details", description="2-3 bullet points explaining the pronounceability factors"),
            ResponseSchema(name="visual_branding_potential_score", description="Score from 1-10 indicating potential for visual branding elements", type="number"),
            ResponseSchema(name="visual_branding_potential_details", description="2-3 bullet points explaining the visual branding potential"),
            ResponseSchema(name="name_generation_methodology", description="The structured approach used to generate and refine the brand name"),
            ResponseSchema(name="rank", description="The ranking score assigned to the name based on strategic fit", type="number")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.output_schemas)

    async def generate_brand_names(
            self,
            run_id: str,
            brand_context: Dict[str, Any],
            brand_values: List[str],
            purpose: str,
            key_attributes: List[str],
            num_names_per_category: int = 3,
            categories: List[str] = None
        ) -> List[Dict[str, Any]]:
        """
        Generate brand name candidates based on the brand context, organized by naming categories.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            brand_context (Dict[str, Any]): Brand context information
            brand_values (List[str]): List of brand values
            purpose (str): Brand purpose
            key_attributes (List[str]): Key brand attributes
            num_names_per_category (int, optional): Number of brand names to generate per category. Defaults to 3.
            categories (List[str], optional): List of naming categories to use. Defaults to all four categories.
            
        Returns:
            List[Dict[str, Any]]: List of generated brand names with their evaluations
        """
        generated_names = []
        timestamp = datetime.now().isoformat()
        
        # Default categories if none provided
        if not categories:
            categories = [
                "Descriptive Names",
                "Suggestive Names",
                "Abstract Names",
                "Experiential Names"
            ]
        
        # Validate required inputs
        if not run_id:
            raise ValueError("Missing required parameter: run_id")
        
        # Ensure brand values and key attributes are lists
        if brand_values and not isinstance(brand_values, list):
            brand_values = [str(brand_values)]
        if key_attributes and not isinstance(key_attributes, list):
            key_attributes = [str(key_attributes)]
            
        # Ensure we have a valid purpose
        if not purpose:
            purpose = "Not specified"
            
        logger.info(
            "Starting brand name generation", 
            extra={
                "run_id": run_id,
                "num_names_per_category": num_names_per_category,
                "categories": categories
            }
        )
        
        try:
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format())
            
            with tracing_v2_enabled():
                # Track all generated names to avoid duplicates across categories
                all_generated_names = []
                
                # Normalize a name for duplicate checking - keep this simple
                def normalize_name(name):
                    if not name:
                        return ""
                    return name.lower().strip()
                
                # Iterate through each naming category
                for category in categories:
                    logger.info(
                        f"Generating names for category: {category}",
                        extra={"run_id": run_id}
                    )
                    
                    # Generate specified number of names for this category
                    for i in range(num_names_per_category):
                        try:
                            logger.debug(
                                f"Generating brand name {i+1}/{num_names_per_category} for category {category}",
                                extra={"run_id": run_id}
                            )
                            
                            # Format existing names to avoid duplication - this guides the LLM
                            existing_names = "\n".join([f"- {name['brand_name']}" for name in all_generated_names])
                            if not existing_names:
                                existing_names = "None yet"
                            
                            # Set up generation prompt with category and existing names
                            generation_prompt = self.generation_prompt.format(
                                format_instructions=self.output_parser.get_format_instructions(),
                                brand_context=brand_context,
                                brand_values=brand_values,
                                purpose=purpose,
                                key_attributes=key_attributes,
                                category=category,
                                existing_names=existing_names
                            )
                            
                            # Create human message
                            human_message = HumanMessage(content=generation_prompt)
                            
                            # Simplified retry mechanism - just handle basic failures
                            max_retries = 3
                            retry_count = 0
                            
                            while retry_count < max_retries:
                                try:
                                    # Get response from LLM
                                    response = await self.llm.ainvoke([system_message, human_message])
                                    
                                    # Log the raw response for debugging
                                    logger.debug(
                                        "Raw LLM response",
                                        extra={
                                            "run_id": run_id,
                                            "response_content": response.content[:1000]  # Log first 1000 chars to avoid excessive logging
                                        }
                                    )
                                    
                                    try:
                                        # Parse the structured output
                                        parsed_output = self.output_parser.parse(response.content)
                                    except Exception as parse_error:
                                        logger.warning(
                                            f"Error parsing LLM response: {str(parse_error)}",
                                            extra={"run_id": run_id}
                                        )
                                        # Try a simplified approach to extract just the brand name if parsing fails
                                        if "brand_name:" in response.content or "brand_name\":" in response.content:
                                            # Simple regex to extract brand name
                                            import re
                                            match = re.search(r'["\']?brand_name["\']?\s*[:=]\s*["\']([^"\'\n]+)', response.content)
                                            if match:
                                                brand_name = match.group(1).strip()
                                                logger.info(f"Extracted brand name using fallback method: {brand_name}")
                                                # Create a minimal valid output
                                                parsed_output = {
                                                    "brand_name": brand_name,
                                                    "naming_category": category,
                                                    "brand_personality_alignment": "Generated using fallback parsing",
                                                    "brand_promise_alignment": "Generated using fallback parsing",
                                                    "target_audience_relevance_score": 5,
                                                    "target_audience_relevance_details": "Generated using fallback parsing",
                                                    "market_differentiation_score": 5,
                                                    "market_differentiation_details": "Generated using fallback parsing",
                                                    "memorability_score": 5,
                                                    "memorability_score_details": "Generated using fallback parsing",
                                                    "pronounceability_score": 5,
                                                    "pronounceability_score_details": "Generated using fallback parsing",
                                                    "visual_branding_potential_score": 5,
                                                    "visual_branding_potential_details": "Generated using fallback parsing",
                                                    "name_generation_methodology": "Generated using fallback parsing",
                                                    "rank": 5
                                                }
                                            else:
                                                # Try another approach: look for any likely brand name
                                                # This handles cases where the LLM outputs a brand name but not in the expected format
                                                # Look for capitalized words or quotes that might indicate a brand name
                                                lines = response.content.split('\n')
                                                for line in lines:
                                                    # Skip empty lines
                                                    if not line.strip():
                                                        continue
                                                    
                                                    # Check if line starts with something that looks like a brand name label
                                                    if re.search(r'^(brand\s*name|name|suggested\s*name)[\s:]*(.+)', line, re.IGNORECASE):
                                                        match = re.search(r'^(?:brand\s*name|name|suggested\s*name)[\s:]*["\'"]?([^"\'\n]+)["\'"]?', line, re.IGNORECASE)
                                                        if match:
                                                            brand_name = match.group(1).strip()
                                                            logger.info(f"Extracted brand name using line search: {brand_name}")
                                                            # Create a minimal valid output
                                                            parsed_output = {
                                                                "brand_name": brand_name,
                                                                "naming_category": category,
                                                                "brand_personality_alignment": "Generated using line search parsing",
                                                                "brand_promise_alignment": "Generated using line search parsing",
                                                                "target_audience_relevance_score": 5,
                                                                "target_audience_relevance_details": "Generated using line search parsing",
                                                                "market_differentiation_score": 5,
                                                                "market_differentiation_details": "Generated using line search parsing",
                                                                "memorability_score": 5,
                                                                "memorability_score_details": "Generated using line search parsing",
                                                                "pronounceability_score": 5,
                                                                "pronounceability_score_details": "Generated using line search parsing",
                                                                "visual_branding_potential_score": 5,
                                                                "visual_branding_potential_details": "Generated using line search parsing",
                                                                "name_generation_methodology": "Generated using line search parsing",
                                                                "rank": 5
                                                            }
                                                            break
                                                    
                                                    # Look for quoted content that might be a brand name
                                                    quotes_match = re.search(r'["\'"]([A-Z][a-zA-Z0-9]*(?:\s*[A-Z][a-zA-Z0-9]*)*)["\'"]', line)
                                                    if quotes_match:
                                                        brand_name = quotes_match.group(1).strip()
                                                        logger.info(f"Extracted brand name from quotes: {brand_name}")
                                                        # Create a minimal valid output
                                                        parsed_output = {
                                                            "brand_name": brand_name,
                                                            "naming_category": category,
                                                            "brand_personality_alignment": "Generated using quotes extraction",
                                                            "brand_promise_alignment": "Generated using quotes extraction",
                                                            "target_audience_relevance_score": 5,
                                                            "target_audience_relevance_details": "Generated using quotes extraction",
                                                            "market_differentiation_score": 5,
                                                            "market_differentiation_details": "Generated using quotes extraction",
                                                            "memorability_score": 5,
                                                            "memorability_score_details": "Generated using quotes extraction",
                                                            "pronounceability_score": 5,
                                                            "pronounceability_score_details": "Generated using quotes extraction",
                                                            "visual_branding_potential_score": 5,
                                                            "visual_branding_potential_details": "Generated using quotes extraction",
                                                            "name_generation_methodology": "Generated using quotes extraction",
                                                            "rank": 5
                                                        }
                                                        break
                                                    
                                                    # Look for capitalized words that might be a brand name (last resort)
                                                    caps_match = re.search(r'\b([A-Z][a-zA-Z0-9]*(?:\s*[A-Z][a-zA-Z0-9]*)*)\b', line)
                                                    if caps_match and len(caps_match.group(1)) > 3:  # Avoid short acronyms
                                                        brand_name = caps_match.group(1).strip()
                                                        logger.info(f"Extracted brand name from capitalized text: {brand_name}")
                                                        # Create a minimal valid output
                                                        parsed_output = {
                                                            "brand_name": brand_name,
                                                            "naming_category": category,
                                                            "brand_personality_alignment": "Generated using capitalization extraction",
                                                            "brand_promise_alignment": "Generated using capitalization extraction",
                                                            "target_audience_relevance_score": 5,
                                                            "target_audience_relevance_details": "Generated using capitalization extraction",
                                                            "market_differentiation_score": 5,
                                                            "market_differentiation_details": "Generated using capitalization extraction",
                                                            "memorability_score": 5,
                                                            "memorability_score_details": "Generated using capitalization extraction",
                                                            "pronounceability_score": 5,
                                                            "pronounceability_score_details": "Generated using capitalization extraction",
                                                            "visual_branding_potential_score": 5,
                                                            "visual_branding_potential_details": "Generated using capitalization extraction",
                                                            "name_generation_methodology": "Generated using capitalization extraction",
                                                            "rank": 5
                                                        }
                                                        break
                                                
                                                # If we still don't have a parsed output, retry
                                                if 'parsed_output' not in locals():
                                                    retry_count += 1
                                                    continue
                                        else:
                                            retry_count += 1
                                            continue
                                    
                                    # Ensure required fields exist
                                    if "brand_name" not in parsed_output or not parsed_output["brand_name"]:
                                        logger.warning(
                                            "Missing brand_name in LLM response, retrying",
                                            extra={"run_id": run_id}
                                        )
                                        retry_count += 1
                                        continue
                                    
                                    # Log successful generation
                                    logger.info(
                                        f"Successfully generated brand name: {parsed_output['brand_name']}",
                                        extra={"run_id": run_id}
                                    )
                                    
                                    # Trust that the LLM followed our uniqueness instructions
                                    # The final Supabase check will catch any duplicates as a safety net
                                    break
                                    
                                except Exception as e:
                                    logger.warning(
                                        f"Error during brand name generation attempt: {str(e)}",
                                        extra={
                                            "run_id": run_id,
                                            "error_type": type(e).__name__,
                                            "retry_count": retry_count
                                        }
                                    )
                                    retry_count += 1
                                    
                                    # If we've had multiple failures, reduce the complexity of the prompt
                                    if retry_count >= 2:
                                        logger.info("Using simplified prompt after multiple failures")
                                        # Create a simpler prompt focused just on generating a name
                                        simplified_prompt = f"""
                                        Generate ONE unique brand name for a {brand_context.get('industry_focus', 'business')} company.
                                        
                                        Important facts:
                                        - Industry: {brand_context.get('industry_focus', 'business')}
                                        - Values: {', '.join(brand_values[:3]) if brand_values else 'Innovation, Quality'}
                                        - Purpose: {purpose[:100] + '...' if len(purpose) > 100 else purpose}
                                        
                                        Previously generated names (AVOID): {existing_names}
                                        
                                        Output the name in this format only:
                                        Brand Name: [YOUR BRAND NAME]
                                        
                                        For example: Brand Name: TechNexus
                                        
                                        Do not include any explanations, just the brand name line.
                                        """
                                        human_message = HumanMessage(content=simplified_prompt)
                            
                            # If we couldn't generate a valid name after retries, try a last resort approach
                            if retry_count >= max_retries:
                                logger.warning(
                                    f"Failed to generate valid brand name after {max_retries} attempts, trying last resort approach",
                                    extra={"run_id": run_id}
                                )
                                
                                # Last resort approach: generate a basic name based on the category
                                try:
                                    # Very simple prompt
                                    last_resort_prompt = f"""
                                    Create a single unique brand name for a {brand_context.get('industry_focus', 'business')} company.
                                    
                                    The name should be in this format: Brand Name: [YOUR SUGGESTED BRAND NAME]
                                    
                                    For example: Brand Name: TechNexus
                                    
                                    ONLY output the brand name in the format above. Do not add any explanations.
                                    """
                                    
                                    last_resort_message = HumanMessage(content=last_resort_prompt)
                                    last_resort_response = await self.llm.ainvoke([last_resort_message])
                                    
                                    # Get just the text, no formatting
                                    response_text = last_resort_response.content.strip()
                                    
                                    # Extract using regex - look for the format "Brand Name: XYZ"
                                    brand_name_match = re.search(r'brand\s*name\s*:\s*(.+)', response_text, re.IGNORECASE)
                                    if brand_name_match:
                                        brand_name = brand_name_match.group(1).strip()
                                        # Remove any quotes or formatting
                                        brand_name = re.sub(r'["\'"`]', '', brand_name)
                                    else:
                                        # If specific format wasn't followed, just take first non-empty line
                                        for line in response_text.split('\n'):
                                            if line.strip():
                                                brand_name = line.strip()
                                                # Remove any labels like "Brand Name:" if present
                                                brand_name = re.sub(r'^(?:brand\s*name\s*:?\s*)?', '', brand_name, flags=re.IGNORECASE)
                                                # Remove any quotes or formatting
                                                brand_name = re.sub(r'["\'"`]', '', brand_name)
                                                break
                                        else:
                                            # If somehow we still don't have a name, use the whole response
                                            brand_name = response_text
                                            # Remove any quotes or formatting
                                            brand_name = re.sub(r'["\'"`]', '', brand_name)
                                    
                                    # Truncate if too long
                                    if len(brand_name) > 50:
                                        brand_name = brand_name[:50]
                                    
                                    # Ensure first character is capitalized
                                    if brand_name:
                                        brand_name = brand_name[0].upper() + brand_name[1:]
                                    
                                    if brand_name:
                                        logger.info(f"Generated emergency fallback name: {brand_name}")
                                        # Create a minimal valid output
                                        parsed_output = {
                                            "brand_name": brand_name,
                                            "naming_category": category,
                                            "brand_personality_alignment": "Generated using emergency fallback",
                                            "brand_promise_alignment": "Generated using emergency fallback",
                                            "target_audience_relevance_score": 5,
                                            "target_audience_relevance_details": "Generated using emergency fallback",
                                            "market_differentiation_score": 5,
                                            "market_differentiation_details": "Generated using emergency fallback",
                                            "memorability_score": 5,
                                            "memorability_score_details": "Generated using emergency fallback",
                                            "pronounceability_score": 5,
                                            "pronounceability_score_details": "Generated using emergency fallback",
                                            "visual_branding_potential_score": 5,
                                            "visual_branding_potential_details": "Generated using emergency fallback",
                                            "name_generation_methodology": "Generated using emergency fallback",
                                            "rank": 5
                                        }
                                    else:
                                        # If we still couldn't generate a name, skip this iteration
                                        logger.error(
                                            "Emergency fallback name generation failed, skipping",
                                            extra={"run_id": run_id}
                                        )
                                        continue
                                except Exception as e:
                                    logger.error(
                                        f"Emergency fallback generation failed: {str(e)}, skipping",
                                        extra={"run_id": run_id}
                                    )
                                    continue
                                
                            # Add metadata to output
                            parsed_output.update({
                                "run_id": run_id,
                                "timestamp": timestamp,
                                "category": category  # Add the category to the output
                            })
                            
                            # Store the name data in Supabase - has built-in duplicate check
                            await self._store_in_supabase(run_id, parsed_output)
                            
                            # For the return data, ensure all numeric fields are properly converted to floats
                            return_data = parsed_output.copy()
                            
                            # Ensure all numeric fields are float type for consistent processing downstream
                            numeric_fields = [
                                "target_audience_relevance_score",
                                "market_differentiation_score",
                                "memorability_score",
                                "pronounceability_score", 
                                "visual_branding_potential_score",
                                "rank"
                            ]
                            
                            for field in numeric_fields:
                                try:
                                    if field in return_data:
                                        return_data[field] = float(return_data[field])
                                    else:
                                        return_data[field] = 5.0  # Default value
                                except (ValueError, TypeError):
                                    logger.warning(f"Could not convert {field} to float, using default value")
                                    return_data[field] = 5.0
                            
                            logger.debug(
                                "Generated valid brand name",
                                extra={
                                    "run_id": run_id,
                                    "brand_name": return_data["brand_name"],
                                    "category": category
                                }
                            )
                            
                            # Add to our tracking lists
                            generated_names.append(return_data)
                            all_generated_names.append(parsed_output)
                            
                            logger.info(
                                f"Generated brand name {i+1}/{num_names_per_category} for category {category}",
                                extra={
                                    "run_id": run_id,
                                    "brand_name": return_data["brand_name"]
                                }
                            )
                        except Exception as e:
                            logger.error(
                                f"Error generating brand name {i+1}/{num_names_per_category} for category {category}",
                                extra={
                                    "run_id": run_id,
                                    "category": category,
                                    "error_type": type(e).__name__,
                                    "error_message": str(e)
                                }
                            )
                            # Continue generating other names even if one fails
                            continue
            
            if not generated_names:
                logger.error(
                    "Failed to generate any brand names, creating a default emergency name",
                    extra={"run_id": run_id}
                )
                # Instead of failing, create a default emergency name as a last resort
                try:
                    # Create an emergency brand name based on the industry
                    industry = brand_context.get('industry_focus', 'Tech')
                    default_name = f"Nexus{industry.replace(' ', '')}Solutions"
                    
                    # Create a minimal valid output
                    emergency_name = {
                        "run_id": run_id,
                        "brand_name": default_name,
                        "naming_category": "Emergency Generated Name",
                        "brand_personality_alignment": "Emergency generated name due to generation failures",
                        "brand_promise_alignment": "Emergency generated name due to generation failures",
                        "target_audience_relevance_score": 5.0,
                        "target_audience_relevance_details": "Generated as emergency fallback",
                        "market_differentiation_score": 5.0,
                        "market_differentiation_details": "Generated as emergency fallback",
                        "memorability_score": 5.0,
                        "memorability_score_details": "Generated as emergency fallback",
                        "pronounceability_score": 5.0,
                        "pronounceability_score_details": "Generated as emergency fallback",
                        "visual_branding_potential_score": 5.0,
                        "visual_branding_potential_details": "Generated as emergency fallback",
                        "name_generation_methodology": "Emergency fallback when all other generation methods failed",
                        "timestamp": datetime.now().isoformat(),
                        "rank": 1.0
                    }
                    
                    # Store the emergency name in Supabase
                    await self._store_in_supabase(run_id, emergency_name)
                    
                    # Add to generated names
                    generated_names.append(emergency_name)
                    
                    logger.info(
                        "Created emergency fallback brand name",
                        extra={
                            "run_id": run_id,
                            "brand_name": default_name
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating emergency brand name: {str(e)}",
                        extra={"run_id": run_id}
                    )
                    # Now we have to fail if we couldn't even create an emergency name
                    raise ValueError("Failed to generate any valid brand names, including emergency fallback")
                
            logger.info(
                "Brand name generation completed",
                extra={
                    "run_id": run_id,
                    "count": len(generated_names)
                }
            )
            return generated_names
                
        except APIError as e:
            logger.error(
                "Supabase API error in brand name generation",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            raise
                
        except Exception as e:
            logger.error(
                "Error generating brand names",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise ValueError(f"Failed to generate brand names: {str(e)}")

    async def _store_in_supabase(self, run_id: str, name_data: Dict[str, Any]) -> None:
        """
        Store the generated brand name information in Supabase.
        
        Args:
            run_id (str): Unique identifier for this workflow run
            name_data (Dict[str, Any]): The brand name data to store
            
        Raises:
            PostgrestError: If there's an error with the Supabase query
            APIError: If there's an API-level error with Supabase
            ValueError: If there's an error with data validation or preparation
        """
        # Setup event loop if not available
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Validate required fields
            if not run_id:
                raise ValueError("Missing required field: run_id")
            if not name_data.get("brand_name"):
                raise ValueError("Missing required field: brand_name")
            
            # Check if this brand name already exists in the database for this run
            brand_name = name_data["brand_name"].strip()
            existing_data = await self.supabase.select(
                table="brand_name_generation",
                columns=["id"],
                filters={
                    "run_id": run_id,
                    "brand_name": brand_name
                }
            )
            
            if existing_data and len(existing_data) > 0:
                logger.warning(
                    f"Brand name '{brand_name}' already exists in database for run_id {run_id}, skipping insertion",
                    extra={"run_id": run_id}
                )
                return
                
            # Prepare data for Supabase
            supabase_data = {
                "run_id": run_id,
                "brand_name": brand_name,  # Use the stripped version
            }
            
            # Add optional fields with safe defaults
            supabase_data["naming_category"] = name_data.get("naming_category", name_data.get("category", ""))
            supabase_data["brand_personality_alignment"] = name_data.get("brand_personality_alignment", "")
            supabase_data["brand_promise_alignment"] = name_data.get("brand_promise_alignment", "")
            
            # Handle score fields - ensure they're stored as floats
            score_fields = [
                "target_audience_relevance_score",
                "market_differentiation_score",
                "memorability_score", 
                "pronounceability_score",
                "visual_branding_potential_score",
                "rank"
            ]
            
            for field in score_fields:
                try:
                    if field in name_data and name_data[field] is not None:
                        supabase_data[field] = float(name_data[field])
                    else:
                        supabase_data[field] = 5.0  # Default value
                except (ValueError, TypeError):
                    logger.warning(f"Invalid {field}: {name_data.get(field)}, defaulting to 5.0")
                    supabase_data[field] = 5.0
            
            # Handle details fields - store as text
            details_fields = [
                "target_audience_relevance_details",
                "market_differentiation_details",
                "memorability_score_details",
                "pronounceability_score_details",
                "visual_branding_potential_details"
            ]
            
            for field in details_fields:
                if field in name_data and name_data[field]:
                    supabase_data[field] = str(name_data[field])
                else:
                    supabase_data[field] = "No details provided"
            
            # Name generation methodology is stored as plain text
            supabase_data["name_generation_methodology"] = name_data.get("name_generation_methodology", "")
            
            # Handle timestamp - use ISO format which PostgreSQL can properly interpret
            try:
                if "timestamp" in name_data and name_data["timestamp"]:
                    # Try to parse the existing timestamp
                    if isinstance(name_data["timestamp"], str):
                        # Parse the string to a datetime object, then convert back to ISO format
                        timestamp_dt = datetime.fromisoformat(name_data["timestamp"].replace('Z', '+00:00'))
                        supabase_data["timestamp"] = timestamp_dt.isoformat()
                    elif isinstance(name_data["timestamp"], datetime):
                        # Already a datetime object, just convert to ISO
                        supabase_data["timestamp"] = name_data["timestamp"].isoformat()
                    else:
                        # Unknown format, use current time
                        raise ValueError("Invalid timestamp format")
                else:
                    # No timestamp provided, use current time
                    raise ValueError("No timestamp provided")
            except Exception as e:
                # If any error occurs with timestamp parsing, use current time
                logger.warning(f"Error processing timestamp: {str(e)}. Using current time.")
                supabase_data["timestamp"] = datetime.now().isoformat()
            
            # Define known valid fields for the brand_name_generation table
            valid_fields = [
                "run_id", "brand_name", "naming_category", "brand_personality_alignment",
                "brand_promise_alignment", "target_audience_relevance_score", "market_differentiation_score",
                "memorability_score", "pronounceability_score", "visual_branding_potential_score",
                "target_audience_relevance_details", "market_differentiation_details",
                "memorability_score_details", "pronounceability_score_details", "visual_branding_potential_details",
                "name_generation_methodology", "timestamp", "rank", "category"
            ]
            
            # Filter out any fields that don't exist in the database schema
            filtered_data = {k: v for k, v in supabase_data.items() if k in valid_fields}
            
            # Log the data we're about to insert (for debugging)
            logger.debug(
                "Inserting brand name data into Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": filtered_data["brand_name"],
                    "table": "brand_name_generation",
                    "data": json.dumps(filtered_data)
                }
            )
            
            # Store in Supabase using the singleton client
            await self.supabase.execute_with_retry(
                operation="insert",
                table="brand_name_generation",
                data=filtered_data
            )
            
            logger.info(
                "Brand name stored in Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": filtered_data["brand_name"]
                }
            )
            
        except APIError as e:
            logger.error(
                "Supabase API error in brand name storage",
                extra={
                    "run_id": run_id,
                    "brand_name": name_data.get("brand_name", "unknown"),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None),
                    "data": json.dumps(name_data)
                }
            )
            raise
        except ValueError as e:
            logger.error(
                "Validation error in brand name storage",
                extra={
                    "run_id": run_id,
                    "brand_name": name_data.get("brand_name", "unknown"),
                    "error_type": "ValueError",
                    "error_message": str(e),
                    "data": json.dumps(name_data)
                }
            )
            raise
        except Exception as e:
            logger.error(
                "Error storing brand name in Supabase",
                extra={
                    "run_id": run_id,
                    "brand_name": name_data.get("brand_name", "unknown"),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "data": json.dumps(name_data)
                }
            )
            raise ValueError(f"Failed to store brand name in Supabase: {str(e)}") 