"""German language translation expert."""

from pathlib import Path
from typing import Dict, Any

from ..utils.logging import get_logger
from .base_language_expert import BaseLanguageTranslationExpert

logger = get_logger(__name__)

class GermanTranslationExpert(BaseLanguageTranslationExpert):
    """German language translation expert for analyzing brand names."""
    
    def __init__(self, **kwargs):
        """Initialize the German language translation expert."""
        super().__init__(
            language_code="de",
            language_name="German",
            **kwargs
        )
    
    def _get_language_specific_prompt_additions(self) -> str:
        """Get German-specific additions to the system prompt."""
        return """
        You are a specialized German language expert with deep understanding of 
        German linguistics, dialects, and cultural nuances across Germany, Austria,
        Switzerland, and other German-speaking regions.
        
        When analyzing brand names for German-speaking markets, consider:
        - German pronunciation rules and phonetics 
        - Regional variations in meaning and connotation
        - Cultural sensitivity in all German-speaking regions
        - German naming conventions and expectations
        - Compound word formations and grammatical gender
        """
    
    def _load_prompts(self):
        """Load prompts from language-specific directory."""
        # Target the language-specific prompt directory
        prompt_dir = Path(__file__).parent / "prompts" / "language_experts" / "german"
        
        # Load system prompt
        try:
            from langchain.prompts import load_prompt
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
            
            # Create the prompt template
            from langchain.prompts import ChatPromptTemplate
            
            # Add critical instruction to ensure analysis is in English
            system_content = self.system_prompt.format() + "\n\nCRITICAL: Your analysis MUST be written in English, regardless of the target language being German. Only the 'direct_translation' field and examples within other fields should contain text in German. All explanations and analysis must be in English."
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_content),
                ("human", self.analysis_prompt.template)
            ])
            
            logger.info(f"Loaded German prompt templates with variables: {self.prompt.input_variables}")
        except Exception as e:
            logger.warning(f"Could not load German prompts: {str(e)}")
            # Fall back to the base class implementation
            super()._load_prompts()
    
    def _validate_language_specific_output(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and potentially modify output for German language requirements."""
        # Ensure target_language is set to German
        analysis["target_language"] = "de"
        
        # Ensure any German-specific validations are performed
        if "phonetic_similarity_undesirable" in analysis and isinstance(analysis["phonetic_similarity_undesirable"], str):
            analysis["phonetic_similarity_undesirable"] = analysis["phonetic_similarity_undesirable"].lower() == "true"
        
        if "adaptation_needed" in analysis and isinstance(analysis["adaptation_needed"], str):
            analysis["adaptation_needed"] = analysis["adaptation_needed"].lower() == "true"
        
        # Ensure rank is a float
        if "rank" in analysis:
            try:
                analysis["rank"] = float(analysis["rank"])
            except (ValueError, TypeError):
                analysis["rank"] = 5.0
                
        return analysis 