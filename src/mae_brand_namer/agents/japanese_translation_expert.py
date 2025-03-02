"""Japanese language translation expert."""

from pathlib import Path
from typing import Dict, Any

from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.dependencies import Dependencies
from .base_language_expert import BaseLanguageTranslationExpert

logger = get_logger(__name__)

class JapaneseTranslationExpert(BaseLanguageTranslationExpert):
    """Japanese language translation expert for analyzing brand names."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None, **kwargs):
        """Initialize the Japanese language translation expert."""
        super().__init__(
            dependencies=dependencies,
            supabase=supabase,
            language_code="ja",
            language_name="Japanese",
            **kwargs
        )
    
    def _get_language_specific_prompt_additions(self) -> str:
        """Get Japanese-specific additions to the system prompt."""
        return """
        You are a specialized Japanese language expert with deep understanding of 
        Japanese linguistics, writing systems, and cultural nuances across Japan.
        
        When analyzing brand names for Japanese-speaking markets, consider:
        - Katakana transliteration of foreign words
        - Pronunciation challenges for Japanese speakers
        - Writing system implications (Kanji, Hiragana, Katakana, Romaji)
        - Cultural sensitivity and potential unwanted associations
        - Pitch accent and phonetic rhythm
        - Word length and syllable count
        """
    
    def _load_prompts(self):
        """Load prompts from language-specific directory."""
        # Target the language-specific prompt directory
        prompt_dir = Path(__file__).parent / "prompts" / "language_experts" / "japanese"
        
        # Load system prompt
        try:
            from langchain.prompts import load_prompt
            self.system_prompt = load_prompt(str(prompt_dir / "system.yaml"))
            self.analysis_prompt = load_prompt(str(prompt_dir / "analysis.yaml"))
            
            # Create the prompt template
            from langchain.prompts import ChatPromptTemplate
            
            # Add critical instruction to ensure analysis is in English
            system_content = self.system_prompt.format() + "\n\nCRITICAL: Your analysis MUST be written in English, regardless of the target language being Japanese. Only the 'direct_translation' field and examples within other fields should contain text in Japanese. All explanations and analysis must be in English."
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_content),
                ("human", self.analysis_prompt.template)
            ])
            
            logger.info(f"Loaded Japanese prompt templates with variables: {self.prompt.input_variables}")
        except Exception as e:
            logger.warning(f"Could not load Japanese prompts: {str(e)}")
            # Fall back to the base class implementation
            super()._load_prompts()
    
    def _validate_language_specific_output(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and potentially modify output for Japanese language requirements."""
        # Ensure target_language is set to Japanese
        analysis["target_language"] = "ja"
        
        # Ensure any Japanese-specific validations are performed
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