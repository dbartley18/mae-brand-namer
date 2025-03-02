"""Factory for creating language translation experts."""

from typing import Dict, Optional, Type, Union, List

from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..config.dependencies import Dependencies
from .base_language_expert import BaseLanguageTranslationExpert
from .spanish_translation_expert import SpanishTranslationExpert
from .french_translation_expert import FrenchTranslationExpert
from .german_translation_expert import GermanTranslationExpert
from .chinese_translation_expert import ChineseTranslationExpert
from .japanese_translation_expert import JapaneseTranslationExpert
from .arabic_translation_expert import ArabicTranslationExpert

logger = get_logger(__name__)

# Registry of all available language experts
LANGUAGE_EXPERTS: Dict[str, Type[BaseLanguageTranslationExpert]] = {
    "spanish": SpanishTranslationExpert,
    "es": SpanishTranslationExpert,
    "french": FrenchTranslationExpert,
    "fr": FrenchTranslationExpert,
    "german": GermanTranslationExpert,
    "de": GermanTranslationExpert,
    "chinese": ChineseTranslationExpert,
    "zh": ChineseTranslationExpert,
    "japanese": JapaneseTranslationExpert,
    "ja": JapaneseTranslationExpert,
    "arabic": ArabicTranslationExpert,
    "ar": ArabicTranslationExpert,
}

# Display-friendly language names
LANGUAGE_DISPLAY_NAMES = {
    "spanish": "Spanish",
    "es": "Spanish",
    "french": "French",
    "fr": "French",
    "german": "German", 
    "de": "German",
    "chinese": "Chinese",
    "zh": "Chinese",
    "japanese": "Japanese",
    "ja": "Japanese",
    "arabic": "Arabic",
    "ar": "Arabic",
}

def get_available_languages() -> List[str]:
    """Return a list of language codes for all available language experts.
    
    Returns:
        List of available language codes (ISO 639-1 format)
    """
    return ["es", "fr", "de", "zh", "ja", "ar"]

def get_language_expert(
    language_code: str,
    dependencies: Optional[Dependencies] = None,
    supabase: Optional[SupabaseManager] = None
) -> Optional[BaseLanguageTranslationExpert]:
    """Get a language translation expert for the specified language.
    
    Args:
        language_code: ISO 639-1 language code or language name
        dependencies: Optional dependencies for initializing the expert
        supabase: Optional Supabase manager for database access
        
    Returns:
        An instance of the appropriate language translation expert, or None if not found
    """
    language_code = language_code.lower().strip()
    
    if language_code not in LANGUAGE_EXPERTS:
        logger.warning(f"No language expert available for language code: {language_code}")
        return None
    
    expert_class = LANGUAGE_EXPERTS[language_code]
    return expert_class(dependencies=dependencies, supabase=supabase)

def get_all_language_experts(
    dependencies: Optional[Dependencies] = None,
    supabase: Optional[SupabaseManager] = None
) -> Dict[str, BaseLanguageTranslationExpert]:
    """Get instances of all available language translation experts.
    
    Args:
        dependencies: Optional dependencies for initializing the experts
        supabase: Optional Supabase manager for database access
        
    Returns:
        Dictionary mapping language codes to expert instances
    """
    experts = {}
    for language_code in get_available_languages():
        experts[language_code] = get_language_expert(
            language_code,
            dependencies=dependencies,
            supabase=supabase
        )
    
    return experts

def get_language_display_name(language_code: str) -> str:
    """Get a user-friendly display name for the specified language code.
    
    Args:
        language_code: ISO 639-1 language code or language name
        
    Returns:
        User-friendly display name, or the original code if not found
    """
    language_code = language_code.lower().strip()
    return LANGUAGE_DISPLAY_NAMES.get(language_code, language_code) 