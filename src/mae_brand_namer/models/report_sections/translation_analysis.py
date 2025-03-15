from pydantic import BaseModel, Field
from typing import Dict

class LanguageAnalysis(BaseModel):
    notes: str = Field(..., description="Notes on the translation and cultural considerations")
    semantic_shift: str = Field(..., description="Description of any semantic shift in translation")
    target_language: str = Field(..., description="The target language for translation")
    adaptation_needed: bool = Field(..., description="Indicates if adaptation is needed")
    direct_translation: str = Field(..., description="Direct translation of the brand name")
    phonetic_retention: str = Field(..., description="Phonetic retention score and comments")
    proposed_adaptation: str = Field(..., description="Proposed adaptation for better cultural fit")
    cultural_acceptability: str = Field(..., description="Cultural acceptability of the translation")
    brand_essence_preserved: str = Field(..., description="How well the brand essence is preserved")
    pronunciation_difficulty: str = Field(..., description="Difficulty of pronunciation in the target language")
    global_consistency_vs_localization: str = Field(..., description="Balance between global consistency and localization")

class TranslationAnalysis(BaseModel):
    translation_analysis: Dict[str, Dict[str, LanguageAnalysis]] = Field(..., description="Translation analysis for various brand names across different languages") 