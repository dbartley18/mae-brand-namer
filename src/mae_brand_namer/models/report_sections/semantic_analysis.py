from pydantic import BaseModel, Field
from typing import Optional, Dict

class SemanticAnalysisDetails(BaseModel):
    """Model for the semantic analysis of a brand name."""
    
    brand_name: str = Field(..., description="The brand name being analyzed")
    etymology: str = Field(..., description="Etymology of the brand name")
    sound_symbolism: str = Field(..., description="Sound symbolism associated with the brand name")
    brand_personality: str = Field(..., description="Personality traits associated with the brand name")
    emotional_valence: str = Field(..., description="Emotional valence of the brand name")
    denotative_meaning: str = Field(..., description="Denotative meaning of the brand name")
    figurative_language: str = Field(..., description="Figurative language associated with the brand name")
    phoneme_combinations: str = Field(..., description="Phoneme combinations in the brand name")
    sensory_associations: str = Field(..., description="Sensory associations with the brand name")
    word_length_syllables: int = Field(..., description="Number of syllables in the brand name")
    alliteration_assonance: bool = Field(..., description="Whether the brand name uses alliteration or assonance")
    compounding_derivation: str = Field(..., description="Compounding or derivation of the brand name")
    semantic_trademark_risk: str = Field(..., description="Trademark risk associated with the brand name")

class SemanticAnalysis(BaseModel):
    """Model for semantic analysis section containing multiple brand analyses."""
    semantic_analysis: Dict[str, SemanticAnalysisDetails] = Field(..., description="Semantic analysis for various brands") 