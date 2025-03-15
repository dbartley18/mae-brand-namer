from pydantic import BaseModel, Field
from typing import Optional

class SemanticAnalysis(BaseModel):
    """Model for the semantic analysis of a brand name."""
    
    etymology: str = Field(..., description="Etymology of the brand name")
    sound_symbolism: Optional[str] = Field(None, description="Sound symbolism associated with the brand name")
    brand_personality: str = Field(..., description="Personality traits associated with the brand name")
    emotional_valence: str = Field(..., description="Emotional valence of the brand name")
    denotative_meaning: str = Field(..., description="Denotative meaning of the brand name")
    figurative_language: Optional[str] = Field(None, description="Figurative language associated with the brand name")
    phoneme_combinations: Optional[str] = Field(None, description="Phoneme combinations in the brand name")
    sensory_associations: Optional[str] = Field(None, description="Sensory associations with the brand name")
    word_length_syllables: Optional[int] = Field(None, description="Number of syllables in the brand name")
    alliteration_assonance: Optional[bool] = Field(None, description="Whether the brand name uses alliteration or assonance")
    compounding_derivation: Optional[str] = Field(None, description="Compounding or derivation of the brand name")
    semantic_trademark_risk: str = Field(..., description="Trademark risk associated with the brand name") 