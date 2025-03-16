from pydantic import BaseModel, Field
from typing import Dict

class LinguisticAnalysisDetails(BaseModel):
    """Model for the linguistic analysis of a brand name."""
    
    brand_name: str = Field(..., description="The brand name being analyzed")
    notes: str = Field(..., description="Notes on the linguistic aspects of the brand name")
    word_class: str = Field(..., description="Word class of the brand name")
    sound_symbolism: str = Field(..., description="Sound symbolism associated with the brand name")
    rhythm_and_meter: str = Field(..., description="Rhythm and meter of the brand name")
    pronunciation_ease: str = Field(..., description="Ease of pronunciation for the brand name")
    euphony_vs_cacophony: str = Field(..., description="Euphony versus cacophony of the brand name")
    inflectional_properties: str = Field(..., description="Inflectional properties of the brand name")
    neologism_appropriateness: str = Field(..., description="Appropriateness of the brand name as a neologism")
    overall_readability_score: str = Field(..., description="Overall readability score of the brand name")
    morphological_transparency: str = Field(..., description="Morphological transparency of the brand name")
    naturalness_in_collocations: str = Field(..., description="Naturalness of the brand name in collocations")
    ease_of_marketing_integration: str = Field(..., description="Ease of marketing integration for the brand name")
    phoneme_frequency_distribution: str = Field(..., description="Phoneme frequency distribution in the brand name")
    semantic_distance_from_competitors: str = Field(..., description="Semantic distance from competitors for the brand name")

class LinguisticAnalysis(BaseModel):
    """Model for linguistic analysis section containing multiple brand analyses."""
    linguistic_analysis: Dict[str, LinguisticAnalysisDetails] = Field(..., description="Linguistic analysis for various brands") 