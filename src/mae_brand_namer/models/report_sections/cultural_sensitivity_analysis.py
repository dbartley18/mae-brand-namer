from pydantic import BaseModel, Field
from typing import Dict

class BrandAnalysis(BaseModel):
    brand_name: str = Field(..., description="The brand name being analyzed")
    notes: str = Field(..., description="General notes on the brand name's cultural sensitivity")
    symbolic_meanings: str = Field(..., description="Symbolic meanings associated with the brand name")
    historical_meaning: str = Field(..., description="Historical meanings and associations of the brand name")
    overall_risk_rating: str = Field(..., description="Overall risk rating for cultural sensitivity")
    regional_variations: str = Field(..., description="Regional variations in the perception of the brand name")
    cultural_connotations: str = Field(..., description="Cultural connotations of the brand name")
    current_event_relevance: str = Field(..., description="Relevance of the brand name to current events")
    religious_sensitivities: str = Field(..., description="Religious sensitivities associated with the brand name")
    social_political_taboos: str = Field(..., description="Social and political taboos related to the brand name")
    age_related_connotations: str = Field(..., description="Age-related connotations of the brand name")
    alignment_with_cultural_values: str = Field(..., description="Alignment of the brand name with cultural values")

class CulturalSensitivityAnalysis(BaseModel):
    cultural_sensitivity_analysis: Dict[str, BrandAnalysis] = Field(..., description="Analysis of cultural sensitivity for various brand names") 