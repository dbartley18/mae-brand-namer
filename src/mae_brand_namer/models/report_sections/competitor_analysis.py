from pydantic import BaseModel, Field
from typing import Dict, List

class CompetitorDetails(BaseModel):
    competitor_name: str = Field(..., description="Name of the competitor")
    risk_of_confusion: int = Field(..., description="Risk of confusion with the competitor")
    competitor_strengths: str = Field(..., description="Strengths of the competitor")
    competitor_weaknesses: str = Field(..., description="Weaknesses of the competitor")
    competitor_positioning: str = Field(..., description="Positioning of the competitor in the market")
    trademark_conflict_risk: str = Field(..., description="Risk of trademark conflict")
    target_audience_perception: str = Field(..., description="Perception of the target audience")
    competitor_differentiation_opportunity: str = Field(..., description="Opportunities for differentiation from the competitor")

class BrandCompetitors(BaseModel):
    brand_name: str = Field(..., description="The brand name being analyzed")
    competitors: List[CompetitorDetails] = Field(..., description="List of competitors for this brand")

class CompetitorAnalysis(BaseModel):
    competitor_analysis: List[BrandCompetitors] = Field(..., description="Competitor analysis for various brand names") 