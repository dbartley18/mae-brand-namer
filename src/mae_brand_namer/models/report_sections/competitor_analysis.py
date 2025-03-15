from pydantic import BaseModel, Field
from typing import Dict

class CompetitorDetails(BaseModel):
    risk_of_confusion: int = Field(..., description="Risk of confusion with the competitor")
    competitor_strengths: str = Field(..., description="Strengths of the competitor")
    competitor_weaknesses: str = Field(..., description="Weaknesses of the competitor")
    competitor_positioning: str = Field(..., description="Positioning of the competitor in the market")
    trademark_conflict_risk: str = Field(..., description="Risk of trademark conflict")
    target_audience_perception: str = Field(..., description="Perception of the target audience")
    competitor_differentiation_opportunity: str = Field(..., description="Opportunities for differentiation from the competitor")

class CompetitorAnalysis(BaseModel):
    competitor_analysis: Dict[str, Dict[str, CompetitorDetails]] = Field(..., description="Competitor analysis for various brand names") 