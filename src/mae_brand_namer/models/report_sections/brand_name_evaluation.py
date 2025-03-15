from pydantic import BaseModel, Field
from typing import Dict

class EvaluationDetails(BaseModel):
    overall_score: int = Field(..., description="Overall score of the brand name evaluation")
    shortlist_status: bool = Field(..., description="Indicates if the brand name is shortlisted")
    evaluation_comments: str = Field(..., description="Comments and analysis of the brand name")

class BrandNameEvaluation(BaseModel):
    brand_name_evaluation: Dict[str, EvaluationDetails] = Field(..., description="Evaluation details for various brand names") 