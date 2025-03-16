from pydantic import BaseModel, Field
from typing import Dict, List

class EvaluationDetails(BaseModel):
    brand_name: str = Field(..., description="The brand name being evaluated")
    overall_score: int = Field(..., description="Overall score of the brand name evaluation")
    shortlist_status: bool = Field(..., description="Indicates if the brand name is shortlisted")
    evaluation_comments: str = Field(..., description="Comments and analysis of the brand name")

class EvaluationLists(BaseModel):
    shortlisted_names: List[EvaluationDetails] = Field(..., description="List of shortlisted brand names with their evaluation details")
    other_names: List[EvaluationDetails] = Field(..., description="List of non-shortlisted brand names with their evaluation details")

class BrandNameEvaluation(BaseModel):
    brand_name_evaluation: EvaluationLists = Field(..., description="Evaluation details organized into shortlisted and other names") 