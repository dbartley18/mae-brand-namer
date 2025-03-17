from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class BrandName(BaseModel):
    """Model representing a single brand name option"""
    brand_name: str = Field(..., description="The brand name")
    naming_category: Optional[str] = Field(None, description="The category this name belongs to")
    brand_promise_alignment: Optional[str] = Field(None, description="Description of how the name aligns with the brand promise")
    brand_personality_alignment: Optional[str] = Field(None, description="Description of how the name aligns with the brand personality")
    name_generation_methodology: Optional[str] = Field(None, description="The methodology used to generate this name")
    rationale: Optional[str] = Field(None, description="The rationale behind this name")
    memorability_score_details: Optional[str] = Field(None, description="Details about the name's memorability")
    pronounceability_score_details: Optional[str] = Field(None, description="Details about the name's pronounceability")
    visual_branding_potential_details: Optional[str] = Field(None, description="Details about the name's visual branding potential")
    audience_relevance: Optional[str] = Field(None, description="Description of the name's relevance to the target audience")
    market_differentiation: Optional[str] = Field(None, description="Description of how the name differentiates in the market")
    trademark_status: Optional[str] = Field(None, description="Initial assessment of trademark availability")
    cultural_considerations: Optional[str] = Field(None, description="Any cultural considerations for this name")
    target_audience_relevance_details: Optional[str] = Field(None, description="Details about the name's relevance to the target audience")
    market_differentiation_details: Optional[str] = Field(None, description="Details about how the name differentiates in the market")
    
    class Config:
        extra = "ignore"

class NameGenerationSection(BaseModel):
    """Model for the brand name generation section of the report."""
    brand_name_generation: Dict[str, List[BrandName]] = Field(..., description="Categories of brand names with lists of brand names in each category") 