from pydantic import BaseModel, Field
from typing import Optional

class BrandName(BaseModel):
    """Model for an individual brand name in the name generation section."""
    
    # Required fields from the example
    brand_name: str = Field(..., description="The generated brand name")
    naming_category: Optional[str] = Field(None, description="Category of the name")
    brand_personality_alignment: str = Field(..., description="How the name aligns with the brand personality")
    brand_promise_alignment: str = Field(..., description="How the name aligns with the brand promise")
    name_generation_methodology: str = Field(..., description="The methodology used to generate the name")
    memorability_score_details: str = Field(..., description="Assessment of how memorable the name is")
    pronounceability_score_details: str = Field(..., description="Assessment of how pronounceable the name is")
    visual_branding_potential_details: str = Field(..., description="Assessment of the visual branding potential")
    target_audience_relevance_details: str = Field(..., description="How relevant the name is to the target audience")
    market_differentiation_details: str = Field(..., description="How the name differentiates from competitors")
    
    # Optional fields
    rationale: Optional[str] = Field(None, description="Rationale for the name")
    
    class Config:
        # Allow extra fields we don't explicitly model
        extra = "ignore" 