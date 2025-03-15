from pydantic import BaseModel, Field
from typing import List

class BrandContext(BaseModel):
    """Model for the brand context."""
    
    brand_values: List[str] = Field(..., description="Core values of the brand")
    brand_mission: str = Field(..., description="Mission statement of the brand")
    brand_promise: str = Field(..., description="Promise made by the brand to its customers")
    brand_purpose: str = Field(..., description="Purpose of the brand")
    customer_needs: List[str] = Field(..., description="Needs of the brand's customers")
    industry_focus: str = Field(..., description="Industry focus of the brand")
    industry_trends: List[str] = Field(..., description="Current trends in the industry")
    target_audience: str = Field(..., description="Target audience of the brand")
    brand_personality: List[str] = Field(..., description="Personality traits of the brand")
    market_positioning: str = Field(..., description="Market positioning of the brand")
    brand_tone_of_voice: str = Field(..., description="Tone of voice used by the brand")
    brand_identity_brief: str = Field(..., description="Brief description of the brand's identity")
    competitive_landscape: str = Field(..., description="Competitive landscape in which the brand operates") 