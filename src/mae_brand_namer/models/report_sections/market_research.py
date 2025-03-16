from pydantic import BaseModel, Field
from typing import List, Dict

class MarketResearchDetails(BaseModel):
    brand_name: str = Field(..., description="The brand name being analyzed")
    market_size: str = Field(..., description="Size of the market")
    industry_name: str = Field(..., description="Name of the industry")
    emerging_trends: str = Field(..., description="Emerging trends in the market")
    key_competitors: List[str] = Field(..., description="List of key competitors")
    potential_risks: str = Field(..., description="Potential risks in the market")
    recommendations: str = Field(..., description="Recommendations for market strategy")
    market_viability: str = Field(..., description="Viability of the market")
    market_growth_rate: str = Field(..., description="Growth rate of the market")
    market_opportunity: str = Field(..., description="Opportunities in the market")
    target_audience_fit: str = Field(..., description="Fit of the target audience with the brand")
    competitive_analysis: str = Field(..., description="Analysis of the competitive landscape")
    customer_pain_points: List[str] = Field(..., description="List of customer pain points")
    market_entry_barriers: str = Field(..., description="Barriers to entering the market")

class MarketResearch(BaseModel):
    market_research: Dict[str, MarketResearchDetails] = Field(..., description="Market research details for various brands, keyed by brand name") 