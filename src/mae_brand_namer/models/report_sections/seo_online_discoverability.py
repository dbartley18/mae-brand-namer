from pydantic import BaseModel, Field
from typing import List, Dict

class SEORecommendations(BaseModel):
    recommendations: List[str] = Field(..., description="List of SEO recommendations")

class SEOOnlineDiscoverabilityDetails(BaseModel):
    search_volume: int = Field(..., description="Search volume for the brand name")
    keyword_alignment: str = Field(..., description="Alignment of the brand name with industry keywords")
    keyword_competition: str = Field(..., description="Level of competition for the brand name keywords")
    seo_recommendations: SEORecommendations = Field(..., description="SEO recommendations for improving discoverability")
    seo_viability_score: float = Field(..., description="SEO viability score for the brand name")
    negative_search_results: bool = Field(..., description="Indicates if there are negative search results")
    unusual_spelling_impact: bool = Field(..., description="Indicates if unusual spelling impacts discoverability")
    branded_keyword_potential: str = Field(..., description="Potential for building branded keyword search volume")
    name_length_searchability: str = Field(..., description="Searchability based on the length of the brand name")
    social_media_availability: bool = Field(..., description="Indicates if social media handles are available")
    competitor_domain_strength: str = Field(..., description="Strength of competitor domains")
    exact_match_search_results: str = Field(..., description="Number of exact match search results")
    social_media_discoverability: str = Field(..., description="Ease of discovering the brand on social media")
    negative_keyword_associations: str = Field(..., description="Level of negative keyword associations")
    non_branded_keyword_potential: str = Field(..., description="Potential for non-branded keyword search volume")
    content_marketing_opportunities: str = Field(..., description="Opportunities for content marketing")

class SEOOnlineDiscoverability(BaseModel):
    seo_online_discoverability: Dict[str, SEOOnlineDiscoverabilityDetails] = Field(..., description="SEO online discoverability analysis for various brand names") 