from pydantic import BaseModel, Field
from typing import Dict, List, Union, Any, Optional

class SurveyDetails(BaseModel):
    brand_name: str = Field(..., description="The brand name being evaluated")
    company_name: str = Field(..., description="The name of the company the persona represents")
    emotional_association: str = Field(..., description="Emotional associations with the brand name")
    personality_fit_score: int = Field(..., description="Score indicating personality fit")
    raw_qualitative_feedback: Dict[str, Any] = Field(..., description="Detailed qualitative feedback on various aspects")
    final_survey_recommendation: str = Field(..., description="Final recommendation based on survey results")
    qualitative_feedback_summary: str = Field(..., description="Summary of qualitative feedback")
    competitor_benchmarking_score: int = Field(..., description="Score benchmarking against competitors")
    brand_promise_perception_score: int = Field(..., description="Score for brand promise perception")
    simulated_market_adoption_score: int = Field(..., description="Score for simulated market adoption")
    competitive_differentiation_score: int = Field(..., description="Score for competitive differentiation")
    
    # Industry and company fields
    industry: Optional[str] = Field(None, description="Industry the brand operates in")
    company_size_employees: Optional[str] = Field(None, description="Size of the company (text)")
    company_revenue: Optional[float] = Field(None, description="Annual revenue of the company")
    
    # Professional role fields
    job_title: Optional[str] = Field(None, description="Job title of target persona")
    seniority: Optional[str] = Field(None, description="Seniority level")
    years_of_experience: Optional[float] = Field(None, description="Years of professional experience")
    department: Optional[str] = Field(None, description="Department within organization")
    education_level: Optional[str] = Field(None, description="Highest education level attained")
    
    # Decision making fields
    decision_making_style: Optional[str] = Field(None, description="Approach to decision making")
    information_sources: Optional[str] = Field(None, description="Primary sources of information")
    pain_points: Optional[str] = Field(None, description="Primary pain points related to products/services")
    attitude_towards_risk: Optional[str] = Field(None, description="Risk tolerance level")
    decision_maker: Optional[bool] = Field(None, description="Whether they are a decision maker")
    budget_authority: Optional[str] = Field(None, description="Level of budget authority")
    
    # Behavioral fields
    purchasing_behavior: Optional[str] = Field(None, description="Habits related to making purchases")
    online_behavior: Optional[str] = Field(None, description="Online activity patterns")
    interaction_with_brand: Optional[str] = Field(None, description="How they typically interact with brands")
    professional_associations: Optional[str] = Field(None, description="Professional groups or associations")
    influence_within_company: Optional[str] = Field(None, description="Level of influence in decision making")
    content_consumption_habits: Optional[str] = Field(None, description="How they consume content")
    vendor_relationship_preferences: Optional[str] = Field(None, description="Preferred vendor relationship style")
    reports_to: Optional[str] = Field(None, description="Who they report to in the organization")
    buying_group_structure: Optional[str] = Field(None, description="Structure of their buying group")
    social_media_usage: Optional[str] = Field(None, description="How they use social media")
    frustrations_annoyances: Optional[str] = Field(None, description="What frustrates them")
    motivations: Optional[str] = Field(None, description="What motivates them")
    current_brand_relationships: Optional[Dict[str, Any]] = Field(None, description="Relationships with other brands")
    success_metrics_product_service: Optional[str] = Field(None, description="How they measure success")
    channel_preferences_brand_interaction: Optional[str] = Field(None, description="Preferred channels for brand interaction")
    
    # Other demographic fields
    generation_age_range: Optional[str] = Field(None, description="Their generation or age range")
    persona_archetype_type: Optional[str] = Field(None, description="Type of archetype this persona represents")

class SurveySimulation(BaseModel):
    survey_simulation: List[SurveyDetails] = Field(..., description="List of survey simulation results for various brands") 