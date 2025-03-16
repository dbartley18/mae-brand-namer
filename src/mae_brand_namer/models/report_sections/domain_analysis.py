from pydantic import BaseModel, Field
from typing import List, Dict

class DomainDetails(BaseModel):
    brand_name: str = Field(..., description="The brand name being analyzed")
    notes: str = Field(..., description="Notes on the domain availability and strategy")
    acquisition_cost: str = Field(..., description="Cost associated with acquiring the domain")
    alternative_tlds: List[str] = Field(..., description="List of alternative top-level domains")
    domain_exact_match: bool = Field(..., description="Indicates if the exact match domain is available")
    hyphens_numbers_present: bool = Field(..., description="Indicates if hyphens or numbers are present in the domain")
    brand_name_clarity_in_url: str = Field(..., description="Clarity of the brand name in the URL")
    domain_length_readability: str = Field(..., description="Readability of the domain length")
    social_media_availability: List[str] = Field(..., description="Available social media handles")
    scalability_future_proofing: str = Field(..., description="Scalability and future-proofing of the domain")
    misspellings_variations_available: bool = Field(..., description="Indicates if misspellings or variations are available")

class DomainAnalysis(BaseModel):
    domain_analysis: Dict[str, DomainDetails] = Field(..., description="Domain analysis for various brand names") 