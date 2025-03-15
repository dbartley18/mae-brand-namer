from pydantic import BaseModel, Field
from typing import List, Optional
from .brand_name import BrandName

class NameCategory(BaseModel):
    """Model for a category of brand names."""
    
    category_name: str = Field(..., description="Name of the naming category (e.g., Symbolic, Descriptive)")
    category_description: Optional[str] = Field(None, description="Description of this naming approach/category")
    names: List[BrandName] = Field(..., description="List of brand names in this category") 