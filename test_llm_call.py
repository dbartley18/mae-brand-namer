import sys
import os
import asyncio
import json
from typing import Dict, List, Any
import logging

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mae_brand_namer.agents.brand_name_evaluator import BrandNameEvaluator
from mae_brand_namer.config.settings import settings
from mae_brand_namer.utils.logging import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger("test_llm_call")

class MockSupabaseClient:
    """Mock Supabase client for testing"""
    def __init__(self):
        self.data = {}

    async def insert(self, *args, **kwargs):
        logger.info(f"Mock insert called with args: {args}, kwargs: {kwargs}")
        return {"data": [{"id": "mock-id"}]}

async def test_context_based_shortlisting():
    """Test the context-based shortlisting method"""
    logger.info("Testing context-based shortlisting method...")

    # Create mock evaluations
    evaluations = [
        {
            "brand_name": "Lumina",
            "strategic_alignment_score": 8,
            "distinctiveness_score": 9,
            "brand_fit_score": 7,
            "memorability_score": 8,
            "pronounceability_score": 9,
            "meaningfulness_score": 7,
            "domain_viability_score": 8,
            "overall_score": 8,
            "evaluation_comments": "Strong brand name with excellent memorability and distinctiveness."
        },
        {
            "brand_name": "Zenith",
            "strategic_alignment_score": 7,
            "distinctiveness_score": 8,
            "brand_fit_score": 9,
            "memorability_score": 7,
            "pronounceability_score": 6,
            "meaningfulness_score": 8,
            "domain_viability_score": 6,
            "overall_score": 7,
            "evaluation_comments": "Good brand name with strong meaning and strategic alignment."
        },
        {
            "brand_name": "Nexus",
            "strategic_alignment_score": 6,
            "distinctiveness_score": 7,
            "brand_fit_score": 8,
            "memorability_score": 9,
            "pronounceability_score": 7,
            "meaningfulness_score": 6,
            "domain_viability_score": 7,
            "overall_score": 7,
            "evaluation_comments": "Solid brand name with excellent memorability scores."
        },
        {
            "brand_name": "Ozone",
            "strategic_alignment_score": 5,
            "distinctiveness_score": 6,
            "brand_fit_score": 5,
            "memorability_score": 6,
            "pronounceability_score": 8,
            "meaningfulness_score": 5,
            "domain_viability_score": 7,
            "overall_score": 6,
            "evaluation_comments": "Average brand name with good pronounceability."
        }
    ]

    # Create a mock brand context
    brand_context = """
    Our brand is a tech startup focused on sustainable energy solutions.
    We aim to revolutionize how businesses manage their energy consumption through AI.
    Our target audience is forward-thinking business leaders who care about sustainability.
    Core values include innovation, environmental responsibility, and transparency.
    Our brand personality is progressive, trustworthy, and innovative.
    """

    # Initialize the evaluator
    mock_supabase = MockSupabaseClient()
    evaluator = BrandNameEvaluator(supabase=mock_supabase)

    try:
        # Direct test of the context-based shortlisting method
        logger.info("Calling _context_based_shortlisting method...")
        updated_evaluations = await evaluator._context_based_shortlisting(evaluations, brand_context)
        
        # Check results
        shortlisted = [e for e in updated_evaluations if e.get("shortlist_status")]
        logger.info(f"Shortlisted {len(shortlisted)} brand names:")
        for brand in shortlisted:
            logger.info(f"- {brand['brand_name']}")
            
        # Print evaluation comments for the first shortlisted name to see rationale
        if shortlisted:
            logger.info(f"First shortlisted name evaluation comments: {shortlisted[0].get('evaluation_comments')}")
            
        logger.info("Test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_context_based_shortlisting()) 