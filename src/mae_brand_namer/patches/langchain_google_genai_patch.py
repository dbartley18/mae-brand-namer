"""
Monkey patch for langchain_google_genai to fix temperature range validation.

The current version of langchain_google_genai limits temperature to [0.0, 1.0]
but Google Gemini actually supports temperatures up to 2.0.
"""

import logging
import sys
from typing import Any, Dict, Optional, Self

from langchain_google_genai import ChatGoogleGenerativeAI as OriginalChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Create a custom class that extends the original
class CustomChatGoogleGenerativeAI(OriginalChatGoogleGenerativeAI):
    """Custom ChatGoogleGenerativeAI class that allows temperatures up to 2.0."""
    
    def validate_environment(self) -> Self:
        """Override validation to allow temperatures up to 2.0."""
        # Skip the temperature validation in the original validate_environment
        if self.temperature is not None and not 0 <= self.temperature <= 2.0:
            logger.warning(
                f"Temperature {self.temperature} is outside the recommended range [0.0, 2.0]. "
                "Clamping to range."
            )
            self.temperature = min(max(0.0, self.temperature), 2.0)
        
        # Call the original validation method but bypass the temperature check
        original_temp = self.temperature
        self.temperature = 0.7  # Set a safe value to pass validation
        super().validate_environment()
        self.temperature = original_temp  # Restore the original temperature
        
        return self

def apply_patches():
    """Apply all monkey patches."""
    # Replace the original ChatGoogleGenerativeAI with our custom version
    import langchain_google_genai
    
    # Replace the classes in the module
    langchain_google_genai.ChatGoogleGenerativeAI = CustomChatGoogleGenerativeAI
    sys.modules['langchain_google_genai'].ChatGoogleGenerativeAI = CustomChatGoogleGenerativeAI
    
    logger.info("Replaced ChatGoogleGenerativeAI with custom version allowing temperatures up to 2.0") 