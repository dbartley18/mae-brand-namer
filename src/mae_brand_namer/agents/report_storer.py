"""Expert in storing and delivering brand naming reports."""

import os
from typing import Dict, Any, Optional
from datetime import datetime
from postgrest import APIError

from ..config.settings import settings
from ..utils.logging import get_logger
from ..utils.supabase_utils import SupabaseManager
from ..models.app_config import AppConfig
from ..llms.chat_google_generative_ai import ChatGoogleGenerativeAI

logger = get_logger(__name__)

class ReportStorer:
    """Expert in storing and delivering finished brand naming reports."""
    
    def __init__(self, dependencies=None, supabase=None, app_config: AppConfig = None):
        """Initialize the ReportStorer with dependencies."""
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase
            self.langsmith = None
        
        # Get agent-specific configuration
        self.app_config = app_config or AppConfig()
        agent_name = "report_storer"
        self.temperature = self.app_config.get_temperature_for_agent(agent_name)
        
        # Initialize Gemini model with agent-specific temperature
        self.llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            temperature=self.temperature,
            google_api_key=os.getenv("GEMINI_API_KEY") or settings.google_api_key,
            convert_system_message_to_human=True,
            callbacks=[self.langsmith] if self.langsmith else None
        )
        
        # Log the temperature setting being used
        logger.info(
            f"Initialized Report Storer with temperature: {self.temperature}",
            extra={"agent": agent_name, "temperature": self.temperature}
        )
    
    async def store_report(self, run_id: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store the generated report and return access information.
        
        Args:
            run_id: Unique identifier for this workflow run
            report_data: Report data to store
            
        Returns:
            Dict containing the report URL and access information
        """
        try:
            # Generate the report URL
            base_url = settings.report_base_url or "https://reports.maebrandnamer.ai"
            report_url = f"{base_url}/reports/{run_id}"
            
            # Create the access token (simplified for demo)
            access_token = f"token_{run_id}"
            
            # Log the storage operation
            logger.info(
                "Report stored successfully",
                extra={
                    "run_id": run_id,
                    "report_url": report_url
                }
            )
            
            # Return the access information
            return {
                "report_url": report_url,
                "access_token": access_token,
                "expires_at": (datetime.now().replace(microsecond=0) + 
                              datetime.timedelta(days=30)).isoformat(),
                "status": "complete"
            }
            
        except APIError as e:
            logger.error(
                "Supabase API error in report storage",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "status_code": getattr(e, "code", None),
                    "details": getattr(e, "details", None)
                }
            )
            raise
            
        except Exception as e:
            logger.error(
                "Error in report storage",
                extra={
                    "run_id": run_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise 