"""Expert in storing and delivering brand naming reports."""

import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from postgrest import APIError

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

class ReportStorer:
    """Expert in storing and delivering finished brand naming reports."""
    
    async def store_report(self, run_id: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store the generated report and return access information.
        
        Args:
            run_id: Unique identifier for this workflow run
            report_data: Report data to store
            
        Returns:
            Dict containing the updated state with report URL and access information
        """
        try:
            # Generate the report URL
            base_url = settings.report_base_url or "https://reports.maebrandnamer.ai"
            report_url = f"{base_url}/reports/{run_id}"
            
            # Create the access token (simplified for demo)
            access_token = f"token_{run_id}"
            
            # Current timestamp
            now = datetime.now()
            
            # Log the storage operation
            logger.info(
                "Report stored successfully",
                extra={
                    "run_id": run_id,
                    "report_url": report_url
                }
            )
            
            # Ensure we're returning all the fields needed by BrandNameGenerationState
            # Create a result that preserves all original report data
            result = {
                **report_data,  # Keep all existing report data
                "report_url": report_url,
                "status": "complete"
            }
            
            # Only set these fields if they don't already exist
            if "created_at" not in result:
                result["created_at"] = now.isoformat()
            
            if "last_updated" not in result:
                result["last_updated"] = now.isoformat()
                
            # Ensure version is always present
            if "version" not in result:
                result["version"] = 1
                
            # Return the complete state data
            return result
            
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