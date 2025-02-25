"""Report storage agent for persisting brand naming reports in Supabase."""

from typing import Dict, Any, Optional
from datetime import datetime
import json

from langchain.callbacks import tracing_enabled
from supabase import APIError, PostgrestError

from ..utils.logging import get_logger
from ..utils.supabase_utils import supabase
from ..config.settings import settings

logger = get_logger(__name__)

class ReportStorer:
    """Agent responsible for storing brand naming reports in Supabase."""

    async def store_report(self, run_id: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store the compiled report in Supabase.

        Args:
            run_id (str): Unique identifier for the workflow run
            report_data (Dict[str, Any]): Compiled report data to store

        Returns:
            Dict[str, Any]: Report metadata including storage details

        Raises:
            APIError: If there's an error communicating with Supabase
            ValueError: If the report data is invalid
        """
        try:
            with tracing_enabled():
                # Validate report data
                if not isinstance(report_data, dict):
                    raise ValueError("Report data must be a dictionary")
                
                if not report_data.get("run_id") == run_id:
                    raise ValueError("Report run_id mismatch")

                # Generate report metadata
                metadata = {
                    "run_id": run_id,
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "format": "json",
                    "file_size_kb": len(json.dumps(report_data).encode()) // 1024
                }

                try:
                    # Store report data in Supabase
                    response = await supabase.execute_with_retry(
                        operation="insert",
                        table="report_compilation",
                        data={
                            "run_id": run_id,
                            "report_data": report_data,
                            "metadata": metadata,
                            "created_at": metadata["created_at"]
                        }
                    )

                    # Generate report URL
                    report_url = f"{settings.supabase_url}/storage/v1/object/report/{run_id}"
                    metadata["report_url"] = report_url

                    logger.info(
                        "Report stored successfully",
                        run_id=run_id,
                        report_url=report_url
                    )

                    return metadata

                except (APIError, PostgrestError) as e:
                    logger.error(
                        "Supabase error storing report",
                        run_id=run_id,
                        error=str(e)
                    )
                    raise APIError(f"Failed to store report: {str(e)}")

        except Exception as e:
            logger.error(
                "Error storing report",
                run_id=run_id,
                error=str(e)
            )
            raise 