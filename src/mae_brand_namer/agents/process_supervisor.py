"""Process supervision and monitoring for the brand naming workflow."""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import traceback

from langchain.callbacks import tracing_enabled
from supabase import create_client, Client
from postgrest.exceptions import APIError
from ..utils.logging import get_logger
from ..config.settings import settings
from ..utils.supabase_utils import SupabaseManager

logger = get_logger(__name__)

class ProcessSupervisor:
    """Supervises and monitors the brand naming workflow process."""
    
    def __init__(self, dependencies=None, supabase: SupabaseManager = None):
        """Initialize the ProcessSupervisor."""
        self.retry_counts: Dict[Tuple[str, str, str], int] = {}  # (run_id, agent_type, task_name) -> retry_count
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        self.retry_backoff = settings.retry_backoff
        self.retry_max_delay = settings.retry_max_delay
        
        # Initialize Supabase client
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase or SupabaseManager()
            self.langsmith = None
    
    def _get_retry_key(self, run_id: str, agent_type: str, task_name: str) -> Tuple[str, str, str]:
        """Get the key for retry count tracking."""
        return (run_id, agent_type, task_name)
    
    def _get_retry_count(self, run_id: str, agent_type: str, task_name: str) -> int:
        """Get the current retry count for a task."""
        key = self._get_retry_key(run_id, agent_type, task_name)
        return self.retry_counts.get(key, 0)
    
    def _increment_retry_count(self, run_id: str, agent_type: str, task_name: str) -> int:
        """Increment the retry count for a task."""
        key = self._get_retry_key(run_id, agent_type, task_name)
        current_count = self.retry_counts.get(key, 0)
        self.retry_counts[key] = current_count + 1
        return current_count + 1
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate the delay before the next retry using exponential backoff."""
        delay = self.retry_delay * (self.retry_backoff ** (retry_count - 1))
        return min(delay, self.retry_max_delay)
    
    async def log_task_start(self, run_id: str, agent_type: str, task_name: str) -> None:
        """
        Log the start of a task execution.
        
        Args:
            run_id (str): Unique identifier for the workflow run
            agent_type (str): Type of agent executing the task
            task_name (str): Name of the task being executed
        """
        try:
            # Skip if run_id is null or empty
            if not run_id:
                logger.warning(
                    "Skipping task logging - missing run_id",
                    agent_type=agent_type,
                    task_name=task_name
                )
                return
                
            start_time = datetime.now()
            
            # Ensure the run_id exists in workflow_runs table first
            await self._ensure_workflow_run_exists(run_id)
            
            # Create/update record in process_logs table
            await self.supabase.table("process_logs").upsert({
                "run_id": run_id,
                "agent_type": agent_type,
                "task_name": task_name,
                "status": "in_progress",
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "retry_count": self._get_retry_count(run_id, agent_type, task_name),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
            
            logger.info(
                "Task started",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name
            )
                
        except Exception as e:
            logger.error(
                "Failed to log task start",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name,
                error=str(e)
            )
            # Don't raise exception - monitoring should not block workflow
    
    async def _ensure_workflow_run_exists(self, run_id: str) -> None:
        """
        Ensure that a workflow run record exists in the workflow_runs table.
        If it doesn't exist, create it.
        
        Args:
            run_id (str): The run ID to check/create
        """
        if not run_id:
            return
            
        try:
            # Check if run exists
            response = await self.supabase.table("workflow_runs").select("run_id") \
                .eq("run_id", run_id).execute()
                
            # If run doesn't exist, create it
            if not response.data or len(response.data) == 0:
                logger.info(f"Creating workflow_runs record for run_id: {run_id}")
                await self.supabase.table("workflow_runs").insert({
                    "run_id": run_id,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "in_progress"
                }).execute()
        except Exception as e:
            logger.error(f"Error ensuring workflow run exists: {str(e)}")
    
    async def log_task_completion(self, run_id: str, agent_type: str, task_name: str) -> None:
        """
        Log the successful completion of a task.
        
        Args:
            run_id (str): Unique identifier for the workflow run
            agent_type (str): Type of agent executing the task
            task_name (str): Name of the task being executed
        """
        try:
            # Skip if run_id is null or empty
            if not run_id:
                logger.warning(
                    "Skipping task completion logging - missing run_id",
                    agent_type=agent_type,
                    task_name=task_name
                )
                return
                
            end_time = datetime.now()
            
            # Ensure the run_id exists in workflow_runs table first
            await self._ensure_workflow_run_exists(run_id)
            
            # Get existing record to calculate duration
            response = await self.supabase.table("process_logs") \
                .select("start_time") \
                .eq("run_id", run_id) \
                .eq("agent_type", agent_type) \
                .eq("task_name", task_name) \
                .execute()
            
            duration_sec = None
            if response.data and len(response.data) > 0:
                start_time_str = response.data[0].get("start_time")
                if start_time_str:
                    try:
                        # Handle different formats that might come from the database
                        if "T" in start_time_str:
                            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                        else:
                            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        logger.warning(f"Could not parse start_time: {start_time_str}")
            
            if start_time:
                duration_sec = int((end_time - start_time).total_seconds())
            
            # Update the record
            await self.supabase.table("process_logs").upsert({
                "run_id": run_id,
                "agent_type": agent_type,
                "task_name": task_name,
                "status": "completed",
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": duration_sec,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
            
            logger.info(
                "Task completed",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name,
                duration_sec=duration_sec
            )
                
        except Exception as e:
            logger.error(
                "Failed to log task completion",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name,
                error=str(e)
            )
            # Don't raise exception - monitoring should not block workflow
    
    async def log_task_error(
        self,
        run_id: str,
        agent_type: str,
        task_name: str,
        error: Exception
    ) -> bool:
        """
        Log a task error and determine if it should be retried.
        
        Args:
            run_id (str): Unique identifier for the workflow run
            agent_type (str): Type of agent executing the task
            task_name (str): Name of the task being executed
            error (Exception): The error that occurred
            
        Returns:
            bool: True if the task should be retried, False otherwise
        """
        try:
            # Skip if run_id is null or empty
            if not run_id:
                logger.warning(
                    "Skipping task error logging - missing run_id",
                    agent_type=agent_type,
                    task_name=task_name,
                    error=str(error)
                )
                return False
                
            retry_count = self._increment_retry_count(run_id, agent_type, task_name)
            should_retry = retry_count < self.max_retries
            current_time = datetime.now()
            
            # Ensure the run_id exists in workflow_runs table first
            await self._ensure_workflow_run_exists(run_id)
            
            # Extract error details from the exception if available
            error_details = {}
            
            # Check for context attribute on the error object
            if hasattr(error, 'context'):
                error_details['context'] = error.context
                
            # Add error type information
            error_details['error_type'] = type(error).__name__
            
            # If error has args, include them
            if hasattr(error, 'args') and error.args:
                error_details['args'] = [str(arg) for arg in error.args]
                
            # For APIError specifically, extract more details
            if isinstance(error, APIError):
                if hasattr(error, 'details'):
                    error_details['api_details'] = error.details
                if hasattr(error, 'code'):
                    error_details['code'] = error.code
                if hasattr(error, 'hint'):
                    error_details['hint'] = error.hint
                    
            # Update the process_logs record
            await self.supabase.table("process_logs").upsert({
                "run_id": run_id,
                "agent_type": agent_type,
                "task_name": task_name,
                "status": "error",
                "error_message": str(error),
                "error_details": error_details,
                "retry_count": retry_count,
                "last_retry_at": current_time.strftime("%Y-%m-%d %H:%M:%S") if should_retry else None,
                "retry_status": "pending" if should_retry else "exhausted",
                "last_updated": current_time.strftime("%Y-%m-%d %H:%M:%S")
            }).execute()
            
            log_level = logger.warning if should_retry else logger.error
            log_level(
                "Task error occurred",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name,
                error=str(error),
                retry_count=retry_count,
                should_retry=should_retry
            )
            
            return should_retry
                
        except Exception as e:
            logger.error(
                "Failed to log task error",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name,
                original_error=str(error),
                logging_error=str(e)
            )
            # Return False to prevent infinite retry loops if monitoring fails
            return False
    
    async def should_retry_task(self, run_id: str, agent_type: str, task_name: str) -> bool:
        """
        Check if a task should be retried based on its retry count and configuration.
        
        Args:
            run_id (str): Unique identifier for the workflow run
            agent_type (str): Type of agent executing the task
            task_name (str): Name of the task
            
        Returns:
            bool: True if the task should be retried, False otherwise
        """
        retry_count = self._get_retry_count(run_id, agent_type, task_name)
        should_retry = retry_count < self.max_retries
        
        if should_retry:
            logger.info(
                f"Task will be retried (attempt {retry_count + 1}/{self.max_retries})",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name
            )
        else:
            logger.error(
                f"Task failed permanently after {retry_count} retries",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name
            )
            
        return should_retry 