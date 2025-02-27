"""Process supervision and monitoring for the brand naming workflow."""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import traceback
import os

from langchain.callbacks import tracing_enabled
from supabase import create_client, Client
from ..utils.logging import get_logger
from ..config.settings import settings
from ..utils.supabase_utils import SupabaseManager
from ..models.app_config import AppConfig
from ..llms.chat_google_generative_ai import ChatGoogleGenerativeAI

logger = get_logger(__name__)

class ProcessSupervisor:
    """Supervises and monitors the brand naming workflow process."""
    
    def __init__(self, dependencies=None, supabase=None, app_config: AppConfig = None):
        """Initialize the ProcessSupervisor with dependencies."""
        self.retry_counts: Dict[Tuple[str, str, str], int] = {}  # (run_id, agent_type, task_name) -> retry_count
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        self.retry_backoff = settings.retry_backoff
        self.retry_max_delay = settings.retry_max_delay
        
        if dependencies:
            self.supabase = dependencies.supabase
            self.langsmith = dependencies.langsmith
        else:
            self.supabase = supabase
            self.langsmith = None
        
        # Get agent-specific configuration
        self.app_config = app_config or AppConfig()
        agent_name = "process_supervisor"
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
            f"Initialized Process Supervisor with temperature: {self.temperature}",
            extra={"agent": agent_name, "temperature": self.temperature}
        )
    
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
            start_time = datetime.now()
            
            # Create/update record in process_logs table
            await self.supabase.table("process_logs").upsert({
                "run_id": run_id,
                "agent_type": agent_type,
                "task_name": task_name,
                "status": "in_progress",
                "start_time": start_time.isoformat(),
                "retry_count": self._get_retry_count(run_id, agent_type, task_name),
                "last_updated": datetime.now().isoformat()
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
    
    async def log_task_completion(self, run_id: str, agent_type: str, task_name: str) -> None:
        """
        Log the successful completion of a task.
        
        Args:
            run_id (str): Unique identifier for the workflow run
            agent_type (str): Type of agent that executed the task
            task_name (str): Name of the completed task
        """
        try:
            end_time = datetime.now()
            
            # Get the existing record to calculate duration
            process_logs = await self.supabase.table("process_logs") \
                .select("*") \
                .eq("run_id", run_id) \
                .eq("task_name", task_name) \
                .execute()
                
            start_time = None
            if process_logs.data and len(process_logs.data) > 0:
                start_time_str = process_logs.data[0].get("start_time")
                if start_time_str:
                    start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            
            duration_sec = None
            if start_time:
                duration_sec = int((end_time - start_time).total_seconds())
            
            # Update the record
            await self.supabase.table("process_logs").upsert({
                "run_id": run_id,
                "agent_type": agent_type,
                "task_name": task_name,
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_sec": duration_sec,
                "last_updated": datetime.now().isoformat()
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
            agent_type (str): Type of agent that encountered the error
            task_name (str): Name of the task that failed
            error (Exception): The error that occurred
            
        Returns:
            bool: True if the task should be retried, False otherwise
        """
        try:
            retry_count = self._increment_retry_count(run_id, agent_type, task_name)
            should_retry = retry_count < self.max_retries
            
            # Update the process_logs record
            await self.supabase.table("process_logs").upsert({
                "run_id": run_id,
                "agent_type": agent_type,
                "task_name": task_name,
                "status": "error",
                "error_message": str(error),
                "retry_count": retry_count,
                "last_retry_at": datetime.now().isoformat() if should_retry else None,
                "retry_status": "pending" if should_retry else "exhausted",
                "last_updated": datetime.now().isoformat()
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