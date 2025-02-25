"""Process supervision and monitoring for the brand naming workflow."""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import traceback

from langchain.callbacks import tracing_enabled
from ..utils.logging import get_logger
from ..utils.supabase_utils import supabase
from ..config.settings import settings

logger = get_logger(__name__)

class ProcessSupervisor:
    """Supervises and monitors the brand naming workflow process."""
    
    def __init__(self):
        """Initialize the ProcessSupervisor."""
        self.retry_counts: Dict[Tuple[str, str, str], int] = {}  # (run_id, agent_type, task_name) -> retry_count
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        self.retry_backoff = settings.retry_backoff
        self.retry_max_delay = settings.retry_max_delay
    
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
            with tracing_enabled():
                task_data = {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "task_name": task_name,
                    "status": "started",
                    "start_time": datetime.now().isoformat(),
                    "metadata": {
                        "retry_count": self._get_retry_count(run_id, agent_type, task_name)
                    }
                }
                
                await supabase.execute_with_retry(
                    operation="insert",
                    table="task_execution",
                    data=task_data
                )
                
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
            raise
    
    async def log_task_completion(self, run_id: str, agent_type: str, task_name: str) -> None:
        """
        Log the successful completion of a task.
        
        Args:
            run_id (str): Unique identifier for the workflow run
            agent_type (str): Type of agent that executed the task
            task_name (str): Name of the completed task
        """
        try:
            with tracing_enabled():
                completion_data = {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "task_name": task_name,
                    "status": "completed",
                    "completion_time": datetime.now().isoformat()
                }
                
                await supabase.execute_with_retry(
                    operation="update",
                    table="task_execution",
                    data=completion_data
                )
                
                logger.info(
                    "Task completed",
                    run_id=run_id,
                    agent_type=agent_type,
                    task_name=task_name
                )
                
        except Exception as e:
            logger.error(
                "Failed to log task completion",
                run_id=run_id,
                agent_type=agent_type,
                task_name=task_name,
                error=str(e)
            )
            raise
    
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
            with tracing_enabled():
                retry_count = self._increment_retry_count(run_id, agent_type, task_name)
                should_retry = retry_count < self.max_retries
                
                error_data = {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "task_name": task_name,
                    "status": "error",
                    "error_time": datetime.now().isoformat(),
                    "error_type": error.__class__.__name__,
                    "error_message": str(error),
                    "error_traceback": traceback.format_exc(),
                    "retry_count": retry_count,
                    "should_retry": should_retry,
                    "retry_delay": self._calculate_retry_delay(retry_count) if should_retry else None
                }
                
                await supabase.execute_with_retry(
                    operation="insert",
                    table="task_error",
                    data=error_data
                )
                
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
                error=str(e)
            )
            raise
    
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
        return retry_count < self.max_retries 