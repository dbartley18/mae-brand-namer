"""Supabase connection pooling and management utilities."""

from typing import Optional
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError
from storage3.exceptions import StorageException
from gotrue.errors import AuthError as AuthException
import asyncio

from ..config.settings import settings
from .logging import get_logger

logger = get_logger(__name__)

class SupabaseManager:
    """Singleton manager for Supabase client connections."""
    
    _instance: Optional['SupabaseManager'] = None
    _client: Optional[Client] = None
    
    def __new__(cls) -> 'SupabaseManager':
        """Ensure only one instance of SupabaseManager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the Supabase client if not already initialized."""
        if self._client is None:
            try:
                if not settings.supabase_url or not settings.supabase_service_key:
                    raise ValueError(
                        "Supabase URL and Service Key must be set in environment variables:\n"
                        "- SUPABASE_URL\n"
                        "- SUPABASE_SERVICE_KEY"
                    )
                    
                # Configure client options
                options = ClientOptions(
                    schema="public",
                    headers={
                        "X-Client-Info": "mae-brand-namer",
                    },
                    postgrest_client_timeout=settings.supabase_timeout
                )
                
                # Create the client
                self._client = create_client(
                    settings.supabase_url,
                    settings.supabase_service_key,
                    options=options
                )
                logger.info("Initialized Supabase client")
                
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {str(e)}")
                raise
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance."""
        if self._client is None:
            raise RuntimeError("Supabase client not initialized")
        return self._client
    
    async def execute_with_retry(self, operation: str, table: str, data: dict, max_retries: Optional[int] = None) -> dict:
        """
        Execute a Supabase operation with retry logic.
        
        Args:
            operation (str): Operation type ('insert', 'update', 'upsert', 'delete')
            table (str): Target table name
            data (dict): Data to operate on
            max_retries (Optional[int]): Maximum number of retry attempts. Defaults to settings value.
            
        Returns:
            dict: Operation response data
            
        Raises:
            APIError: If the operation fails after all retries
            ValueError: If operation type is invalid
        """
        if max_retries is None:
            max_retries = settings.max_retries
            
        retries = 0
        last_error = None
        
        # Debug log the operation we're about to attempt
        logger.debug(
            f"Executing Supabase {operation} operation",
            extra={
                "table": table,
                "operation": operation,
                "data_keys": list(data.keys())
            }
        )
        
        while retries < max_retries:
            try:
                if operation == "insert":
                    response = self.client.table(table).insert(data).execute()
                elif operation == "update":
                    response = self.client.table(table).update(data).eq("id", data["id"]).execute()
                elif operation == "upsert":
                    response = self.client.table(table).upsert(data).execute()
                elif operation == "delete":
                    response = self.client.table(table).delete().eq("id", data["id"]).execute()
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                # Log successful operation
                logger.debug(
                    f"Supabase {operation} operation succeeded",
                    extra={
                        "table": table,
                        "operation": operation,
                        "status": "success",
                    }
                )
                
                return response.data
                
            except APIError as e:
                last_error = e
                # Check for specific error types
                error_message = str(e)
                error_code = getattr(e, "code", "unknown")
                error_details = getattr(e, "details", "unknown")
                
                # Log more detailed error information
                logger.error(
                    f"Supabase {operation} operation failed",
                    extra={
                        "table": table,
                        "operation": operation,
                        "error_code": error_code,
                        "error_message": error_message,
                        "error_details": error_details,
                        "attempt": retries + 1,
                        "max_retries": max_retries
                    }
                )
                
                # Determine if we should retry
                retries += 1
                if retries < max_retries:
                    # Add exponential backoff
                    wait_time = 2 ** retries * 0.1  # 0.2s, 0.4s, 0.8s, etc.
                    logger.warning(
                        f"Retrying Supabase operation in {wait_time:.2f}s",
                        extra={
                            "attempt": retries,
                            "max_retries": max_retries,
                            "wait_time": wait_time,
                            "table": table
                        }
                    )
                    await asyncio.sleep(wait_time)
                    continue
                break
            except Exception as e:
                # Handle unexpected errors
                logger.error(
                    f"Unexpected error in Supabase {operation} operation",
                    extra={
                        "table": table,
                        "operation": operation,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                raise
                
        # If we get here, all retries have failed
        logger.error(
            "Supabase operation failed after all retries",
            extra={
                "error": str(last_error),
                "operation": operation,
                "table": table,
                "attempts": retries
            }
        )
        raise last_error

# Global instance
supabase = SupabaseManager() 