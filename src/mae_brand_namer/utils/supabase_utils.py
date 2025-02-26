"""Supabase connection pooling and management utilities."""

from typing import Optional
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError
from storage3.exceptions import StorageException
from gotrue.errors import AuthError as AuthException

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
                
                return response.data
                
            except (APIError) as e:
                last_error = e
                retries += 1
                if retries < max_retries:
                    logger.warning(
                        "Supabase operation failed, retrying",
                        extra={
                            "attempt": retries,
                            "max_retries": max_retries,
                            "error": str(e),
                            "operation": operation,
                            "table": table
                        }
                    )
                    continue
                break
                
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