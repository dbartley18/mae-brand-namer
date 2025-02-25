"""Supabase connection pooling and management utilities."""

from typing import Optional
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from supabase import APIError, StorageException, AuthException, PostgrestError

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
                # Configure client options
                options = ClientOptions(
                    schema="public",
                    headers={
                        "X-Client-Info": "mae-brand-namer",
                    },
                    postgrest_client_timeout=10  # 10 second timeout
                )
                
                # Create the client
                self._client = create_client(
                    settings.supabase_url,
                    settings.supabase_key,
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
    
    def execute_with_retry(self, operation: str, table: str, data: dict, max_retries: int = 3) -> dict:
        """
        Execute a Supabase operation with retry logic.
        
        Args:
            operation (str): Operation type ('insert', 'update', 'upsert', 'delete')
            table (str): Target table name
            data (dict): Data to operate on
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            dict: Operation response data
            
        Raises:
            APIError: If the operation fails after all retries
        """
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
                
            except (APIError, PostgrestError) as e:
                last_error = e
                retries += 1
                if retries < max_retries:
                    logger.warning(
                        f"Supabase operation failed (attempt {retries}/{max_retries})",
                        error=str(e),
                        operation=operation,
                        table=table
                    )
                    continue
                break
                
        logger.error(
            "Supabase operation failed after all retries",
            error=str(last_error),
            operation=operation,
            table=table
        )
        raise last_error

# Global instance
supabase = SupabaseManager() 