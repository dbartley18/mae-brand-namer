"""Supabase connection pooling and management utilities."""

from typing import Optional, Dict, List
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError
from storage3.exceptions import StorageException
from gotrue.errors import AuthError as AuthException
import asyncio
import os

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
    
    def table(self, table_name: str):
        """
        Access a Supabase table directly.
        
        Args:
            table_name (str): Name of the table to access
            
        Returns:
            Table query builder for the specified table
            
        Raises:
            RuntimeError: If the client is not initialized
        """
        if self._client is None:
            raise RuntimeError("Supabase client not initialized")
        return self._client.table(table_name)
    
    async def execute_with_retry(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a SQL query with retry logic."""
        retry_count = 0
        max_retries = int(os.getenv('MAX_RETRIES', '3'))
        base_delay = float(os.getenv('RETRY_DELAY', '1'))
        backoff_factor = float(os.getenv('RETRY_BACKOFF', '2'))
        
        while True:
            try:
                result = await self.client.query(query, params).execute()
                return result.data
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Failed to execute query after {max_retries} retries: {str(e)}")
                    raise
                
                delay = base_delay * (backoff_factor ** (retry_count - 1))
                logger.warning(f"Query failed, retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries}): {str(e)}")
                await asyncio.sleep(delay)
    
    async def storage_upload_with_retry(self, bucket: str, path: str, file: bytes) -> dict:
        """
        Upload a file to Supabase Storage with retry logic.
        
        Args:
            bucket: The storage bucket name
            path: The path within the bucket where the file will be stored
            file: The file content as bytes
            
        Returns:
            The upload result from Supabase
        """
        retry_count = 0
        max_retries = int(os.getenv('MAX_RETRIES', '3'))
        base_delay = float(os.getenv('RETRY_DELAY', '1'))
        backoff_factor = float(os.getenv('RETRY_BACKOFF', '2'))
        
        # Ensure the bucket exists
        self._ensure_bucket_exists(bucket)
        
        while True:
            try:
                result = await self.client.storage.from_(bucket).upload(
                    path=path,
                    file=file,
                    file_options={"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
                )
                return result
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Failed to upload file after {max_retries} retries: {str(e)}")
                    raise
                
                delay = base_delay * (backoff_factor ** (retry_count - 1))
                logger.warning(f"Upload failed, retrying in {delay:.2f} seconds (attempt {retry_count}/{max_retries}): {str(e)}")
                await asyncio.sleep(delay)
    
    async def storage_get_public_url(self, bucket: str, path: str) -> str:
        """
        Get the public URL for a file in Supabase Storage.
        
        Args:
            bucket: The storage bucket name
            path: The path to the file within the bucket
            
        Returns:
            The public URL for the file
        """
        try:
            # Get the public URL from Supabase
            public_url = self.client.storage.from_(bucket).get_public_url(path)
            return public_url
        except Exception as e:
            logger.error(f"Failed to get public URL: {str(e)}")
            raise
    
    def _ensure_bucket_exists(self, bucket: str) -> None:
        """
        Check if a bucket exists, and create it if it doesn't.
        
        Args:
            bucket: The storage bucket name
        """
        try:
            # List buckets to check if the target bucket exists
            buckets = self.client.storage.list_buckets()
            bucket_exists = any(b["name"] == bucket for b in buckets)
            
            if not bucket_exists:
                logger.info(f"Creating storage bucket: {bucket}")
                self.client.storage.create_bucket(
                    bucket,
                    {"public": True}  # Make the bucket public so reports can be accessed via URLs
                )
        except Exception as e:
            logger.error(f"Error checking/creating bucket {bucket}: {str(e)}")
            # Continue anyway, the upload will fail if the bucket doesn't exist

# Global instance
supabase = SupabaseManager() 