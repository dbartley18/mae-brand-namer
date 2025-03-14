"""Supabase connection pooling and management utilities."""

from typing import Optional, Dict, List, Any
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest.exceptions import APIError
from storage3.exceptions import StorageException
from gotrue.errors import AuthError as AuthException
import asyncio
import os
import time
import logging
import traceback

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
    
    def table(self, table_name: str = None, **kwargs):
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
            
        # Extract table_name from kwargs if not provided directly
        if table_name is None and 'table_name' in kwargs:
            table_name = kwargs.pop('table_name')
            
        # Validate required parameters
        if table_name is None:
            raise ValueError("Table name is required but was not provided")
            
        # Handle backward compatibility with code that passes 'operation'
        if 'operation' in kwargs:
            logger.debug(f"Operation (ignored): {kwargs['operation']}")
        
        return self._client.table(table_name)
    
    async def execute_with_retry(self, operation: str, table: str, data: dict, max_retries: Optional[int] = None) -> dict:
        """
        Execute a Supabase operation with retry logic.
        
        Args:
            operation (str): Operation type ('insert', 'update', 'upsert', 'delete', 'select')
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
                elif operation == "select":
                    # Handle select operations
                    query = self.client.table(table).select(data.get("select", "*"))
                    
                    # Apply filters if provided
                    for key, value in data.items():
                        if key != "select" and key != "order" and key != "limit":
                            # Check if the value already includes an operator
                            if isinstance(value, str) and len(value) > 3 and "." in value:
                                # Extract operator and actual value
                                parts = value.split(".", 1)
                                if len(parts) == 2:
                                    operator, actual_value = parts
                                    # Apply the appropriate operator method
                                    if operator == "eq":
                                        query = query.eq(key, actual_value)
                                    elif operator == "neq":
                                        query = query.neq(key, actual_value)
                                    elif operator == "gt":
                                        query = query.gt(key, actual_value)
                                    elif operator == "gte":
                                        query = query.gte(key, actual_value)
                                    elif operator == "lt":
                                        query = query.lt(key, actual_value)
                                    elif operator == "lte":
                                        query = query.lte(key, actual_value)
                                    elif operator == "like":
                                        query = query.like(key, actual_value)
                                    elif operator == "ilike":
                                        query = query.ilike(key, actual_value)
                                    elif operator == "is":
                                        query = query.is_(key, actual_value)
                                    elif operator == "in":
                                        # Handle 'in' operator which needs a list
                                        if actual_value.startswith("(") and actual_value.endswith(")"):
                                            values_list = actual_value[1:-1].split(",")
                                            query = query.in_(key, values_list)
                                        else:
                                            # If not properly formatted, fall back to eq
                                            query = query.eq(key, value)
                                    else:
                                        # Unknown operator, use eq as fallback
                                        query = query.eq(key, value)
                                else:
                                    # No valid operator format, use eq as fallback
                                    query = query.eq(key, value)
                            else:
                                # No operator, use standard eq operator
                                query = query.eq(key, value)
                    
                    # Apply ordering if specified
                    if "order" in data:
                        query = query.order(data["order"])
                    
                    # Apply limit if specified
                    if "limit" in data:
                        query = query.limit(data["limit"])
                    
                    # Execute the query
                    response = query.execute()
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
    
    async def storage_upload_with_retry(self, bucket: str = None, path: str = None, file: bytes = None, **kwargs) -> dict:
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
        
        # Extract parameters from kwargs if not provided directly
        if bucket is None and 'bucket' in kwargs:
            bucket = kwargs.pop('bucket')
        if path is None and 'path' in kwargs:
            path = kwargs.pop('path')
        if file is None and 'file' in kwargs:
            file = kwargs.pop('file')
            
        # Validate required parameters
        if bucket is None or path is None or file is None:
            missing = []
            if bucket is None: missing.append('bucket')
            if path is None: missing.append('path')
            if file is None: missing.append('file')
            raise ValueError(f"Missing required parameters for storage upload: {', '.join(missing)}")
        
        # Log for debugging
        logger.debug(f"Uploading to bucket: {bucket}, path: {path}")
        
        # Handle backward compatibility with code that passes 'operation'
        if 'operation' in kwargs:
            logger.debug(f"Operation (ignored): {kwargs['operation']}")
            
        # Ensure the bucket exists
        self._ensure_bucket_exists(bucket)
        
        while True:
            try:
                result = await self.client.storage.from_(bucket).upload(
                    path=path,
                    file=file,
                    file_options={"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
                    # Enable file overwriting instead of getting 409 conflict errors
                    upsert=True
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
    
    async def storage_get_public_url(self, bucket: str = None, path: str = None, **kwargs) -> str:
        """
        Get the public URL for a file in Supabase Storage.
        
        Args:
            bucket: The storage bucket name
            path: The path to the file within the bucket
            
        Returns:
            The public URL for the file
        """
        try:
            # Extract parameters from kwargs if not provided directly
            if bucket is None and 'bucket' in kwargs:
                bucket = kwargs.pop('bucket')
            if path is None and 'path' in kwargs:
                path = kwargs.pop('path')
                
            # Validate required parameters
            if bucket is None or path is None:
                missing = []
                if bucket is None: missing.append('bucket')
                if path is None: missing.append('path')
                raise ValueError(f"Missing required parameters for getting public URL: {', '.join(missing)}")
            
            # Log for debugging
            logger.debug(f"Getting public URL for bucket: {bucket}, path: {path}")
            
            # Handle backward compatibility with code that passes 'operation'
            if 'operation' in kwargs:
                logger.debug(f"Operation (ignored): {kwargs['operation']}")
                
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