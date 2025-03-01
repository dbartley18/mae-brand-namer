"""Rate limiter utilities for API calls."""

import asyncio
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls.
    
    Tracks call times and enforces a maximum number of calls per minute by
    implementing a token bucket algorithm.
    """
    
    def __init__(self, rpm_limit: int = 15, buffer_percent: float = 0.1):
        """Initialize the rate limiter.
        
        Args:
            rpm_limit: Maximum requests per minute
            buffer_percent: Additional buffer percentage to stay under the limit
        """
        self.rpm_limit = rpm_limit
        self.buffer_percent = buffer_percent
        self.calls = []  # List of timestamps for recent calls
        self.lock = asyncio.Lock()  # Lock for thread safety
        self.last_warning_time = 0  # To prevent log flooding
    
    async def wait_if_needed(self, call_id: Optional[str] = None) -> float:
        """Wait if needed to respect the rate limit.
        
        Args:
            call_id: Optional identifier for the call (for logging)
            
        Returns:
            float: The wait time in seconds
        """
        async with self.lock:
            # First, clean up old calls (older than 60 seconds)
            current_time = time.time()
            self.calls = [t for t in self.calls if current_time - t < 60]
            
            # Calculate how many calls we've made in the last minute
            calls_in_last_minute = len(self.calls)
            
            # Calculate effective limit with buffer
            effective_limit = self.rpm_limit * (1 - self.buffer_percent)
            
            # Calculate wait time if we're approaching the limit
            wait_time = 0
            if calls_in_last_minute >= effective_limit:
                # We need to wait - calculate how long until we're under the limit
                # Find the oldest call that puts us over the limit
                oldest_relevant_call = self.calls[-(int(effective_limit) + 1)]
                # Wait until that call is a minute old
                wait_time = max(0, 60 - (current_time - oldest_relevant_call))
                
                # Log warning if we haven't recently
                if current_time - self.last_warning_time > 30:
                    logger.warning(
                        f"Rate limit {effective_limit}/{self.rpm_limit} RPM approaching - throttling API calls. "
                        f"Current usage: {calls_in_last_minute} calls in last minute. "
                        f"Waiting {wait_time:.2f}s for {call_id or 'unknown call'}"
                    )
                    self.last_warning_time = current_time
            
            # Record this call's time after any wait
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Update current time after waiting
                current_time = time.time()
            
            # Add this call to the history
            self.calls.append(current_time)
            
            return wait_time

# Global instance for Google API rate limiting (15 RPM)
google_api_limiter = RateLimiter(rpm_limit=15, buffer_percent=0.1) 