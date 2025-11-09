"""
Middleware components for the spam detection API.
"""

import time
import logging
from typing import Dict, List
from collections import defaultdict
from threading import Lock
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiting middleware to prevent API abuse."""

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.storage: Dict[str, List[float]] = defaultdict(list)
        self.lock = Lock()

    async def __call__(self, request: Request, call_next):
        """Process rate limiting for incoming requests."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        with self.lock:
            # Clean old entries
            self.storage[client_ip] = [
                req_time for req_time in self.storage[client_ip]
                if current_time - req_time < self.window_seconds
            ]

            # Check rate limit
            if len(self.storage[client_ip]) >= self.max_requests:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds} seconds.",
                        "error": "Too Many Requests",
                        "retry_after": self.window_seconds
                    }
                )

            # Add current request
            self.storage[client_ip].append(current_time)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.max_requests - len(self.storage[client_ip]))
        )
        return response


async def log_request_middleware(request: Request, call_next):
    """
    Middleware to log each request with processing time.
    Adds X-Process-Time header to responses.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(f"Request: {request.url} - Duration: {process_time:.4f} seconds")
    response.headers["X-Process-Time"] = str(process_time)

    return response
