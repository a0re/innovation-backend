"""
Exception handlers for the API.
"""

import logging
from datetime import datetime
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler.
    Returns consistent JSON error responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error": "An error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": str(exc),
            "error": "Validation error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error. Please contact administrator.",
            "error": "Something went wrong",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )
