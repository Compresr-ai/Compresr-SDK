"""
SDK Configuration
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class APIConfig:
    """API configuration."""

    DEFAULT_BASE_URL: str = os.getenv("COMPRESR_BASE_URL", "https://api.compresr.ai")
    API_KEY_PREFIX: str = "cmp_"
    DEFAULT_TIMEOUT: int = 60
    BATCH_TIMEOUT: int = 120
    STREAM_TIMEOUT: int = 300


@dataclass(frozen=True)
class Endpoints:
    """API endpoints."""

    COMPRESS: str = "/api/compress/generate"
    COMPRESS_BATCH: str = "/api/compress/batch"
    COMPRESS_STREAM: str = "/api/compress/stream"


@dataclass(frozen=True)
class Headers:
    """HTTP headers."""

    API_KEY: str = "X-API-Key"
    CONTENT_TYPE: str = "Content-Type"
    ACCEPT: str = "Accept"
    JSON: str = "application/json"
    SSE: str = "text/event-stream"


@dataclass(frozen=True)
class StatusCodes:
    """HTTP status codes."""

    OK: int = 200
    BAD_REQUEST: int = 400
    UNAUTHORIZED: int = 401
    FORBIDDEN: int = 403
    NOT_FOUND: int = 404
    VALIDATION_ERROR: int = 422
    RATE_LIMITED: int = 429
    SERVER_ERROR: int = 500


# Singleton instances
API_CONFIG = APIConfig()
ENDPOINTS = Endpoints()
HEADERS = Headers()
STATUS_CODES = StatusCodes()
