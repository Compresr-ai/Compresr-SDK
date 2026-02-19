"""
SDK Configuration
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class APIConfig:
    """API configuration."""

    BASE_URL: str = "https://api.compresr.ai"  # Fixed, not configurable
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


@dataclass(frozen=True)
class Models:
    """Available compression models.

    Agnostic models (public API):
    - A_CMPRSR_V1: LLM-based abstractive compression using Qwen3-4B (default)
    - A_CMPRSR_V1_FLASH: Fast extractive compression using LLMLingua-2

    Question-specific models (demo/JWT auth):
    - QS_CMPRSR_V1: Question-specific compression
    - QSR_CMPRSR_V1: Question-specific with reranking
    - QSLF_CMPRSR_V1: Question-specific with longformer
    """

    # Agnostic models (public API)
    A_CMPRSR_V1: str = "A_CMPRSR_V1"
    A_CMPRSR_V1_FLASH: str = "A_CMPRSR_V1_FLASH"

    # Question-specific models (demo/JWT auth only)
    QS_CMPRSR_V1: str = "QS_CMPRSR_V1"
    QSR_CMPRSR_V1: str = "QSR_CMPRSR_V1"
    QSLF_CMPRSR_V1: str = "QSLF_CMPRSR_V1"

    # Aliases
    DEFAULT: str = "A_CMPRSR_V1"
    FAST: str = "A_CMPRSR_V1_FLASH"
    DEFAULT_QS: str = "QS_CMPRSR_V1"


# Singleton instances
API_CONFIG = APIConfig()
ENDPOINTS = Endpoints()
HEADERS = Headers()
STATUS_CODES = StatusCodes()
MODELS = Models()
