"""
SDK Configuration
"""

import os
from dataclasses import dataclass


def _get_base_url() -> str:
    """Get base URL from environment or default to production."""
    return os.getenv("COMPRESR_BASE_URL", "https://api.compresr.ai")


@dataclass(frozen=True)
class APIConfig:
    """API configuration."""

    API_KEY_PREFIX: str = "cmp_"
    DEFAULT_TIMEOUT: int = 60
    STREAM_TIMEOUT: int = 300

    @property
    def BASE_URL(self) -> str:
        """Get base URL (reads from env each time for test flexibility)."""
        return _get_base_url()


@dataclass(frozen=True)
class Endpoints:
    """API endpoints."""

    # Agnostic compression (no query required)
    COMPRESS_AGNOSTIC: str = "/api/compress/question-agnostic/"
    COMPRESS_AGNOSTIC_STREAM: str = "/api/compress/question-agnostic/stream"

    # Query-specific compression (query required)
    COMPRESS_QS: str = "/api/compress/question-specific/"
    COMPRESS_QS_STREAM: str = "/api/compress/question-specific/stream"


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

    Compression models (CompressionClient):
    - espresso_v1: Agnostic compression (default) - no query needed
    - latte_v1: Query-specific compression - query REQUIRED, supports compression_ratio

    Filter models (FilterClient):
    - coldbrew_v1: Chunk-level filter - query REQUIRED, no compression_ratio

    Agentic models (for tool/history compression):
    - agentic_history_lingua: History compression (Lingua) - no query
    - agentic_tool_output_gemfilter: Tool output (GemFilter) - query required
    - agentic_tool_output_lingua: Tool output (Lingua) - no query
    - agentic_tool_discovery_sat: Tool discovery (SaT) - query required
    """

    # Compression models
    ESPRESSO: str = "espresso_v1"
    LATTE: str = "latte_v1"

    # Filter models
    COLDBREW: str = "coldbrew_v1"

    # Agentic models
    AGENTIC_HISTORY_LINGUA: str = "agentic_history_lingua"
    AGENTIC_TOOL_OUTPUT_GEMFILTER: str = "agentic_tool_output_gemfilter"
    AGENTIC_TOOL_OUTPUT_LINGUA: str = "agentic_tool_output_lingua"
    AGENTIC_TOOL_DISCOVERY_SAT: str = "agentic_tool_discovery_sat"

    # Default model aliases
    DEFAULT_COMPRESSION: str = "espresso_v1"
    DEFAULT_FILTER: str = "coldbrew_v1"
    DEFAULT: str = "espresso_v1"


# Allowed models per client type
ALLOWED_COMPRESSION_MODELS = frozenset({"espresso_v1", "latte_v1"})
ALLOWED_FILTER_MODELS = frozenset({"coldbrew_v1"})

# Models that require a query parameter
QUERY_REQUIRED_MODELS = frozenset({"latte_v1", "coldbrew_v1"})

# Endpoint routing
AGNOSTIC_ENDPOINT_MODELS = frozenset({"espresso_v1"})
QS_ENDPOINT_MODELS = frozenset({"latte_v1", "coldbrew_v1"})


# Singleton instances
API_CONFIG = APIConfig()
ENDPOINTS = Endpoints()
HEADERS = Headers()
STATUS_CODES = StatusCodes()
MODELS = Models()
