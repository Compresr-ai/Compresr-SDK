"""SDK configuration."""

import os
from dataclasses import dataclass


def _get_base_url() -> str:
    return os.getenv("COMPRESR_BASE_URL", "https://api.compresr.ai")


@dataclass(frozen=True)
class APIConfig:
    API_KEY_PREFIX: str = "cmp_"
    DEFAULT_TIMEOUT: int = 300

    @property
    def BASE_URL(self) -> str:
        return _get_base_url()


@dataclass(frozen=True)
class Endpoints:
    COMPRESS: str = "/api/compress/question-specific/"
    COMPRESS_STREAM: str = "/api/compress/question-specific/stream"
    COMPRESS_BATCH: str = "/api/compress/question-specific/batch"
    COMPRESS_TOOL_OUTPUT: str = "/api/compress/tool-output/"


@dataclass(frozen=True)
class Headers:
    API_KEY: str = "X-API-Key"
    CONTENT_TYPE: str = "Content-Type"
    ACCEPT: str = "Accept"
    JSON: str = "application/json"
    SSE: str = "text/event-stream"


@dataclass(frozen=True)
class StatusCodes:
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
    """Convenience names. The backend is the authority on which models exist —
    pass any model name as a string and the API will validate."""

    LATTE: str = "latte_v1"
    LATTE_V1: str = "latte_v1"
    LATTE_V2: str = "latte_v2"
    DEFAULT: str = "latte_v1"


API_CONFIG = APIConfig()
ENDPOINTS = Endpoints()
HEADERS = Headers()
STATUS_CODES = StatusCodes()
MODELS = Models()
