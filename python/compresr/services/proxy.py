"""
HTTP Client - Base HTTP functionality for Compresr SDK.

Internal module providing HTTP methods for API calls.
Do not use directly - use CompressionClient or FilterClient.
"""

import json
import ssl
from typing import Any, Dict, Generator, NoReturn, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from ..config import API_CONFIG, HEADERS, STATUS_CODES
from ..exceptions import AuthenticationError, CompresrError
from ..exceptions import ConnectionError as CompresrConnectionError
from ..exceptions import RateLimitError, ScopeError, ServerError, ValidationError


class HTTPClient:
    """Internal HTTP client for Compresr API."""

    def __init__(
        self,
        api_key: str,
        timeout: Optional[int] = None,
    ):
        if not api_key:
            raise AuthenticationError("API key is required")
        if not api_key.startswith(API_CONFIG.API_KEY_PREFIX):
            raise AuthenticationError(
                f"Invalid API key format. Keys must start with '{API_CONFIG.API_KEY_PREFIX}'"
            )

        self._api_key = api_key
        self._base_url = API_CONFIG.BASE_URL
        self._timeout = timeout or API_CONFIG.DEFAULT_TIMEOUT
        self._async_client: Optional["httpx.AsyncClient"] = None

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            HEADERS.API_KEY: self._api_key,
            HEADERS.CONTENT_TYPE: HEADERS.JSON,
            HEADERS.ACCEPT: HEADERS.JSON,
            "User-Agent": "compresr-python-sdk/2.0.0",
        }

    def _url(self, endpoint: str) -> str:
        return f"{self._base_url}{endpoint}"

    def _extract_error_message(self, body: Dict[str, Any]) -> str:
        """Extract user-friendly error message from backend response."""
        # Try common error fields
        if "error" in body:
            return str(body["error"])
        if "detail" in body:
            detail = body["detail"]
            # Pydantic validation errors come as list of dicts
            if isinstance(detail, list):
                errors = []
                for err in detail:
                    if isinstance(err, dict):
                        loc = err.get("loc", [])
                        msg = err.get("msg", "")
                        field = ".".join(str(x) for x in loc) if loc else ""
                        errors.append(f"{field}: {msg}" if field else msg)
                    else:
                        errors.append(str(err))
                return "; ".join(errors) if errors else "Validation error"
            return str(detail)
        if "message" in body:
            return str(body["message"])
        return "Unknown error"

    def _handle_error(self, status_code: int, body: Dict[str, Any]) -> NoReturn:
        """Handle error response with user-friendly messages."""
        msg = self._extract_error_message(body)

        if status_code == STATUS_CODES.UNAUTHORIZED:
            raise AuthenticationError(
                f"Authentication failed: {msg}. Check your API key is valid.", response_data=body
            )
        elif status_code == STATUS_CODES.FORBIDDEN:
            raise ScopeError(
                f"Permission denied: {msg}. Your API key may lack the required scope.",
                response_data=body,
            )
        elif status_code == STATUS_CODES.VALIDATION_ERROR:
            field = body.get("field")
            raise ValidationError(
                f"Invalid request: {msg}",
                field=field if isinstance(field, str) else None,
                response_data=body,
            )
        elif status_code == STATUS_CODES.RATE_LIMITED:
            retry = body.get("retry_after")
            retry_msg = f" Retry after {retry} seconds." if retry else ""
            raise RateLimitError(
                f"Rate limit exceeded: {msg}.{retry_msg}",
                retry_after=retry if isinstance(retry, int) else None,
                response_data=body,
            )
        elif status_code >= 500:
            raise ServerError(
                f"Server error: {msg}. Please try again later or contact support.",
                response_data=body,
            )
        else:
            raise CompresrError(f"Request failed ({status_code}): {msg}", response_data=body)

    # ==================== Sync ====================

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync POST request."""
        url = self._url(endpoint)
        body = json.dumps(data).encode("utf-8")
        req = Request(url, data=body, headers=self._headers, method="POST")

        try:
            ctx = ssl.create_default_context()
            with urlopen(req, timeout=self._timeout, context=ctx) as resp:
                result: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                return result
        except HTTPError as e:
            try:
                err = json.loads(e.read().decode("utf-8"))
            except Exception:
                err = {"error": str(e), "detail": e.reason}
            self._handle_error(e.code, err)
        except URLError as e:
            raise CompresrConnectionError(f"Connection failed: {e.reason}")
        except TimeoutError:
            raise CompresrConnectionError("Request timed out")
        except Exception as e:
            raise CompresrError(f"Request failed: {str(e)}")

    def stream(self, endpoint: str, data: Dict[str, Any]) -> Generator[str, None, None]:
        """Sync streaming POST request."""
        if not HTTPX_AVAILABLE:
            raise ImportError("Streaming requires httpx: pip install httpx")

        url = self._url(endpoint)
        with httpx.Client(timeout=self._timeout) as client:
            with client.stream("POST", url, json=data, headers=self._headers) as resp:
                if resp.status_code >= 400:
                    try:
                        err = (
                            json.loads(resp.read())
                            if resp.read()
                            else {"error": f"HTTP {resp.status_code}"}
                        )
                    except Exception:
                        err = {"error": f"HTTP {resp.status_code}"}
                    self._handle_error(resp.status_code, err)

                for line in resp.iter_text():
                    for single in line.strip().split("\n"):
                        if single.startswith("data: "):
                            chunk = single[6:]
                            if chunk == "[DONE]":
                                return
                            try:
                                parsed = json.loads(chunk)
                                if "content" in parsed:
                                    yield parsed["content"]
                            except json.JSONDecodeError:
                                continue

    # ==================== Async ====================

    async def post_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Async POST request."""
        if not HTTPX_AVAILABLE:
            raise ImportError("Async requires httpx: pip install httpx")

        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)

        url = self._url(endpoint)
        try:
            resp = await self._async_client.post(url, json=data)
            body: Dict[str, Any] = resp.json()
            if resp.status_code >= 400:
                self._handle_error(resp.status_code, body)
            return body
        except httpx.TimeoutException:
            raise CompresrConnectionError("Request timed out")
        except httpx.ConnectError as e:
            raise CompresrConnectionError(f"Connection failed: {str(e)}")

    async def close(self) -> None:
        """Close async client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
