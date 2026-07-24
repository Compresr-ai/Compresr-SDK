"""Internal HTTP client for Compresr API requests."""

import asyncio
import json
import os
import ssl
import time
import warnings
from typing import Any, Dict, Generator, NoReturn, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import httpx

from ..config import API_CONFIG, HEADERS, STATUS_CODES
from ..exceptions import (
    ApiKeyBudgetError,
    AuthenticationError,
    BudgetLimitError,
    CompresrConnectionError,
    CompresrError,
    ContentPolicyError,
    ContextWindowExceededError,
    DailyLimitError,
    InsufficientCreditsError,
    ModelNotFoundError,
    RateLimitError,
    ScopeError,
    ServerError,
    ServiceUnavailableError,
    TargetAuthenticationError,
    ValidationError,
)
from ..retry import RetryConfig, compute_backoff

try:
    from importlib.metadata import version as get_version

    SDK_VERSION = get_version("compresr")
except Exception:
    SDK_VERSION = "0.0.0-dev"


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class HTTPClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        from ..credentials import resolve_api_key

        api_key = resolve_api_key(api_key)
        if not api_key:
            raise AuthenticationError(
                "No API key found. Pass api_key='cmp_...', set COMPRESR_API_KEY, "
                "or run `compresr-sdk login` to authenticate."
            )
        if not api_key.startswith(API_CONFIG.API_KEY_PREFIX):
            raise AuthenticationError(
                f"Invalid API key format. Keys must start with '{API_CONFIG.API_KEY_PREFIX}'"
            )

        self._api_key = api_key
        self._base_url = (base_url or API_CONFIG.BASE_URL).rstrip("/")
        self._timeout = timeout or API_CONFIG.DEFAULT_TIMEOUT
        self._retry_config = retry_config if retry_config is not None else RetryConfig()
        self._warn_if_cleartext()
        # Eager init avoids a race where two coroutines build duplicate
        # AsyncClients on the first call. httpx is a hard dependency now,
        # so there's no reason to defer.
        self._async_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=self._timeout, headers=self._headers
        )

    def _warn_if_cleartext(self) -> None:
        parsed = urlparse(self._base_url)
        scheme = (parsed.scheme or "").lower()
        if scheme != "http":
            return
        host = (parsed.hostname or "").lower()
        if host in _LOCAL_HOSTS:
            return
        if os.environ.get("COMPRESR_ALLOW_INSECURE") == "1":
            warnings.warn(
                f"COMPRESR_BASE_URL is {scheme} — API key will be transmitted "
                "in cleartext. Use HTTPS in production.",
                stacklevel=2,
            )
            return
        from ..exceptions.exceptions import CompresrError

        raise CompresrError(
            f"Refusing to send API key over cleartext http to {self._base_url!r}. "
            "Set COMPRESR_ALLOW_INSECURE=1 to override (dev only).",
            code="insecure_base_url",
        )

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            HEADERS.API_KEY: self._api_key,
            HEADERS.CONTENT_TYPE: HEADERS.JSON,
            HEADERS.ACCEPT: HEADERS.JSON,
            "User-Agent": f"compresr-python-sdk/{SDK_VERSION}",
        }

    def _url(self, endpoint: str) -> str:
        # Endpoint must be a server-relative path. Reject absolute URLs to
        # stop a misconfigured caller from redirecting the request to an
        # arbitrary host with the API key attached.
        assert endpoint.startswith("/") and "://" not in endpoint, f"Invalid endpoint: {endpoint!r}"
        return f"{self._base_url}{endpoint}"

    def _extract_error_message(self, body: Dict[str, Any]) -> str:
        if "error" in body:
            return str(body["error"])
        if "detail" in body:
            detail = body["detail"]
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
        msg = self._extract_error_message(body)
        code = body.get("code", "")

        if code == "insufficient_credits":
            raise InsufficientCreditsError(
                f"Insufficient credits: {msg}",
                credits_required=body.get("credits_required"),
                credits_remaining=body.get("credits_remaining"),
                response_data=body,
            )
        elif code == "budget_limit_reached":
            raise BudgetLimitError(
                f"Budget limit reached: {msg}",
                current_budget=body.get("current_budget"),
                budget_used=body.get("budget_used"),
                response_data=body,
            )
        elif code == "api_key_budget_exceeded":
            raise ApiKeyBudgetError(
                f"API key budget exceeded: {msg}",
                api_key_budget=body.get("api_key_budget"),
                api_key_used=body.get("api_key_used"),
                response_data=body,
            )
        elif code == "daily_limit_exceeded":
            raise DailyLimitError(
                f"Daily limit exceeded: {msg}",
                daily_limit=body.get("daily_limit"),
                requests_used=body.get("requests_used"),
                response_data=body,
            )
        elif code == "model_not_found":
            raise ModelNotFoundError(
                f"Model not found: {msg}",
                model_name=body.get("model_name"),
                available_models=body.get(
                    "available_models", body.get("details", {}).get("available_models")
                ),
                response_data=body,
            )
        elif code == "context_window_exceeded":
            raise ContextWindowExceededError(
                f"Context window exceeded: {msg}",
                max_tokens=body.get("max_tokens"),
                actual_tokens=body.get("actual_tokens"),
                response_data=body,
            )
        elif code == "content_policy_violation":
            raise ContentPolicyError(
                f"Content policy violation: {msg}",
                provider=body.get("provider"),
                response_data=body,
            )
        elif code == "target_authentication_error":
            raise TargetAuthenticationError(
                f"Target API key invalid: {msg}",
                provider=body.get("provider"),
                response_data=body,
            )
        elif code == "service_unavailable":
            raise ServiceUnavailableError(
                f"Service unavailable: {msg}",
                service=body.get("service"),
                retry_after=body.get("retry_after"),
                response_data=body,
            )

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
        elif status_code == 402:  # Payment Required
            raise InsufficientCreditsError(
                f"Payment required: {msg}",
                response_data=body,
            )
        elif status_code == 503:  # Service Unavailable
            raise ServiceUnavailableError(
                f"Service temporarily unavailable: {msg}",
                retry_after=body.get("retry_after"),
                response_data=body,
            )
        elif status_code >= 500:
            raise ServerError(
                f"Server error: {msg}. Please try again later or contact support.",
                response_data=body,
            )
        else:
            raise CompresrError(f"Request failed ({status_code}): {msg}", response_data=body)

    def _is_retryable_status(self, status_code: int) -> bool:
        return status_code in self._retry_config.retry_on_status

    def _attempt_request_sync(
        self,
        method: str,
        endpoint: str,
        body: Optional[bytes],
    ) -> Dict[str, Any]:
        url = self._url(endpoint)
        req = Request(url, data=body, headers=self._headers, method=method)
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
            if "retry_after" not in err:
                header = e.headers.get("Retry-After") if e.headers else None
                if header is not None:
                    try:
                        err["retry_after"] = float(header)
                    except (TypeError, ValueError):
                        pass
            self._handle_error(e.code, err)
        except URLError as e:
            raise CompresrConnectionError(f"Connection failed: {e.reason}")
        except TimeoutError:
            raise CompresrConnectionError("Request timed out")
        except Exception as e:
            raise CompresrError(f"Request failed: {str(e)}")

    def _do_request_sync(
        self,
        method: str,
        endpoint: str,
        body: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        cfg = self._retry_config
        attempt = 0
        while True:
            try:
                return self._attempt_request_sync(method, endpoint, body)
            except (RateLimitError, ServiceUnavailableError) as e:
                if attempt >= cfg.max_retries or not self._is_retryable_status(
                    429 if isinstance(e, RateLimitError) else 503
                ):
                    raise
                delay = compute_backoff(attempt, cfg, retry_after=e.retry_after)
                if delay > 0:
                    time.sleep(delay)
                attempt += 1

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._do_request_sync("POST", endpoint, json.dumps(data).encode("utf-8"))

    def stream(self, endpoint: str, data: Dict[str, Any]) -> Generator[str, None, None]:
        url = self._url(endpoint)
        headers = {**self._headers, "Accept": "text/event-stream"}

        with httpx.Client(timeout=self._timeout) as client:
            with client.stream("POST", url, json=data, headers=headers) as resp:
                if resp.status_code >= 400:
                    try:
                        err = resp.json()
                    except Exception:
                        err = {"error": f"HTTP {resp.status_code}"}
                    self._handle_error(resp.status_code, err)

                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            return
                        try:
                            parsed = json.loads(chunk)
                            if "content" in parsed:
                                yield parsed["content"]
                        except json.JSONDecodeError:
                            if chunk:
                                yield chunk

    async def _attempt_request_async(
        self,
        method: str,
        endpoint: str,
        json_body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        url = self._url(endpoint)
        try:
            resp = await self._async_client.request(method, url, json=json_body)
            body: Dict[str, Any] = resp.json()
            if resp.status_code >= 400:
                if "retry_after" not in body:
                    header = resp.headers.get("Retry-After")
                    if header is not None:
                        try:
                            body["retry_after"] = float(header)
                        except (TypeError, ValueError):
                            pass
                self._handle_error(resp.status_code, body)
            return body
        except httpx.TimeoutException:
            raise CompresrConnectionError("Request timed out")
        except httpx.ConnectError as e:
            raise CompresrConnectionError(f"Connection failed: {str(e)}")

    async def _do_request_async(
        self,
        method: str,
        endpoint: str,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = self._retry_config
        attempt = 0
        while True:
            try:
                return await self._attempt_request_async(method, endpoint, json_body)
            except (RateLimitError, ServiceUnavailableError) as e:
                if attempt >= cfg.max_retries or not self._is_retryable_status(
                    429 if isinstance(e, RateLimitError) else 503
                ):
                    raise
                delay = compute_backoff(attempt, cfg, retry_after=e.retry_after)
                if delay > 0:
                    await asyncio.sleep(delay)
                attempt += 1

    async def post_async(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._do_request_async("POST", endpoint, json_body=data)

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()

    # Kept for back-compat with any caller that pinned the old name.
    async def close(self) -> None:
        """Deprecated alias for :meth:`aclose`."""
        await self.aclose()
