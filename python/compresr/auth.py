"""Browser-based login: opens the consent page, receives the key on a
loopback callback, stores it in ``~/.compresr/credentials``."""

from __future__ import annotations

import html
import http.server
import os
import re
import secrets
import sys
import threading
import time
import webbrowser
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

from .credentials import DEFAULT_PROFILE, clear, load, save

_PORT_RANGE = range(9876, 9886)

_ALLOWED_APP_HOSTS = frozenset({"compresr.ai", "www.compresr.ai", "staging.compresr.ai"})
_ALLOWED_BASE_HOSTS = frozenset({"api.compresr.ai", "api-staging.compresr.ai"})
_DEV_HOST_ALLOWED_ENV = "COMPRESR_ALLOW_LOCAL_URLS"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_ERROR_CODE_WHITELIST = frozenset(
    {
        "state_mismatch",
        "no_token",
        "access_denied",
        "server_error",
        "invalid_request",
        "unauthorized_client",
        "temporarily_unavailable",
    }
)

_STATE_LEN = 64
_PROGRESS_INTERVAL_S = 10.0
_POLL_INTERVAL_S = 0.15
_MAX_TIMEOUT_S = 600.0


def _default_app_url() -> str:
    return os.environ.get("COMPRESR_APP_URL", "https://compresr.ai")


def _default_base_url() -> str:
    return os.environ.get("COMPRESR_BASE_URL", "https://api.compresr.ai")


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def _validate_url(url: str, allowed_hosts: frozenset, kind: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise RuntimeError(f"Invalid {kind}: {e}") from None
    if parsed.username or parsed.password:
        raise RuntimeError(f"Refusing {kind} {url!r}: URLs with userinfo are not permitted.")
    if parsed.query or parsed.fragment:
        raise RuntimeError(f"Refusing {kind} {url!r}: query/fragment not permitted here.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if _is_loopback(host):
        return url
    if os.environ.get(_DEV_HOST_ALLOWED_ENV) == "1":
        return url
    if parsed.scheme != "https":
        raise RuntimeError(
            f"Refusing {kind} {url!r}: scheme must be https. "
            f"Set {_DEV_HOST_ALLOWED_ENV}=1 to allow non-loopback overrides."
        )
    if host not in allowed_hosts:
        raise RuntimeError(
            f"Refusing {kind} {url!r}: host {host!r} not in allow-list "
            f"{sorted(allowed_hosts)}. Set {_DEV_HOST_ALLOWED_ENV}=1 to override."
        )
    return url


def _sanitize_error_code(raw: Optional[str]) -> str:
    if not raw:
        return "unknown_error"
    if raw in _ERROR_CODE_WHITELIST:
        return raw
    if re.fullmatch(r"[A-Za-z0-9_\-]{1,32}", raw):
        return raw
    return "unknown_error"


class _CallbackResult:
    __slots__ = ("token", "state", "error")

    def __init__(self) -> None:
        self.token: Optional[str] = None
        self.state: Optional[str] = None
        self.error: Optional[str] = None


_LANDING_PAGE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
               Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  display: flex; align-items: center; justify-content: center;
  min-height: 100vh; padding: 24px;
  background: radial-gradient(1200px 600px at 50% 0%, #eef2ff 0%, #ffffff 60%);
  color: #0f172a;
}
.card {
  max-width: 420px; width: 100%; background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.08); border-radius: 16px;
  box-shadow: 0 20px 40px -20px rgba(15, 23, 42, 0.15);
  padding: 32px 28px; text-align: center;
}
.badge {
  width: 64px; height: 64px; border-radius: 999px;
  display: inline-flex; align-items: center; justify-content: center;
  margin-bottom: 20px;
}
.badge svg { width: 32px; height: 32px; }
.badge-ok { background: rgba(16, 185, 129, 0.12); color: #059669; }
.badge-err { background: rgba(239, 68, 68, 0.12); color: #dc2626; }
h1 { font-size: 20px; font-weight: 600; margin-bottom: 8px; letter-spacing: -0.01em; }
p { font-size: 14px; color: #475569; line-height: 1.55; }
p + p { margin-top: 12px; }
.brand { margin-top: 24px; font-size: 12px; color: #94a3b8; letter-spacing: 0.05em; }
kbd {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
  padding: 2px 6px; border-radius: 4px; background: #f1f5f9; color: #0f172a;
  border: 1px solid rgba(15, 23, 42, 0.08);
}
@media (prefers-color-scheme: dark) {
  body { background: radial-gradient(1200px 600px at 50% 0%, #0b1220 0%, #030712 60%); color: #e2e8f0; }
  .card { background: #0f172a; border-color: rgba(255,255,255,0.08); box-shadow: 0 20px 40px -20px rgba(0,0,0,0.5); }
  p { color: #94a3b8; }
  kbd { background: #1e293b; color: #e2e8f0; border-color: rgba(255,255,255,0.08); }
  .brand { color: #64748b; }
}
"""

_CHECK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M20 6L9 17l-5-5"/></svg>'
)
_X_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M18 6L6 18M6 6l12 12"/></svg>'
)


def _landing_html(*, ok: bool, title: str, body: str) -> str:
    # `body` is trusted HTML — callers must escape any reflected input first.
    badge_class = "badge-ok" if ok else "badge-err"
    svg = _CHECK_SVG if ok else _X_SVG
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)} — Compresr</title>"
        f"<style>{_LANDING_PAGE_CSS}</style></head><body>"
        f'<main class="card">'
        f'  <div class="badge {badge_class}">{svg}</div>'
        f"  <h1>{html.escape(title)}</h1>"
        f"  {body}"
        f'  <div class="brand">compresr</div>'
        f"</main></body></html>"
    )


def _make_handler(
    result: _CallbackResult, expected_state: str, port: int
) -> type[http.server.BaseHTTPRequestHandler]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        # One-shot: subsequent requests get 410 so a stale tab can't overwrite.
        _consumed = False

        def log_message(self, *_args: object, **_kwargs: object) -> None:
            return

        def _hostile_origin(self) -> bool:
            host = self.headers.get("Host", "")
            if host != f"127.0.0.1:{port}":
                return True
            sec_site = self.headers.get("Sec-Fetch-Site")
            if sec_site is None or sec_site in {"none", "same-origin"}:
                return False
            mode = self.headers.get("Sec-Fetch-Mode", "")
            dest = self.headers.get("Sec-Fetch-Dest", "")
            return mode != "navigate" or dest != "document"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/cb":
                self._respond_text(404, "not found")
                return
            if self._hostile_origin():
                self._respond_text(403, "forbidden")
                return
            if _Handler._consumed:
                self._respond_text(410, "gone")
                return

            params = parse_qs(parsed.query)
            error = params.get("error", [None])[0]
            state = params.get("state", [None])[0]
            token = params.get("token", [None])[0]

            # State first — bad state must not set result.error, otherwise any
            # tab could DoS active logins with random /cb?state=x.
            if (
                state is None
                or len(state) != _STATE_LEN
                or not secrets.compare_digest(state, expected_state)
            ):
                self._respond_html(
                    400,
                    "State parameter mismatch",
                    (
                        "<p>The response didn't match this login session — "
                        "possibly a stale tab or a replayed URL.</p>"
                        "<p>Retry <kbd>compresr-sdk login</kbd> from your terminal.</p>"
                    ),
                    ok=False,
                )
                return

            _Handler._consumed = True
            if error:
                code = _sanitize_error_code(error)
                result.error = code
                safe = html.escape(code, quote=True)
                self._respond_html(
                    400,
                    "Authorization failed",
                    (
                        f"<p>The provider responded with <kbd>{safe}</kbd>.</p>"
                        "<p>Return to your terminal and retry "
                        "<kbd>compresr-sdk login</kbd>.</p>"
                    ),
                    ok=False,
                )
                return
            if not token:
                result.error = "no_token"
                self._respond_html(
                    400,
                    "No token received",
                    "<p>The authorization completed but no token came back.</p>",
                    ok=False,
                )
                return

            result.token = token
            result.state = state
            self._respond_html(
                200,
                "You're logged in",
                (
                    "<p>Your API key was delivered to the SDK.</p>"
                    "<p>You can close this tab and return to your terminal.</p>"
                ),
                ok=True,
            )

        def _security_headers(self) -> None:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'",
            )
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")

        def _respond_html(self, status: int, title: str, body: str, *, ok: bool) -> None:
            self._respond(
                status,
                _landing_html(ok=ok, title=title, body=body).encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )

        def _respond_text(self, status: int, text: str) -> None:
            self._respond(
                status,
                text.encode("utf-8"),
                content_type="text/plain; charset=utf-8",
            )

        def _respond(self, status: int, body: bytes, *, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return _Handler


def _bind_server(result: _CallbackResult, state: str) -> tuple[http.server.HTTPServer, int]:
    last_err: Optional[OSError] = None
    for port in _PORT_RANGE:
        handler_cls = _make_handler(result, state, port)
        try:
            server = http.server.HTTPServer(("127.0.0.1", port), handler_cls)
            return server, port
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(f"No free port in {_PORT_RANGE.start}-{_PORT_RANGE.stop - 1}: {last_err}")


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def login(
    *,
    app_url: Optional[str] = None,
    base_url: Optional[str] = None,
    profile: str = DEFAULT_PROFILE,
    timeout: float = 120.0,
    open_browser: bool = True,
) -> str:
    """Interactive browser login. Returns the newly-stored API key."""
    if not (0 < timeout <= _MAX_TIMEOUT_S):
        raise ValueError(f"timeout must be in (0, {_MAX_TIMEOUT_S}] seconds (got {timeout})")

    raw_app = (app_url or _default_app_url()).rstrip("/")
    raw_base = (base_url or _default_base_url()).rstrip("/")
    app_url = _validate_url(raw_app, _ALLOWED_APP_HOSTS, "app_url")
    base_url = _validate_url(raw_base, _ALLOWED_BASE_HOSTS, "base_url")

    state = secrets.token_hex(32)
    result = _CallbackResult()
    server, port = _bind_server(result, state)
    callback = f"http://127.0.0.1:{port}/cb"
    auth_url = f"{app_url}/authorize?{urlencode({'state': state, 'callback': callback})}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    browser_opened = False
    if open_browser:
        try:
            browser_opened = bool(webbrowser.open(auth_url, new=1, autoraise=True))
        except webbrowser.Error:
            browser_opened = False

    if browser_opened:
        _stderr(f"Opened browser to {auth_url}")
    else:
        _stderr(
            "Could not open a browser automatically. Copy this URL and open it manually:\n"
            f"    {auth_url}"
        )

    deadline = time.monotonic() + timeout
    last_progress = time.monotonic()
    try:
        while time.monotonic() < deadline:
            if result.token or result.error:
                break
            now = time.monotonic()
            if now - last_progress >= _PROGRESS_INTERVAL_S:
                remaining = int(deadline - now)
                _stderr(f"Waiting for browser callback… ({remaining}s remaining, Ctrl-C to abort)")
                last_progress = now
            time.sleep(_POLL_INTERVAL_S)
    finally:
        # server.shutdown may raise during Ctrl-C; suppress so server_close
        # still runs and the port is released.
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass
        thread.join(timeout=2.0)

    if result.error:
        code = _sanitize_error_code(result.error)
        raise RuntimeError(f"Login failed: {code}")

    token = result.token
    result.token = None  # scrub before we return or raise
    if not token:
        raise TimeoutError(f"Login timed out after {timeout:g}s. Retry `compresr-sdk login`.")

    path = save(token, profile=profile, base_url=base_url)
    print(f"Saved credentials to {path} (profile: {profile})")
    return token


def _revoke_server_key(api_key: str, base_url: str, timeout: float = 5.0) -> bool:
    """POST /api/authorize/logout with the API key. Best-effort — never raises."""
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/api/authorize/logout"
    req = urllib.request.Request(
        url,
        data=b"",
        method="POST",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            return 200 <= status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def logout(
    profile: str = DEFAULT_PROFILE,
    *,
    revoke_server_key: bool = True,
    base_url: Optional[str] = None,
) -> bool:
    """Remove stored credentials for ``profile`` and revoke the server-side key.

    ``revoke_server_key=True`` (default) makes a best-effort ``POST
    /api/authorize/logout`` before deleting local credentials — this closes the
    "logged out locally but key is still ACTIVE on the server" gap. Network
    failures do not block the local clear. Set ``revoke_server_key=False`` to
    skip the server call entirely.
    """
    if revoke_server_key:
        try:
            key = load(profile)
        except Exception:
            key = None
        if key:
            _revoke_server_key(key, base_url or _default_base_url())
    return clear(profile)


__all__ = ["login", "logout"]
