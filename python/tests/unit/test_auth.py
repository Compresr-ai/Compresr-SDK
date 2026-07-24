"""Unit tests for compresr.auth — browser login flow.

Real browser interaction is impossible in CI, so these tests drive the local
loopback callback directly with ``urllib`` while ``login()`` waits.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import pytest

from compresr import auth, credentials

# Loopback URL is always accepted by the app_url validator (dev/test escape
# hatch), so tests don't need to touch COMPRESR_ALLOW_LOCAL_URLS or invent a
# fake HTTPS origin.
_TEST_APP_URL = "http://127.0.0.1:12345"
_TEST_BASE_URL = "http://127.0.0.1:54321"


@pytest.fixture
def tmp_creds(tmp_path, monkeypatch):
    path = tmp_path / "creds"
    monkeypatch.setenv("COMPRESR_CREDENTIALS_FILE", str(path))
    yield path


def _hit_callback(
    port: int,
    *,
    token: Optional[str],
    state: Optional[str],
    error: Optional[str] = None,
    path: str = "/cb",
) -> int:
    params: dict[str, str] = {}
    if error is not None:
        params["error"] = error
    if token is not None:
        params["token"] = token
    if state is not None:
        params["state"] = state
    url = f"http://127.0.0.1:{port}{path}?{urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def _run_login_in_thread(result_slot: dict, **kwargs) -> threading.Thread:
    def target():
        try:
            result_slot["value"] = auth.login(**kwargs)
        except BaseException as e:
            result_slot["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def _peek_callback_port(started_url_slot: dict, monkeypatch) -> None:
    """Monkeypatch webbrowser.open so the test learns the auth URL + port.

    Accepts *args/**kwargs because login() now calls open(url, new=1,
    autoraise=True) — a fixed-arity mock would raise TypeError.
    """
    import webbrowser

    def capture(url, *_args, **_kwargs):
        started_url_slot["url"] = url
        return True

    monkeypatch.setattr(webbrowser, "open", capture)


def _wait_for_url(slot: dict, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while "url" not in slot and time.time() < deadline:
        time.sleep(0.05)
    assert "url" in slot, "login() never opened a browser URL"


def _port_from_url(url: str) -> int:
    """Extract the loopback callback port from the /authorize URL query."""
    q = parse_qs(urlparse(url).query)
    callback = q.get("callback", [""])[0]
    return int(callback.rsplit(":", 1)[-1].split("/")[0])


def _state_from_url(url: str) -> str:
    return parse_qs(urlparse(url).query).get("state", [""])[0]


class TestLoginRoundtrip:
    def test_valid_callback_saves_key_and_returns(self, tmp_creds, monkeypatch):
        started: dict = {}
        _peek_callback_port(started, monkeypatch)

        result_slot: dict = {}
        t = _run_login_in_thread(
            result_slot,
            app_url=_TEST_APP_URL,
            base_url=_TEST_BASE_URL,
            timeout=5.0,
        )
        _wait_for_url(started)

        state = _state_from_url(started["url"])
        port = _port_from_url(started["url"])
        assert _hit_callback(port, token="cmp_test_key_test_key_1234", state=state) == 200

        t.join(timeout=5)
        assert result_slot.get("value") == "cmp_test_key_test_key_1234"
        assert credentials.load("default") == "cmp_test_key_test_key_1234"

    def test_state_mismatch_does_not_kill_flow(self, tmp_creds, monkeypatch):
        """Bad state must NOT set result.error — any tab could DoS the login
        otherwise. Should keep waiting and eventually TimeoutError."""
        started: dict = {}
        _peek_callback_port(started, monkeypatch)

        result_slot: dict = {}
        t = _run_login_in_thread(
            result_slot,
            app_url=_TEST_APP_URL,
            base_url=_TEST_BASE_URL,
            timeout=1.5,
        )
        _wait_for_url(started)
        port = _port_from_url(started["url"])

        # Hit /cb with a wrong-length AND wrong-value state — should be
        # rejected without touching result.
        _hit_callback(port, token="cmp_should_not_stick", state="deadbeef" * 4)
        _hit_callback(port, token="cmp_should_not_stick", state="a" * 64)

        t.join(timeout=5)
        assert isinstance(result_slot.get("error"), TimeoutError)
        assert credentials.load("default") is None

    def test_state_matches_error_gets_raised(self, tmp_creds, monkeypatch):
        """Server-side error with correct state → RuntimeError with sanitized code."""
        started: dict = {}
        _peek_callback_port(started, monkeypatch)

        result_slot: dict = {}
        t = _run_login_in_thread(
            result_slot,
            app_url=_TEST_APP_URL,
            base_url=_TEST_BASE_URL,
            timeout=5.0,
        )
        _wait_for_url(started)
        state = _state_from_url(started["url"])
        port = _port_from_url(started["url"])

        _hit_callback(port, token=None, state=state, error="access_denied")

        t.join(timeout=5)
        assert isinstance(result_slot.get("error"), RuntimeError)
        assert "access_denied" in str(result_slot["error"])

    def test_unknown_error_code_is_sanitized(self, tmp_creds, monkeypatch):
        """Attacker-supplied gibberish error code → generic ``unknown_error``."""
        started: dict = {}
        _peek_callback_port(started, monkeypatch)

        result_slot: dict = {}
        t = _run_login_in_thread(
            result_slot,
            app_url=_TEST_APP_URL,
            base_url=_TEST_BASE_URL,
            timeout=5.0,
        )
        _wait_for_url(started)
        state = _state_from_url(started["url"])
        port = _port_from_url(started["url"])

        _hit_callback(port, token=None, state=state, error="<script>alert(1)</script>")

        t.join(timeout=5)
        err = result_slot.get("error")
        assert isinstance(err, RuntimeError)
        assert "unknown_error" in str(err)
        assert "<script>" not in str(err)

    def test_non_cb_path_ignored(self, tmp_creds, monkeypatch):
        """Favicon / .well-known / any non-/cb prefetch must not disturb the
        login state. Login should time out cleanly, not raise state_mismatch."""
        started: dict = {}
        _peek_callback_port(started, monkeypatch)

        result_slot: dict = {}
        t = _run_login_in_thread(
            result_slot,
            app_url=_TEST_APP_URL,
            base_url=_TEST_BASE_URL,
            timeout=1.5,
        )
        _wait_for_url(started)
        port = _port_from_url(started["url"])

        assert _hit_callback(port, token=None, state=None, path="/favicon.ico") == 404
        assert _hit_callback(port, token=None, state=None, path="/.well-known/x") == 404

        t.join(timeout=5)
        assert isinstance(result_slot.get("error"), TimeoutError)

    def test_timeout_raises_timeout_error(self, tmp_creds, monkeypatch):
        started: dict = {}
        _peek_callback_port(started, monkeypatch)

        result_slot: dict = {}
        t = _run_login_in_thread(
            result_slot,
            app_url=_TEST_APP_URL,
            base_url=_TEST_BASE_URL,
            timeout=1.0,
        )

        t.join(timeout=5)
        assert isinstance(result_slot.get("error"), TimeoutError)


class TestUrlValidation:
    def test_rejects_non_https_app_url(self, tmp_creds):
        with pytest.raises(RuntimeError, match="scheme must be https"):
            auth.login(app_url="http://evil.example", base_url=_TEST_BASE_URL, timeout=1.0)

    def test_rejects_off_allowlist_https_app_url(self, tmp_creds):
        with pytest.raises(RuntimeError, match="not in allow-list"):
            auth.login(app_url="https://evil.example", base_url=_TEST_BASE_URL, timeout=1.0)

    def test_rejects_off_allowlist_base_url(self, tmp_creds):
        with pytest.raises(RuntimeError, match="not in allow-list"):
            auth.login(app_url=_TEST_APP_URL, base_url="https://evil.example", timeout=1.0)

    def test_allows_loopback_urls(self, tmp_creds, monkeypatch):
        """Loopback URLs must always pass — dev/test escape hatch."""
        import webbrowser

        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: False)
        # Should NOT raise from the validator; will TimeoutError instead.
        with pytest.raises(TimeoutError):
            auth.login(
                app_url="http://127.0.0.1:8000",
                base_url="http://127.0.0.1:8001",
                timeout=0.5,
            )

    def test_dev_env_var_bypasses_allowlist(self, tmp_creds, monkeypatch):
        import webbrowser

        monkeypatch.setenv("COMPRESR_ALLOW_LOCAL_URLS", "1")
        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: False)
        with pytest.raises(TimeoutError):
            auth.login(
                app_url="https://staging-preview.example",
                base_url="https://api-staging-preview.example",
                timeout=0.5,
            )

    def test_rejects_bad_timeout(self, tmp_creds):
        with pytest.raises(ValueError, match="timeout"):
            auth.login(timeout=0)
        with pytest.raises(ValueError, match="timeout"):
            auth.login(timeout=1e6)

    def test_rejects_dot_localhost_wildcard(self, tmp_creds):
        # foo.localhost was the round-1 bypass — must be rejected now.
        with pytest.raises(RuntimeError, match="not in allow-list"):
            auth.login(
                app_url="https://foo.localhost",
                base_url=_TEST_BASE_URL,
                timeout=1.0,
            )

    def test_rejects_userinfo(self, tmp_creds):
        with pytest.raises(RuntimeError, match="userinfo"):
            auth.login(
                app_url="https://user:pass@compresr.ai",
                base_url=_TEST_BASE_URL,
                timeout=1.0,
            )

    def test_rejects_query_or_fragment(self, tmp_creds):
        with pytest.raises(RuntimeError, match="query/fragment"):
            auth.login(
                app_url="https://compresr.ai?q=1",
                base_url=_TEST_BASE_URL,
                timeout=1.0,
            )
        with pytest.raises(RuntimeError, match="query/fragment"):
            auth.login(
                app_url="https://compresr.ai#x",
                base_url=_TEST_BASE_URL,
                timeout=1.0,
            )

    def test_rejects_off_allowlist_with_trailing_dot(self, tmp_creds):
        # `evil.compresr.ai.` normalizes but must NOT match the allow-list.
        with pytest.raises(RuntimeError, match="not in allow-list"):
            auth.login(
                app_url="https://evil.compresr.ai.",
                base_url=_TEST_BASE_URL,
                timeout=1.0,
            )


class TestSanitization:
    def test_sanitize_error_code_whitelisted(self):
        assert auth._sanitize_error_code("access_denied") == "access_denied"

    def test_sanitize_error_code_short_alphanum_ok(self):
        assert auth._sanitize_error_code("custom_code_9") == "custom_code_9"

    def test_sanitize_error_code_dangerous_replaced(self):
        assert auth._sanitize_error_code("<script>alert(1)</script>") == "unknown_error"

    def test_sanitize_error_code_too_long_replaced(self):
        assert auth._sanitize_error_code("a" * 100) == "unknown_error"

    def test_sanitize_error_code_none_returns_generic(self):
        assert auth._sanitize_error_code(None) == "unknown_error"
        assert auth._sanitize_error_code("") == "unknown_error"


class TestLogout:
    def test_logout_removes_profile(self, tmp_creds):
        credentials.save("cmp_x_XXXXXXXXXXXXXX", profile="default")
        assert auth.logout("default") is True
        assert credentials.load("default") is None

    def test_logout_missing_profile_returns_false(self, tmp_creds):
        assert auth.logout("does-not-exist") is False
