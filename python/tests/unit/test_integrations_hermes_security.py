"""Unit tests for compresr.integrations.hermes._security."""

from compresr.integrations.hermes._security import (
    DEFAULT_BASE_URL,
    resolve_base_url,
    sanitize_secret,
    secure_base_url,
)


class TestSanitizeSecret:
    def test_strips_whitespace(self):
        assert sanitize_secret("  cmp_abc  ", "KEY") == "cmp_abc"

    def test_rejects_crlf(self):
        assert sanitize_secret("cmp_abc\r\nX: y", "KEY") == ""

    def test_rejects_nul(self):
        assert sanitize_secret("cmp_abc\x00", "KEY") == ""

    def test_empty_returns_empty(self):
        assert sanitize_secret("", "KEY") == ""


class TestSecureBaseUrl:
    def test_https_public_host_accepted(self):
        assert secure_base_url("https://api.compresr.ai") == "https://api.compresr.ai"

    def test_http_public_host_rejected(self):
        assert secure_base_url("http://api.compresr.ai") == DEFAULT_BASE_URL

    def test_http_localhost_accepted(self):
        assert secure_base_url("http://localhost:8000") == "http://localhost:8000"

    def test_metadata_host_rejected(self):
        assert secure_base_url("https://metadata.google.internal") == DEFAULT_BASE_URL

    def test_numeric_shorthand_host_rejected(self):
        assert secure_base_url("https://2852039166") == DEFAULT_BASE_URL

    def test_private_ip_literal_rejected(self):
        assert secure_base_url("https://10.0.0.5") == DEFAULT_BASE_URL

    def test_link_local_ip_rejected(self):
        assert secure_base_url("https://169.254.169.254") == DEFAULT_BASE_URL

    def test_noncanonical_loopback_rejected(self):
        assert secure_base_url("https://127.0.0.2") == DEFAULT_BASE_URL
        assert secure_base_url("https://127.1") == DEFAULT_BASE_URL
        assert secure_base_url("https://[0:0:0:0:0:0:0:1]") == DEFAULT_BASE_URL

    def test_canonical_localhost_still_allowed_over_https(self):
        assert secure_base_url("https://127.0.0.1:8443") == "https://127.0.0.1:8443"
        assert secure_base_url("https://[::1]:8443") == "https://[::1]:8443"

    def test_garbage_rejected(self):
        assert secure_base_url("not a url") == DEFAULT_BASE_URL

    def test_custom_default(self):
        assert secure_base_url("ftp://x", default="https://d") == "https://d"


class TestResolveBaseUrl:
    def test_none_and_blank_pass_through(self):
        assert resolve_base_url(None) is None
        assert resolve_base_url("") is None
        assert resolve_base_url("   ") is None

    def test_strips_trailing_api_segment(self):
        assert resolve_base_url("https://api.compresr.ai/api") == "https://api.compresr.ai"

    def test_strips_trailing_slash_then_api(self):
        assert resolve_base_url("https://api.compresr.ai/api/") == "https://api.compresr.ai"

    def test_valid_url_without_api_suffix_unchanged(self):
        assert resolve_base_url("https://staging.compresr.ai") == "https://staging.compresr.ai"

    def test_insecure_url_replaced_with_default(self):
        assert resolve_base_url("http://evil.example.com") == DEFAULT_BASE_URL
