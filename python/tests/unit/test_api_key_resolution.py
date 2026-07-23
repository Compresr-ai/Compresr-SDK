"""Verify CompressionClient falls back to env → credentials file when no api_key
is passed explicitly."""

from __future__ import annotations

import pytest

from compresr import CompressionClient, credentials
from compresr.exceptions import AuthenticationError


@pytest.fixture
def tmp_creds(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPRESR_CREDENTIALS_FILE", str(tmp_path / "creds"))
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    monkeypatch.delenv("COMPRESR_PROFILE", raising=False)
    yield


class TestApiKeyResolution:
    def test_explicit_api_key_wins(self, tmp_creds, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", "cmp_env_test_key_1234")
        credentials.save("cmp_file_test_key_1234", profile="default")
        client = CompressionClient(api_key="cmp_explicit_test_key_1234")
        assert client._api_key == "cmp_explicit_test_key_1234"

    def test_env_used_when_no_arg(self, tmp_creds, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", "cmp_env_test_key_1234")
        client = CompressionClient()
        assert client._api_key == "cmp_env_test_key_1234"

    def test_credentials_file_used_when_no_arg_no_env(self, tmp_creds):
        credentials.save("cmp_file_test_key_1234", profile="default")
        client = CompressionClient()
        assert client._api_key == "cmp_file_test_key_1234"

    def test_raises_when_nothing_configured(self, tmp_creds):
        with pytest.raises(AuthenticationError, match="No API key found"):
            CompressionClient()

    def test_invalid_prefix_still_rejected(self, tmp_creds):
        with pytest.raises(AuthenticationError, match="Invalid API key format"):
            CompressionClient(api_key="wrong_prefix_key")
