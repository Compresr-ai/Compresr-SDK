"""Shared fixtures for unit tests of the Hermes integration."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from . import hermes_fakes

_HERMES_ENV_VARS = (
    "COMPRESR_API_KEY",
    "COMPRESR_BASE_URL",
    "COMPRESR_MODEL",
    "COMPRESR_TIMEOUT",
    "COMPRESR_COARSE",
    "COMPRESR_DISABLE_PLACEHOLDERS",
    "COMPRESR_TARGET_RATIO",
    "COMPRESR_TOOL_OUTPUT_ENABLED",
    "COMPRESR_TOOL_OUTPUT_MODEL",
    "COMPRESR_TOOL_OUTPUT_MIN_TOKENS",
    "COMPRESR_TOOL_OUTPUT_TIMEOUT",
    "COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB",
    "COMPRESR_TOOL_OUTPUT_TARGET_RATIO",
    "TERMINAL_ENV",
)

_HERMES_INTEGRATION_MODULES = (
    "compresr.integrations.hermes",
    "compresr.integrations.hermes._config",
    "compresr.integrations.hermes._security",
    "compresr.integrations.hermes.cache",
    "compresr.integrations.hermes.engine",
    "compresr.integrations.hermes.plugin",
    "compresr.integrations.hermes.recovery",
    "compresr.integrations.hermes.tool_output",
)


def _purge_integration_modules() -> None:
    for name in _HERMES_INTEGRATION_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture
def hermes_env(monkeypatch, tmp_path):
    """Fresh fake-Hermes environment: fake host modules, isolated hermes home,
    scrubbed env vars, no ambient credentials, purged integration modules."""
    for var in _HERMES_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("compresr.credentials.load", lambda *a, **k: None)

    home = tmp_path / "hermes_home"
    home.mkdir()
    state = SimpleNamespace(home=home, active_env=None, redact=None)

    hermes_fakes.install(monkeypatch, home)
    _purge_integration_modules()

    def load(name: str):
        return importlib.import_module(f"compresr.integrations.hermes.{name}")

    def reinstall(**kwargs):
        hermes_fakes.install(monkeypatch, home, **kwargs)
        _purge_integration_modules()

    state.load = load
    state.reinstall = reinstall
    yield state
    _purge_integration_modules()
