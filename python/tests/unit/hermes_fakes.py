"""Fake Hermes runtime modules for unit-testing compresr.integrations.hermes."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

SUMMARY_PREFIX = "[CONTEXT SUMMARY]\n"


class FakeContextCompressor:
    """Minimal stand-in for hermes agent.context_compressor.ContextCompressor,
    covering the surface CompresrContextEngine inherits and calls."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = kwargs.get("model", "")
        self.context_length = kwargs.get("config_context_length", 0)
        self.threshold_percent = kwargs.get("threshold_percent", 80.0)
        self.protect_first_n = kwargs.get("protect_first_n", 3)
        self.protect_last_n = kwargs.get("protect_last_n", 20)
        self.summary_target_ratio = kwargs.get("summary_target_ratio", 0.2)
        self.abort_on_summary_failure = kwargs.get("abort_on_summary_failure", False)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.compression_count = 0
        self._previous_summary = ""
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error: Optional[str] = None
        self.update_model_calls: List[tuple] = []

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        return "\n".join(str(t.get("content", "")) for t in turns)

    def _record_compression_failure_cooldown(self, seconds: float, error: str) -> None:
        self._summary_failure_cooldown_until = time.monotonic() + seconds
        self._recorded_cooldown = (seconds, error)

    def _clear_compression_failure_cooldown(self) -> None:
        self._summary_failure_cooldown_until = 0.0

    def _strip_summary_prefix(self, text: str) -> str:
        if text.startswith(SUMMARY_PREFIX):
            return text[len(SUMMARY_PREFIX) :]
        return text

    def _with_summary_prefix(self, body: str) -> str:
        return SUMMARY_PREFIX + body

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: Any = "",
        provider: str = "",
        api_mode: str = "",
        max_tokens: Optional[int] = None,
    ) -> None:
        self.update_model_calls.append(
            (model, context_length, base_url, api_key, provider, api_mode, max_tokens)
        )
        self.model = model
        self.context_length = context_length

    def get_status(self) -> Dict[str, Any]:
        return {"model": self.model, "context_length": self.context_length}


class LocalEnvironment:
    """Class name must match what cache._agent_visible_cache_path checks."""


FakeLocalEnvironment = LocalEnvironment


def install(
    monkeypatch,
    hermes_home: Path,
    active_env: Any = None,
    redact=None,
    max_line_length: int = 2000,
) -> None:
    """Install fake hermes modules into sys.modules for one test."""
    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []
    ctx_mod = types.ModuleType("agent.context_compressor")
    ctx_mod.ContextCompressor = FakeContextCompressor
    engine_mod = types.ModuleType("agent.context_engine")
    engine_mod.sanitize_memory_context = lambda v: str(v) if v else ""
    redact_mod = types.ModuleType("agent.redact")
    redact_mod.redact_sensitive_text = redact or (lambda s: s)
    agent_pkg.context_compressor = ctx_mod
    agent_pkg.context_engine = engine_mod
    agent_pkg.redact = redact_mod

    constants_mod = types.ModuleType("hermes_constants")
    constants_mod.get_hermes_home = lambda: hermes_home

    tools_pkg = types.ModuleType("tools")
    tools_pkg.__path__ = []
    terminal_mod = types.ModuleType("tools.terminal_tool")
    terminal_mod.get_active_env = lambda task_id: active_env
    cred_mod = types.ModuleType("tools.credential_files")
    cred_mod.map_cache_path_to_container = lambda p, container_base: (
        f"{container_base}/cache/compresr/tool-output/{Path(p).name}"
    )
    limits_mod = types.ModuleType("tools.tool_output_limits")
    limits_mod.get_max_line_length = lambda: max_line_length
    tools_pkg.terminal_tool = terminal_mod
    tools_pkg.credential_files = cred_mod
    tools_pkg.tool_output_limits = limits_mod

    for name, mod in {
        "agent": agent_pkg,
        "agent.context_compressor": ctx_mod,
        "agent.context_engine": engine_mod,
        "agent.redact": redact_mod,
        "hermes_constants": constants_mod,
        "tools": tools_pkg,
        "tools.terminal_tool": terminal_mod,
        "tools.credential_files": cred_mod,
        "tools.tool_output_limits": limits_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
