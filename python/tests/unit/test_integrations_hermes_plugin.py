"""Unit tests for the Hermes plugin entry point and the repo-root shim."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

API_KEY = "cmp_" + "p" * 32

SHIM_PATH = Path(__file__).resolve().parents[3] / "hermes-plugin" / "__init__.py"


class FakeCtx:
    def __init__(self, with_engine: bool = True, with_command: bool = True):
        self.hooks: List[tuple] = []
        self.engines: List[Any] = []
        self.commands: Dict[str, Callable] = {}
        if not with_engine:
            self.register_context_engine = None
            del self.register_context_engine
        if not with_command:
            self.register_command = None
            del self.register_command

    def register_hook(self, name: str, callback: Callable) -> None:
        self.hooks.append((name, callback))

    def register_context_engine(self, engine: Any) -> None:
        self.engines.append(engine)

    def register_command(self, name: str, handler: Callable, description: str = "", **kw) -> None:
        self.commands[name] = handler


class MinimalCtx:
    def __init__(self):
        self.hooks: List[tuple] = []

    def register_hook(self, name: str, callback: Callable) -> None:
        self.hooks.append((name, callback))


@pytest.fixture
def plugin(hermes_env):
    return hermes_env.load("plugin")


class TestRegister:
    def test_registers_hook_engine_and_command(self, plugin, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", API_KEY)
        ctx = FakeCtx()
        plugin.register(ctx)
        assert ctx.hooks[0][0] == "transform_tool_result"
        assert len(ctx.engines) == 1
        assert ctx.engines[0].name == "compresr"
        assert "compresr" in ctx.commands

    def test_registers_cache_dir_when_host_supports_it(self, plugin, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", API_KEY)
        ctx = FakeCtx()
        captured: List[str] = []
        ctx.register_cache_dir = lambda p: captured.append(p)
        plugin.register(ctx)
        assert captured == ["cache/compresr/tool-output"]

    def test_register_starts_model_probe(self, plugin, hermes_env, monkeypatch):
        tool_output = hermes_env.load("tool_output")
        started: List[bool] = []
        monkeypatch.setattr(
            tool_output.ToolOutputCompressor,
            "start_model_probe",
            lambda self: started.append(True),
        )
        monkeypatch.setenv("COMPRESR_API_KEY", API_KEY)
        plugin.register(FakeCtx())
        assert started == [True]

    def test_engine_not_registered_without_key(self, plugin):
        ctx = FakeCtx()
        plugin.register(ctx)
        assert ctx.engines == []
        assert ctx.hooks[0][0] == "transform_tool_result"

    def test_minimal_ctx_supported(self, plugin, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", API_KEY)
        ctx = MinimalCtx()
        plugin.register(ctx)
        assert ctx.hooks[0][0] == "transform_tool_result"

    def test_status_command_returns_json(self, plugin, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", API_KEY)
        ctx = FakeCtx()
        plugin.register(ctx)
        status = json.loads(ctx.commands["compresr"](""))
        assert "tool_output" in status
        assert status["context_engine"]["engine"] == "compresr"

    def test_status_command_without_engine(self, plugin):
        ctx = FakeCtx()
        plugin.register(ctx)
        status = json.loads(ctx.commands["compresr"](""))
        assert "context_engine" not in status


class TestShim:
    def _load_shim(self):
        spec = importlib.util.spec_from_file_location("hermes_plugin_shim", SHIM_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_shim_delegates_to_sdk_register(self, hermes_env, monkeypatch):
        monkeypatch.setenv("COMPRESR_API_KEY", API_KEY)
        shim = self._load_shim()
        ctx = FakeCtx()
        shim.register(ctx)
        assert ctx.hooks[0][0] == "transform_tool_result"

    def test_shim_survives_missing_sdk(self, monkeypatch):
        shim = self._load_shim()
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _blocked(name, *args, **kwargs):
            if name.startswith("compresr"):
                raise ImportError("compresr not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _blocked)
        for mod_name in [m for m in sys.modules if m.startswith("compresr")]:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
        ctx = FakeCtx()
        shim.register(ctx)
        assert ctx.hooks == []
