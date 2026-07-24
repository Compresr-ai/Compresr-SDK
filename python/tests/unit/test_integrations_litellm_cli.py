"""Tests for the ``compresr-litellm`` CLI wrapper.

Verifies that ``register`` injects entries into litellm's in-memory registries
without touching site-packages, and that ``main`` delegates to litellm's
own click command.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

from compresr.integrations.litellm import cli  # noqa: E402
from compresr.integrations.litellm import (  # noqa: E402
    CompresrGuardrail,
    initialize_guardrail,
)


def test_register_populates_litellm_registries(monkeypatch):
    """`register` is idempotent and writes both registries directly."""
    from litellm.proxy.guardrails import guardrail_registry

    monkeypatch.setitem(guardrail_registry.guardrail_initializer_registry, "compresr", None)
    monkeypatch.setitem(guardrail_registry.guardrail_class_registry, "compresr", None)

    cli.register()

    assert guardrail_registry.guardrail_initializer_registry["compresr"] is initialize_guardrail
    assert guardrail_registry.guardrail_class_registry["compresr"] is CompresrGuardrail

    # Second call: still pointing at the same callables (no breakage).
    cli.register()
    assert guardrail_registry.guardrail_initializer_registry["compresr"] is initialize_guardrail


def test_main_calls_register_then_run_server():
    """`main` registers first, then hands off to litellm's CLI."""
    call_order = []

    def fake_register():
        call_order.append("register")

    def fake_run_server():
        call_order.append("run_server")

    with patch.object(cli, "register", side_effect=fake_register):
        with patch("litellm.proxy.proxy_cli.run_server", new=fake_run_server):
            cli.main()

    assert call_order == ["register", "run_server"]
