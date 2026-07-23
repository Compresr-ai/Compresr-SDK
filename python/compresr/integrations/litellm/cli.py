"""``compresr-litellm`` — stock ``litellm`` CLI with the Compresr guardrail pre-registered."""

from __future__ import annotations

import sys


def register() -> None:
    """Inject the Compresr guardrail into LiteLLM's in-memory registries."""
    try:
        from litellm.proxy.guardrails.guardrail_registry import (
            guardrail_class_registry,
            guardrail_initializer_registry,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "compresr-litellm requires the LiteLLM proxy. "
            "Install with: pip install 'compresr[litellm]'"
        ) from exc

    from . import CompresrGuardrail, initialize_guardrail

    guardrail_initializer_registry["compresr"] = initialize_guardrail
    guardrail_class_registry["compresr"] = CompresrGuardrail


def main() -> None:
    register()
    try:
        from litellm.proxy.proxy_cli import run_server
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "compresr-litellm requires the LiteLLM proxy. "
            "Install with: pip install 'compresr[litellm]'"
        ) from exc
    run_server()


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
