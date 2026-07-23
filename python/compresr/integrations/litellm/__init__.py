"""LiteLLM integration for Compresr (proxy guardrail)."""

from __future__ import annotations

import os

try:
    import litellm  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "compresr LiteLLM integration requires litellm. "
        "Install with: pip install compresr[litellm]"
    ) from exc

from .defaults import DEFAULTS, CompresrDefaults
from .guardrail import (
    CompresrGuardrail,
    CompresrGuardrailError,
    CompresrGuardrailMissingSecrets,
)
from .initializer import (
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from .types import (
    CompresrGuardrailConfigModel,
    CompresrGuardrailConfigModelOptionalParams,
)


def _maybe_auto_install_shim() -> None:
    if os.environ.get("COMPRESR_AUTO_INSTALL_SHIM", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from .shim_installer import install

        install(verbose=False)
    except Exception:  # pragma: no cover - best-effort
        pass


_maybe_auto_install_shim()


__all__ = [
    "DEFAULTS",
    "CompresrDefaults",
    "CompresrGuardrail",
    "CompresrGuardrailConfigModel",
    "CompresrGuardrailConfigModelOptionalParams",
    "CompresrGuardrailError",
    "CompresrGuardrailMissingSecrets",
    "guardrail_class_registry",
    "guardrail_initializer_registry",
    "initialize_guardrail",
]
