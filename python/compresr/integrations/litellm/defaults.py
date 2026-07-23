"""Single source of truth for Compresr LiteLLM guardrail defaults.

Every default value used by the guardrail (constructor fallback, Pydantic
config-model defaults, docstrings/descriptions) reads from this module.
Change a value here and it propagates everywhere — never edit defaults in
``guardrail.py`` or ``types.py`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, Optional


@dataclass(frozen=True)
class CompresrDefaults:
    """Frozen container holding every guardrail default in one place."""

    # ── Transport ──
    api_base: str = "https://api.compresr.ai"
    timeout_seconds: float = 10.0
    cache_ttl_seconds: int = 300

    # ── Compression model / strength ──
    compression_model: str = "latte_v2"
    target_ratio: float = 0.5
    coarse: bool = True
    min_chars_to_compress: int = 500

    # ── Per-message targeting ──
    compress_tool_outputs: bool = True
    compress_system: bool = False
    compress_history: bool = False
    compress_last_user: bool = False

    # ── Per-role ratio overrides ──
    target_ratio_by_role: Optional[Dict[str, float]] = None

    # ── Failure policy ──
    fail_closed: bool = False


DEFAULTS: Final[CompresrDefaults] = CompresrDefaults()


__all__ = ["CompresrDefaults", "DEFAULTS"]
