"""Hermes plugin entry point: wires the tool-output hook, the context engine,
and the /compresr stats command. Everything is opt-in and fail-open."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .tool_output import ToolOutputCompressor

logger = logging.getLogger(__name__)

# Secrets this plugin needs, so a Hermes host can prompt for them when the
# plugin is enabled (mirrors hermes-plugin/plugin.yaml's ``requires_env``).
REQUIRES_ENV = [
    {
        "name": "COMPRESR_API_KEY",
        "description": "Compresr API key (starts with cmp_)",
        "url": "https://compresr.ai",
        "secret": True,
    }
]


def register(ctx: Any) -> None:
    from . import cache

    try:
        cache.ensure_cache_root()
    except Exception as e:
        logger.warning("compresr: could not initialize cache root: %s", e)

    # Mount/translate the cache dir so recovery works on non-Local backends.
    register_dir = getattr(ctx, "register_cache_dir", None)
    if callable(register_dir):
        try:
            register_dir(cache.cache_relpath())
        except Exception as e:
            logger.warning("compresr: register_cache_dir failed: %s", e)

    compressor = ToolOutputCompressor()
    ctx.register_hook("transform_tool_result", compressor.on_transform_tool_result)
    if not compressor.api_key:
        logger.info(
            "compresr: tool-output hook loaded but no API key found — inactive. "
            "Set COMPRESR_API_KEY in ~/.hermes/.env or run `compresr-sdk login`."
        )
    elif not compressor.enabled:
        logger.info(
            "compresr: tool-output hook loaded but disabled — set "
            "compresr.tool_output_enabled: true in config.yaml."
        )

    engine = _register_context_engine(ctx)
    _register_status_command(ctx, compressor, engine)


def _register_context_engine(ctx: Any) -> Optional[Any]:
    if not hasattr(ctx, "register_context_engine"):
        logger.debug("compresr: host has no register_context_engine — skipping engine")
        return None
    try:
        from .engine import CompresrContextEngine

        engine = CompresrContextEngine()
    except Exception as e:
        logger.warning("compresr: context engine unavailable (%s)", e)
        return None
    # Without a key, registering would end sessions in a provider-side
    # context-limit error under abort_on_summary_failure=True.
    if not engine.is_available():
        logger.info(
            "compresr: context engine not registered — no API key found. "
            "Set COMPRESR_API_KEY and select it with context.engine: compresr."
        )
        return None
    ctx.register_context_engine(engine)
    return engine


def _register_status_command(
    ctx: Any, compressor: ToolOutputCompressor, engine: Optional[Any]
) -> None:
    if not hasattr(ctx, "register_command"):
        return

    def _status(raw_args: str = "") -> str:
        status: dict = {"tool_output": compressor.get_status()}
        if engine is not None:
            try:
                status["context_engine"] = engine.get_status()
            except Exception as e:
                status["context_engine"] = {"error": str(e)}
        return json.dumps(status, indent=2, default=str)

    try:
        ctx.register_command(
            "compresr",
            _status,
            description="Show Compresr compression stats (tool output + context engine)",
        )
    except Exception as e:
        logger.debug("compresr: could not register /compresr command: %s", e)


__all__ = ["register"]
