"""Pydantic config model for the Compresr LiteLLM guardrail.

All field defaults are sourced from :mod:`defaults` so the guardrail runtime,
the proxy YAML schema, and any docs generated from this model stay in sync.
"""

from typing import Dict, Optional

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from pydantic import BaseModel, Field

from .defaults import DEFAULTS


class CompresrGuardrailConfigModelOptionalParams(BaseModel):
    """Optional parameters for the Compresr guardrail."""

    compression_model_name: Optional[str] = Field(
        default=DEFAULTS.compression_model,
        description=(
            f"Compresr compression model. Defaults to '{DEFAULTS.compression_model}' "
            "(query-specific compression)."
        ),
    )
    target_compression_ratio: Optional[float] = Field(
        default=DEFAULTS.target_ratio,
        description=(
            "Target compression ratio. 0-1 is the fraction of tokens to remove "
            "(0.5 = remove ~50%); a value >1 is an Nx factor (e.g. 4 = ~4x smaller)."
        ),
    )
    coarse: Optional[bool] = Field(
        default=DEFAULTS.coarse,
        description=(
            "Use coarse (paragraph-level) compression instead of token-level. "
            "Faster but less granular. Only applies to query-specific models. "
            f"Defaults to {DEFAULTS.coarse}; set False for token-level granularity."
        ),
    )
    min_chars_to_compress: Optional[int] = Field(
        default=DEFAULTS.min_chars_to_compress,
        description=(
            "Skip compressing any message whose text is shorter than this many characters."
        ),
    )
    compress_tool_outputs: Optional[bool] = Field(
        default=DEFAULTS.compress_tool_outputs,
        description="Compress tool/function result messages (search hits, RAG, API dumps).",
    )
    compress_system: Optional[bool] = Field(
        default=DEFAULTS.compress_system,
        description="Compress system messages. OFF by default.",
    )
    compress_history: Optional[bool] = Field(
        default=DEFAULTS.compress_history,
        description="Compress prior (non-last) user messages. OFF by default.",
    )
    compress_last_user: Optional[bool] = Field(
        default=DEFAULTS.compress_last_user,
        description=(
            "Also replace the last user message content with its compressed form. "
            "The query sent to Compresr is always the original verbatim text."
        ),
    )
    fail_closed: Optional[bool] = Field(
        default=DEFAULTS.fail_closed,
        description=(
            "If True, raise an error when Compresr is unavailable instead of "
            "forwarding the original uncompressed request."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "HTTP timeout in seconds for calls to the Compresr API. "
            f"When unset, falls back to COMPRESR_TIMEOUT env var, then "
            f"{DEFAULTS.timeout_seconds}s."
        ),
    )
    target_ratio_by_role: Optional[Dict[str, float]] = Field(
        default=DEFAULTS.target_ratio_by_role,
        description=(
            "Per-role compression-ratio overrides, e.g. "
            '`{"system": 0.3, "tool": 0.6}`. Roles not listed fall back to '
            "`target_compression_ratio`."
        ),
    )
    cache_ttl: Optional[int] = Field(
        default=DEFAULTS.cache_ttl_seconds,
        description=(
            "Seconds to cache compression results in LiteLLM's DualCache, keyed "
            "by (content, query, model, ratio, coarse). Saves API calls when the "
            "same tool output repeats in an agent loop."
        ),
    )


class CompresrGuardrailConfigModel(
    GuardrailConfigModel[CompresrGuardrailConfigModelOptionalParams]
):
    """Configuration parameters for the Compresr context-compression guardrail."""

    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for Compresr (must start with 'cmp_'). Falls back to the "
            "`COMPRESR_API_KEY` environment variable."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Base URL for the Compresr API. Set this to your internal service URL "
            f"for on-prem deployments. Falls back to `COMPRESR_BASE_URL`, "
            f"then {DEFAULTS.api_base}."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Compresr (context compression)"
