"""Compresr context-compression guardrail for the LiteLLM proxy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

from fastapi import HTTPException
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
    get_metadata_variable_name_from_kwargs,
)
from litellm.types.guardrails import GuardrailEventHooks

from .defaults import DEFAULTS

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

INSTALL_HINT = (
    "The 'compresr' package is required for the Compresr guardrail. "
    "Install it with: pip install 'compresr[litellm]'"
)


class CompresrGuardrailMissingSecrets(Exception):
    """Raised when the Compresr API key is missing."""


class CompresrGuardrailError(Exception):
    """Raised when the Compresr SDK is unavailable (import failure)."""


def _resolve(value: Any, default: Any) -> Any:
    return default if value is None else value


def _content_to_text(content: Any) -> str:
    """Collapse a message ``content`` (str or list-of-parts) to a single string.

    For the multimodal list shape, joins all ``{type:"text", text:...}`` parts
    with blank-line separators. Non-text parts are ignored. Returns "" when the
    content is neither a string nor a list of recognisable parts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


def _replace_text_in_content(content: Any, new_text: str) -> Any:
    """Write ``new_text`` back into a ``content`` value, preserving shape.

    - ``str`` -> returns ``new_text`` directly.
    - ``list`` of parts -> rebuilds the list: the first text part becomes
      ``new_text``, subsequent text parts are dropped, all non-text parts
      pass through in their original order.
    """
    if isinstance(content, str):
        return new_text
    if isinstance(content, list):
        out: List[Any] = []
        replaced = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                if not replaced:
                    out.append({**part, "text": new_text})
                    replaced = True
                continue
            out.append(part)
        if not replaced:
            out.insert(0, {"type": "text", "text": new_text})
        return out
    return new_text


def _cache_key(content: str, query: str, model: str, ratio: float, coarse: Any) -> str:
    raw = f"{model}|{ratio}|{coarse}|{query}|{content}".encode("utf-8")
    return f"compresr:v1:{hashlib.sha256(raw).hexdigest()[:32]}"


class CompresrGuardrail(CustomGuardrail):
    """Query-aware context compression as a LiteLLM proxy guardrail.

    On the ``pre_call`` hook, bulky tool/function outputs are compressed with
    respect to the originating tool call's intent (``name + arguments``,
    resolved via ``tool_call_id``). The user's question is kept verbatim and
    is also the fallback query when no per-target intent can be derived.

    Tool/function outputs are compressed by default. Compressing the system
    prompt, prior user history, and the last user message itself is opt-in.
    Assistant turns are never compressed. Multimodal messages have their text
    parts compressed; non-text parts (images, audio, files) pass through.
    """

    def __init__(
        self,
        guardrail_name: Optional[str] = "compresr",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        compression_model_name: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
        min_chars_to_compress: Optional[int] = None,
        compress_tool_outputs: Optional[bool] = None,
        compress_system: Optional[bool] = None,
        compress_history: Optional[bool] = None,
        compress_last_user: Optional[bool] = None,
        fail_closed: Optional[bool] = None,
        timeout: Optional[float] = None,
        target_ratio_by_role: Optional[Dict[str, float]] = None,
        cache_ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("COMPRESR_API_KEY")
        if self.api_key is None:
            raise CompresrGuardrailMissingSecrets(
                "Couldn't get Compresr API key, either set the `COMPRESR_API_KEY` "
                "in the environment or pass `api_key` to the guardrail in the config file"
            )

        self.api_base = api_base or os.environ.get("COMPRESR_BASE_URL") or DEFAULTS.api_base
        self.compression_model_name = _resolve(compression_model_name, DEFAULTS.compression_model)
        self.target_compression_ratio = _resolve(target_compression_ratio, DEFAULTS.target_ratio)
        self.coarse = _resolve(coarse, DEFAULTS.coarse)
        self.min_chars_to_compress = _resolve(min_chars_to_compress, DEFAULTS.min_chars_to_compress)
        self.compress_tool_outputs = _resolve(compress_tool_outputs, DEFAULTS.compress_tool_outputs)
        self.compress_system = _resolve(compress_system, DEFAULTS.compress_system)
        self.compress_history = _resolve(compress_history, DEFAULTS.compress_history)
        self.compress_last_user = _resolve(compress_last_user, DEFAULTS.compress_last_user)
        self.fail_closed = _resolve(fail_closed, DEFAULTS.fail_closed)
        self.target_ratio_by_role = target_ratio_by_role or dict(
            DEFAULTS.target_ratio_by_role or {}
        )
        self.cache_ttl = _resolve(cache_ttl, DEFAULTS.cache_ttl_seconds)

        if timeout is not None:
            self.timeout = timeout
        else:
            try:
                self.timeout = float(
                    os.environ.get("COMPRESR_TIMEOUT", str(DEFAULTS.timeout_seconds))
                )
            except (ValueError, TypeError):
                verbose_proxy_logger.warning(
                    "Compresr Guardrail: Invalid COMPRESR_TIMEOUT value '%s', "
                    "falling back to default %ss",
                    os.environ.get("COMPRESR_TIMEOUT"),
                    DEFAULTS.timeout_seconds,
                )
                self.timeout = DEFAULTS.timeout_seconds

        self._client: Any = None

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            **kwargs,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from compresr import CompressionClient
        except ImportError as e:
            raise CompresrGuardrailError(INSTALL_HINT) from e

        assert self.api_key is not None
        self._client = CompressionClient(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=int(self.timeout),
        )
        return self._client

    async def aclose(self) -> None:
        """Release the underlying HTTP client (if any).

        Safe to call multiple times. LiteLLM does not currently invoke this
        on proxy shutdown — operators running the guardrail outside the proxy
        should call it explicitly to avoid leaking connections.
        """
        client = self._client
        self._client = None
        if client is None:
            return
        for closer in ("aclose", "close"):
            fn = getattr(client, closer, None)
            if fn is None:
                continue
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:  # pragma: no cover - best-effort cleanup
                verbose_proxy_logger.debug(
                    "Compresr Guardrail: client close raised %s (ignored)", e
                )
            return

    @staticmethod
    def _compresr_exceptions() -> Tuple[type, type, type]:
        from compresr.exceptions import (
            AuthenticationError,
            CompresrError,
            ValidationError,
        )

        return CompresrError, ValidationError, AuthenticationError

    def _resolve_request_config(self, data: dict) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            candidate = metadata.get("guardrail_config")
            if isinstance(candidate, dict):
                overrides = candidate

        def pick(name: str, current: Any) -> Any:
            return overrides[name] if name in overrides else current

        return {
            "compression_model_name": pick("compression_model_name", self.compression_model_name),
            "target_compression_ratio": pick(
                "target_compression_ratio", self.target_compression_ratio
            ),
            "coarse": pick("coarse", self.coarse),
            "min_chars_to_compress": pick("min_chars_to_compress", self.min_chars_to_compress),
            "compress_tool_outputs": pick("compress_tool_outputs", self.compress_tool_outputs),
            "compress_system": pick("compress_system", self.compress_system),
            "compress_history": pick("compress_history", self.compress_history),
            "compress_last_user": pick("compress_last_user", self.compress_last_user),
            "fail_closed": pick("fail_closed", self.fail_closed),
            "target_ratio_by_role": pick("target_ratio_by_role", self.target_ratio_by_role),
        }

    @staticmethod
    def _extract_query(messages: List[dict]) -> Tuple[str, Optional[int]]:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                return _content_to_text(messages[idx].get("content")), idx
        return "", None

    @staticmethod
    def _render_intent(fn: Dict[str, Any]) -> str:
        name = (fn.get("name") or "").strip()
        args = fn.get("arguments")
        if isinstance(args, dict):
            try:
                args = json.dumps(args, separators=(",", ":"))
            except (TypeError, ValueError):
                args = str(args)
        args = (str(args) if args is not None else "").strip()
        if name and args:
            return f"{name}: {args}"
        return name or args

    @classmethod
    def _query_for_target(cls, messages: List[dict], target_idx: int, fallback: str) -> str:
        msg = messages[target_idx]
        role = msg.get("role")
        if role not in ("tool", "function"):
            return fallback

        tool_call_id = msg.get("tool_call_id")
        fn_name = msg.get("name")

        for j in range(target_idx - 1, -1, -1):
            prev = messages[j]
            if prev.get("role") != "assistant":
                continue
            for tc in prev.get("tool_calls") or []:
                if tool_call_id and tc.get("id") == tool_call_id:
                    intent = cls._render_intent(tc.get("function") or {})
                    if intent:
                        return intent
            fc = prev.get("function_call")
            if fc and (not fn_name or fc.get("name") == fn_name):
                intent = cls._render_intent(fc)
                if intent:
                    return intent
        return fallback

    def _select_targets(
        self,
        messages: List[dict],
        query_idx: Optional[int],
        cfg: Dict[str, Any],
    ) -> List[int]:
        """Indices of messages whose text content should be compressed.

        Text content includes both ``str`` content and the joined text of
        multimodal list-of-parts content. Messages with no text content (e.g.
        a tool output that's image-only) are skipped.
        """
        min_chars = cfg["min_chars_to_compress"]
        targets: List[int] = []
        for idx, msg in enumerate(messages):
            if idx == query_idx and not cfg["compress_last_user"]:
                continue
            role = msg.get("role")
            if role in ("tool", "function"):
                if not cfg["compress_tool_outputs"]:
                    continue
            elif role == "system":
                if not cfg["compress_system"]:
                    continue
            elif role == "user":
                if idx != query_idx and not cfg["compress_history"]:
                    continue
            else:
                continue

            text = _content_to_text(msg.get("content"))
            if len(text) < min_chars:
                continue
            targets.append(idx)
        return targets

    def _ratio_for_target(self, role: Any, cfg: Dict[str, Any]) -> float:
        per_role: Dict[str, float] = cfg.get("target_ratio_by_role") or {}
        if isinstance(role, str) and role in per_role:
            return float(per_role[role])
        return float(cfg["target_compression_ratio"])

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages = data.get("messages")
        if not messages:
            return data

        cfg = self._resolve_request_config(data)
        fallback_query, query_idx = self._extract_query(messages)
        targets = self._select_targets(messages, query_idx, cfg)
        if not targets:
            verbose_proxy_logger.debug("Compresr Guardrail: no messages eligible for compression")
            return data

        target_texts = [_content_to_text(messages[i].get("content")) for i in targets]
        queries = [self._query_for_target(messages, i, fallback_query) for i in targets]
        ratios = [self._ratio_for_target(messages[i].get("role"), cfg) for i in targets]

        try:
            results = await self._compress(
                contexts=target_texts,
                queries=queries,
                ratios=ratios,
                cfg=cfg,
                cache=cache,
            )
        except Exception as e:
            return self._handle_error(e, data, cfg)

        self._apply_results(data, messages, targets, results)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    async def _compress(
        self,
        contexts: List[str],
        queries: List[str],
        ratios: List[float],
        cfg: Dict[str, Any],
        cache: Optional[DualCache] = None,
    ) -> List[Dict[str, Any]]:
        if not (len(contexts) == len(queries) == len(ratios)):
            raise ValueError(
                "contexts, queries, and ratios must have equal length "
                f"(got {len(contexts)}, {len(queries)}, {len(ratios)})"
            )

        client = self._get_client()
        model = cfg["compression_model_name"]
        coarse = cfg["coarse"]

        results: List[Optional[Dict[str, Any]]] = [None] * len(contexts)
        cache_keys: List[Optional[str]] = [None] * len(contexts)
        miss_idxs: List[int] = []

        for idx, (ctx, q, r) in enumerate(zip(contexts, queries, ratios)):
            if cache is None:
                miss_idxs.append(idx)
                continue
            key = _cache_key(ctx, q, model, r, coarse)
            cache_keys[idx] = key
            try:
                cached = await cache.async_get_cache(key=key)
            except Exception as e:  # pragma: no cover - cache backend can be flaky
                verbose_proxy_logger.debug("Compresr cache lookup failed: %s", e)
                cached = None
            if isinstance(cached, dict) and "compressed_context" in cached:
                results[idx] = cached
            else:
                miss_idxs.append(idx)

        if miss_idxs:
            miss_results = await self._call_backend(
                client=client,
                contexts=[contexts[i] for i in miss_idxs],
                queries=[queries[i] for i in miss_idxs],
                ratios=[ratios[i] for i in miss_idxs],
                model=model,
                coarse=coarse,
            )
            for slot_idx, res in zip(miss_idxs, miss_results):
                results[slot_idx] = res
                write_key = cache_keys[slot_idx]
                if cache is not None and write_key is not None:
                    try:
                        await cache.async_set_cache(key=write_key, value=res, ttl=self.cache_ttl)
                    except Exception as e:  # pragma: no cover
                        verbose_proxy_logger.debug("Compresr cache write failed: %s", e)

        return [r for r in results if r is not None]

    async def _call_backend(
        self,
        client: Any,
        contexts: List[str],
        queries: List[str],
        ratios: List[float],
        model: str,
        coarse: Any,
    ) -> List[Dict[str, Any]]:
        if not contexts:
            return []

        uniform_ratio = ratios[0] if all(r == ratios[0] for r in ratios) else None

        if uniform_ratio is None:
            coros = [
                client.compress_async(
                    context=ctx,
                    compression_model_name=model,
                    query=q,
                    target_compression_ratio=r,
                    coarse=coarse,
                )
                for ctx, q, r in zip(contexts, queries, ratios)
            ]
            responses = await asyncio.gather(*coros)
            return [self._single_to_result(resp) for resp in responses]

        if len(contexts) == 1:
            resp = await client.compress_async(
                context=contexts[0],
                compression_model_name=model,
                query=queries[0],
                target_compression_ratio=uniform_ratio,
                coarse=coarse,
            )
            return [self._single_to_result(resp)]

        resp = await client.compress_batch_async(
            contexts=contexts,
            queries=queries,
            compression_model_name=model,
            target_compression_ratio=uniform_ratio,
            coarse=coarse,
        )
        return [self._batch_item_to_result(item) for item in resp.data.results]

    @staticmethod
    def _single_to_result(resp: Any) -> Dict[str, Any]:
        data = resp.data
        return {
            "compressed_context": data.compressed_context,
            "tokens_saved": data.tokens_saved,
            "actual_compression_ratio": data.actual_compression_ratio,
            "duration_ms": data.duration_ms,
        }

    @staticmethod
    def _batch_item_to_result(item: Any) -> Dict[str, Any]:
        return {
            "compressed_context": item.compressed_context,
            "tokens_saved": item.tokens_saved,
            "actual_compression_ratio": item.actual_compression_ratio,
            "duration_ms": item.duration_ms,
        }

    def _apply_results(
        self,
        data: dict,
        messages: List[dict],
        targets: List[int],
        results: List[Dict[str, Any]],
    ) -> None:
        total_tokens_saved = 0
        ratios: List[float] = []
        total_duration_ms = 0
        for target_idx, result in zip(targets, results):
            original = messages[target_idx]
            messages[target_idx] = {
                **original,
                "content": _replace_text_in_content(
                    original.get("content"), result["compressed_context"]
                ),
            }
            total_tokens_saved += result.get("tokens_saved", 0) or 0
            if result.get("actual_compression_ratio") is not None:
                ratios.append(result["actual_compression_ratio"])
            total_duration_ms += result.get("duration_ms", 0) or 0

        data["messages"] = messages

        metadata_field = get_metadata_variable_name_from_kwargs(data)
        if not isinstance(data.get(metadata_field), dict):
            data[metadata_field] = {}
        data[metadata_field]["compresr_stats"] = {
            "messages_compressed": len(targets),
            "tokens_saved": total_tokens_saved,
            "compression_ratio": (round(sum(ratios) / len(ratios), 4) if ratios else None),
            "duration_ms": total_duration_ms,
        }

    def _handle_error(self, error: Exception, data: dict, cfg: Dict[str, Any]) -> dict:
        if isinstance(error, (CompresrGuardrailError, CompresrGuardrailMissingSecrets)):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Compresr Guardrail misconfigured",
                    "message": str(error),
                },
            )

        try:
            _CompresrError, ValidationError, AuthenticationError = self._compresr_exceptions()
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Compresr Guardrail misconfigured",
                    "message": INSTALL_HINT,
                },
            )

        if isinstance(error, ValidationError):
            raise HTTPException(
                status_code=400,
                detail={"error": "Compresr validation error", "message": str(error)},
            )
        if isinstance(error, AuthenticationError):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Compresr authentication error",
                    "message": str(error),
                },
            )

        if cfg["fail_closed"]:
            verbose_proxy_logger.warning(
                "Compresr Guardrail: unavailable and fail_closed=true, blocking request"
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Compresr Guardrail Unavailable",
                    "message": (
                        "Context compression service is temporarily unavailable "
                        "and fail_closed is set"
                    ),
                    "original_error": str(error),
                },
            )

        verbose_proxy_logger.warning(
            "Compresr Guardrail: compression unavailable (%s), forwarding original "
            "uncompressed request (fail-open)",
            str(error),
        )
        # Surface fail-open state in the applied-guardrails header so callers
        # can distinguish "didn't fire" from "fired but Compresr was down".
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=f"{self.guardrail_name}:fail_open"
        )
        return data

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from .types import CompresrGuardrailConfigModel

        return CompresrGuardrailConfigModel
