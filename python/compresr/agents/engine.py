"""Provider-agnostic agent engine.

``_Engine`` wraps LangChain's ``init_chat_model`` + ``create_agent`` with
Compresr's ``CompresrToolMiddleware`` and normalizes the final
``AIMessage`` into a provider-shape-free :class:`NormalizedResult`. The
public provider-shaped facades (Anthropic, OpenAI, native) consume this
result and remap it back into provider-native shapes.

The engine is intentionally private (underscored) — facades own the
public surface. Only the normalized result type leaks out.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Sequence, cast

from compresr.exceptions import CompresrError
from compresr.integrations._shared import CompressionPolicy

from .normalized import Citation, CompresrStats, NormalizedResult

logger = logging.getLogger(__name__)


def _import_lc() -> tuple[Any, Any, Any]:
    """Resolve ``init_chat_model``, ``create_agent``, and middleware lazily."""
    try:
        from langchain.agents import create_agent  # type: ignore[import-not-found]
        from langchain.chat_models import init_chat_model  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - install-time guard
        raise ImportError("_Engine requires langchain>=1.0. pip install compresr[agents]") from exc
    from compresr.integrations.langchain.middleware import CompresrToolMiddleware

    return init_chat_model, create_agent, CompresrToolMiddleware


# Re-export the lazy hooks at module level so tests can patch them.
def init_chat_model(*args: Any, **kwargs: Any) -> Any:
    _init, _create, _mw = _import_lc()
    return _init(*args, **kwargs)


def create_agent(*args: Any, **kwargs: Any) -> Any:
    _init, _create, _mw = _import_lc()
    return _create(*args, **kwargs)


def _CompresrToolMiddleware(*args: Any, **kwargs: Any) -> Any:
    _init, _create, _mw = _import_lc()
    return _mw(*args, **kwargs)


_KNOWN_PROVIDERS = {"anthropic", "openai", "google_genai"}


# Detection is by tool name because ``WebSearchTool.__new__`` returns a
# ``StructuredTool`` (not a ``WebSearchTool`` instance), so ``isinstance``
# would always be False.
_WEB_SEARCH_TOOL_NAMES = frozenset({"tavily_search", "brave_search"})


def _is_solo_web_search(tools: Sequence) -> bool:
    """True when ``tools`` is exactly one Compresr-built web-search tool."""
    if len(tools) != 1:
        return False
    name = getattr(tools[0], "name", None)
    return isinstance(name, str) and name in _WEB_SEARCH_TOOL_NAMES


def _extract_last_user_text(messages: Sequence) -> str:
    """Pull the last user/human message text out of a facade-shaped chain.

    Supports the message shapes that the facades feed the engine — plain
    dicts (Anthropic/OpenAI style) and LangChain ``BaseMessage`` instances.
    Falls back to the first message's content when no explicit user message
    is found, and to ``""`` on empty input.
    """
    for msg in reversed(list(messages)):
        role = None
        content: Any = None
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            content = getattr(msg, "content", None)
        if role in ("user", "human") and content is not None:
            return _content_text_any(content)
    if messages:
        first = messages[0]
        content = (
            first.get("content") if isinstance(first, dict) else getattr(first, "content", None)
        )
        return _content_text_any(content)
    return ""


def _content_text_any(content: Any) -> str:
    """Stringify the ``content`` field of a chat message.

    Handles strings, lists of text blocks (Anthropic/OpenAI structured form),
    and arbitrary fallbacks via ``str()``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


# Kwargs that propagate to the underlying chat model via ``.bind(...)`` per
# call. Covers Anthropic / OpenAI / Gemini. LangChain's chat models normalize
# most of these across providers; unsupported keys are silently ignored
# upstream. Keep this conservative — only well-known LLM-level knobs land
# here so unrelated kwargs (e.g. ``config``) don't leak into ``.bind``.
_LLM_BIND_KWARGS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_output_tokens",
        "stop",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "logprobs",
        "top_logprobs",
    }
)

# Constructor-only kwargs — transport, not per-call LLM knobs. Baked into the
# chat model at build time so a custom httpx client (corporate proxy / custom
# CA bundle / verify=False) reaches the provider SDK. Requires langchain-anthropic
# >= the release that adds ``http_client``/``http_async_client`` to ChatAnthropic;
# langchain-openai already supports both.
_LLM_CONSTRUCTOR_KWARGS = frozenset({"http_client", "http_async_client"})


def _parse_llm(llm: str) -> tuple[str, Optional[str]]:
    """Split ``"provider[:model]"`` (or ``"provider[/model]"``) into
    ``(provider, model_or_None)``.

    Both ``:`` and ``/`` are accepted as the separator for parity with
    the TypeScript SDK. A provider-only string (e.g. ``"anthropic"``)
    returns ``(provider, None)`` — the call site is expected to supply
    the model via ``messages.create(model=...)``.

    Raises ``ValueError`` if the input is empty.
    """
    if not llm or not llm.strip():
        raise ValueError(
            "llm must be 'provider' or 'provider:model', "
            "e.g. 'anthropic' or 'anthropic:claude-haiku-4-5'"
        )
    if ":" in llm:
        provider, model = llm.split(":", 1)
        model_clean = model.strip() or None
    elif "/" in llm:
        provider, model = llm.split("/", 1)
        model_clean = model.strip() or None
    else:
        provider, model_clean = llm, None
    return provider.strip(), model_clean


def _content_text(msg: Any) -> str:
    """Join the text parts of an ``AIMessage``'s content.

    Handles both string ``content`` and the list-of-dicts form emitted by
    Anthropic/Responses-style providers.
    """
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _normalize_content_blocks(msg: Any) -> list:
    """Best-effort pass-through of normalized content blocks."""
    blocks = getattr(msg, "content_blocks", None)
    if blocks:
        return list(blocks)
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        return list(content)
    return []


def _normalize_tool_uses(msg: Any) -> list:
    """Map ``AIMessage.tool_calls`` to ``[{id, name, input}]``."""
    tool_calls = getattr(msg, "tool_calls", None) or []
    out: list = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        out.append(
            {
                "id": tc.get("id"),
                "name": tc.get("name"),
                "input": tc.get("args"),
            }
        )
    return out


def _extract_citations(msg: Any) -> list:
    """Walk content blocks for citation-shaped data across providers.

    Recognizes:

    * Anthropic ``web_search_result_location`` citations on text blocks.
    * OpenAI Responses ``url_citation`` annotations.
    * Gemini ``grounding_metadata`` URI hits.

    Unknown shapes are silently skipped — future versions will validate
    against live responses.
    """
    out: list[Citation] = []
    blocks = _normalize_content_blocks(msg)
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for cit in block.get("citations", []) or []:
            if not isinstance(cit, dict):
                continue
            if cit.get("type") in (None, "web_search_result_location"):
                url = cit.get("url")
                if not url:
                    continue
                out.append(
                    Citation(
                        url=url,
                        title=cit.get("title"),
                        cited_text=cit.get("cited_text"),
                        provider_metadata=dict(cit),
                    )
                )
        for ann in block.get("annotations", []) or []:
            if not isinstance(ann, dict):
                continue
            if ann.get("type") == "url_citation":
                url = ann.get("url")
                if not url:
                    continue
                out.append(
                    Citation(
                        url=url,
                        title=ann.get("title"),
                        cited_text=ann.get("text") or ann.get("cited_text"),
                        provider_metadata=dict(ann),
                    )
                )
    response_metadata = getattr(msg, "response_metadata", None) or {}
    grounding = response_metadata.get("grounding_metadata") or {}
    for chunk in grounding.get("grounding_chunks", []) or []:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web") or {}
        url = web.get("uri")
        if not url:
            continue
        out.append(
            Citation(
                url=url,
                title=web.get("title"),
                cited_text=None,
                provider_metadata=dict(chunk),
            )
        )
    return out


def _normalize_stop_reason(msg: Any) -> str:
    response_metadata = getattr(msg, "response_metadata", None) or {}
    return (
        response_metadata.get("stop_reason") or response_metadata.get("finish_reason") or "end_turn"
    )


def _normalize_usage(msg: Any) -> dict:
    """Return the usage block recorded on a single AIMessage.

    Kept as a single-message helper for callers that legitimately want
    per-turn data. The aggregate used by ``_Engine._normalize`` is
    :func:`_aggregate_usage`.
    """
    usage_metadata = getattr(msg, "usage_metadata", None)
    if usage_metadata:
        return dict(usage_metadata)
    response_metadata = getattr(msg, "response_metadata", None) or {}
    return dict(response_metadata.get("usage", {}) or {})


# Total fields surfaced on the normalized usage dict. Provider-specific keys
# (e.g. ``reasoning_tokens``) are preserved as they appear on the AIMessage.
_USAGE_NUMERIC_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

# Provider-specific cache-read aliases under ``input_token_details``.
_INPUT_DETAILS_CACHE_READ_ALIASES = (
    "cached_tokens",
    "cache_read",
    "priority_cache_read",
    "flex_cache_read",
    "cached_content_token_count",
)


def _aggregate_usage(messages: list) -> dict:
    """Sum ``usage_metadata`` across every AIMessage in the conversation.

    Each agent turn produces one AIMessage with its own usage block. Taking
    only the last turn (as the previous implementation did) under-counts any
    multi-turn loop — tool use, ReAct, etc. — by the number of LLM round-trips
    minus one. The benchmark probe on a 12-search question showed 9 AIMessages
    and a 5× discrepancy between summed and last-turn input tokens, so the
    fix is to walk every AIMessage and add.

    LangChain reports cache tokens in two places: top-level
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` (newer
    versions) and nested ``input_token_details.cache_read`` /
    ``cache_creation`` (current shape). Both are folded in.
    """
    total: dict[str, int] = {k: 0 for k in _USAGE_NUMERIC_KEYS}
    extras: dict[str, int] = {}
    ai_message_count = 0
    for msg in messages or []:
        if type(msg).__name__ != "AIMessage":
            continue
        per = _normalize_usage(msg)
        if not per:
            continue
        ai_message_count += 1
        for k in _USAGE_NUMERIC_KEYS:
            v = per.get(k)
            if isinstance(v, (int, float)):
                total[k] += int(v)
        # Top-level cache fields (newer LangChain) win over nested
        # ``input_token_details`` (older) to avoid double-counting when both
        # are present.
        has_top_cache = isinstance(per.get("cache_read_input_tokens"), (int, float)) or isinstance(
            per.get("cache_creation_input_tokens"), (int, float)
        )
        if not has_top_cache:
            details = per.get("input_token_details")
            if isinstance(details, dict):
                msg_cache_read = 0
                msg_cache_create = 0
                for alias in _INPUT_DETAILS_CACHE_READ_ALIASES:
                    v = details.get(alias)
                    if isinstance(v, (int, float)):
                        msg_cache_read += int(v)
                # When cache_control sets a ttl, langchain-anthropic writes
                # the count to the TTL-specific keys and zeroes ``cache_creation``.
                ttl_writes = int(details.get("ephemeral_5m_input_tokens", 0) or 0) + int(
                    details.get("ephemeral_1h_input_tokens", 0) or 0
                )
                msg_cache_create = ttl_writes or int(details.get("cache_creation", 0) or 0)
                total["cache_read_input_tokens"] += msg_cache_read
                total["cache_creation_input_tokens"] += msg_cache_create
                # langchain-anthropic 1.x inflates ``input_tokens`` to
                # fresh + read + create; subtract so it means fresh-only.
                this_input = per.get("input_tokens")
                if isinstance(this_input, (int, float)) and (msg_cache_read or msg_cache_create):
                    overcount = msg_cache_read + msg_cache_create
                    total["input_tokens"] -= min(overcount, int(this_input))
        # Carry through any other numeric extras (reasoning_tokens, …).
        for k, v in per.items():
            if k in _USAGE_NUMERIC_KEYS or k == "input_token_details":
                continue
            if isinstance(v, (int, float)):
                extras[k] = extras.get(k, 0) + int(v)
    out: dict[str, int] = {**total, **extras, "ai_message_count": ai_message_count}
    return out


def _tool_name(t: Any) -> str:
    name = getattr(t, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(t, dict):
        return str(t.get("name") or "")
    return ""


def _sorted_tools_for_cache(tools: Sequence) -> list:
    """Stable tool order so the Anthropic tool-block cache hash doesn't drift."""
    indexed = list(enumerate(tools))
    indexed.sort(key=lambda p: (_tool_name(p[1]), p[0]))
    return [t for _, t in indexed]


def _last_ai_message(messages: list) -> Any:
    """Return the final assistant message from an agent invoke result."""
    for msg in reversed(messages or []):
        # Prefer AIMessage by class name to avoid a hard import here.
        if type(msg).__name__ == "AIMessage":
            return msg
    return messages[-1] if messages else None


class _Engine:
    """Private engine shared by the provider-shape facades.

    Args:
        compresr_client: A live ``CompressionClient`` instance. The engine
            does not construct one itself — facades own client lifetime.
        llm: Provider-only (``"anthropic"``) or ``"provider:model"``
            selector (e.g. ``"anthropic:claude-haiku-4-5"``). Provider-only
            requires the call site to pass ``model=`` to ``run()`` —
            matching how the Anthropic/OpenAI SDKs work.
        llm_api_key: Optional provider API key. Forwarded to
            ``init_chat_model`` as ``api_key=`` when supplied.
        policy: Optional :class:`CompressionPolicy`. Defaults to the
            shared in-package default when omitted.
    """

    def __init__(
        self,
        *,
        compresr_client: Any,
        llm: str,
        llm_api_key: Optional[str] = None,
        policy: Optional[CompressionPolicy] = None,
        enable_prompt_cache: bool = True,
        prompt_cache_ttl: str = "5m",
        prompt_cache_min_messages: int = 2,
        openai_prompt_cache_key: Optional[str] = None,
        llm_http_client: Any = None,
        llm_http_async_client: Any = None,
    ) -> None:
        self._compresr_client = compresr_client
        self._provider, self._default_model_name = _parse_llm(llm)
        self._policy = policy or CompressionPolicy()
        self._llm_api_key = llm_api_key
        self._enable_prompt_cache = bool(enable_prompt_cache)
        self._prompt_cache_ttl = prompt_cache_ttl
        self._prompt_cache_min_messages = int(prompt_cache_min_messages)
        self._openai_prompt_cache_key = openai_prompt_cache_key
        # Custom httpx clients for the LLM provider transport (corporate proxy /
        # custom CA / verify settings). Baked into every chat-model build; a
        # per-call http_client passed to run() overrides these.
        self._llm_http_client = llm_http_client
        self._llm_http_async_client = llm_http_async_client
        # Chat-model cache keyed by effective model name. Construction is
        # deferred to the first ``run()`` so a provider-only client
        # (``llm="anthropic"``) doesn't try to resolve a model at init time.
        # Keyed by (model_name, sorted_kwargs_tuple) so distinct per-call knob
        # combinations get their own cached chat model.
        self._chat_models: dict[tuple[str, tuple], Any] = {}

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def default_model_name(self) -> Optional[str]:
        """Default model from the constructor; ``None`` if provider-only."""
        return self._default_model_name

    @property
    def model_name(self) -> Optional[str]:
        """Backwards-compatible alias for :attr:`default_model_name`.

        Deprecated — prefer :attr:`default_model_name`. Now ``Optional[str]``
        because the constructor accepts provider-only strings.
        """
        return self._default_model_name

    def _get_or_build_chat(self, model_name: str, extra_kwargs: Optional[dict] = None) -> Any:
        """Return a cached chat model for ``model_name`` + ``extra_kwargs``,
        building on miss.

        ``extra_kwargs`` are baked into the chat-model constructor so they
        survive ``create_agent``'s internal ``bind_tools`` call (which would
        otherwise strip a post-hoc ``chat.bind(...)``). Cache is keyed by
        ``(model_name, sorted_kwargs_tuple)`` so distinct knob combinations
        get their own cached model.
        """
        cache_key = (model_name, tuple(sorted((extra_kwargs or {}).items())))
        chat = self._chat_models.get(cache_key)
        if chat is not None:
            return chat
        chat = self._build_chat(model_name, **(extra_kwargs or {}))
        self._chat_models[cache_key] = chat
        return chat

    def _build_chat(self, model_name: str, **extra_kwargs: Any) -> Any:
        """Lift ``provider:model`` into a ``BaseChatModel`` with optional
        per-call knobs baked into the constructor."""
        kwargs: dict = dict(extra_kwargs)
        if self._llm_api_key is not None:
            kwargs.setdefault("api_key", self._llm_api_key)
        # Client-level custom httpx clients. setdefault so a per-call override
        # from run(http_client=...) wins. Only Anthropic and OpenAI chat models
        # accept these kwargs; google_genai uses a different transport.
        if self._provider in ("anthropic", "openai"):
            if self._llm_http_client is not None:
                kwargs.setdefault("http_client", self._llm_http_client)
            if self._llm_http_async_client is not None:
                kwargs.setdefault("http_async_client", self._llm_http_async_client)
        # OpenAI caching is server-side; we only pass ``prompt_cache_key``
        # and ``prompt_cache_retention`` via ``model_kwargs`` so they survive
        # ``bind_tools``.
        if self._provider == "openai" and self._enable_prompt_cache:
            openai_extra = self._openai_cache_model_kwargs()
            if openai_extra:
                mk = dict(kwargs.get("model_kwargs") or {})
                mk.update(openai_extra)
                kwargs["model_kwargs"] = mk
        # OpenAI Responses output_version is a constructor kwarg in newer
        # langchain-openai; older versions reject unknown kwargs. Try both.
        if self._provider == "openai":
            try:
                return init_chat_model(
                    f"{self._provider}:{model_name}",
                    output_version="responses/v1",
                    **kwargs,
                )
            except TypeError:  # pragma: no cover - older langchain-openai
                logger.debug("init_chat_model rejected output_version; falling back without it")
        if self._provider == "anthropic":
            return self._build_anthropic_chat(model_name, kwargs)
        return init_chat_model(f"{self._provider}:{model_name}", **kwargs)

    def _build_anthropic_chat(self, model_name: str, kwargs: dict) -> Any:
        """Build an Anthropic chat model, shimming ``http_client`` support on
        versions of langchain-anthropic that predate the native field.

        Native field present (PR merged / future release) → pass through. Absent
        (current PyPI) → strip the unknown kwargs so the constructor doesn't drop
        them silently, then swap the cached anthropic SDK client for one built
        with the caller's httpx client. The shim is a no-op once native support
        ships, so it can be removed then. See langchain issue #36056."""
        from langchain_anthropic import ChatAnthropic

        native = "http_client" in ChatAnthropic.model_fields
        sync_client = kwargs.get("http_client")
        async_client = kwargs.get("http_async_client")
        if not native:
            kwargs.pop("http_client", None)
            kwargs.pop("http_async_client", None)

        chat = init_chat_model(f"anthropic:{model_name}", **kwargs)

        if not native and (sync_client is not None or async_client is not None):
            self._inject_anthropic_http_client(chat, sync_client, async_client)
        return chat

    @staticmethod
    def _inject_anthropic_http_client(chat: Any, sync_client: Any, async_client: Any) -> None:
        """Replace ChatAnthropic's cached SDK clients with ones built around the
        caller's httpx client, reusing langchain's own ``_client_params`` so
        api_key / base_url / retries / headers stay correct. Writes through
        ``__dict__`` to seed the ``cached_property`` before first access."""
        import anthropic

        params = dict(chat._client_params)
        if sync_client is not None:
            chat.__dict__["_client"] = anthropic.Anthropic(**{**params, "http_client": sync_client})
        if async_client is not None:
            chat.__dict__["_async_client"] = anthropic.AsyncClient(
                **{**params, "http_client": async_client}
            )

    def _openai_cache_model_kwargs(self) -> dict:
        """Build the OpenAI-specific cache controls. ``prompt_cache_retention``
        is only set for ``ttl="1h"`` so older langchain-openai doesn't reject
        the unknown field on the default path."""
        out: dict = {}
        if self._openai_prompt_cache_key:
            out["prompt_cache_key"] = self._openai_prompt_cache_key
        if self._prompt_cache_ttl == "1h":
            out["prompt_cache_retention"] = "24h"
        return out

    def _build_middleware(self) -> list:
        """Build the middleware stack used for every run.

        The compression knobs come straight from
        :meth:`CompressionPolicy.tool_kwargs`.
        """
        mw: list = [
            _CompresrToolMiddleware(
                client=self._compresr_client,
                **self._policy.tool_kwargs(),
            )
        ]
        cache_mw = self._maybe_build_cache_middleware()
        if cache_mw is not None:
            # AFTER CompresrToolMiddleware so cache markers stamp the
            # post-compression content.
            mw.append(cache_mw)
        return mw

    def _maybe_build_cache_middleware(self) -> Optional[Any]:
        """Anthropic prompt-cache middleware when enabled, else None. Silently
        degrades on older ``langchain-anthropic`` without the middleware module."""
        if not self._enable_prompt_cache or self._provider != "anthropic":
            return None
        try:
            from langchain_anthropic.middleware import (  # type: ignore[import-not-found]
                AnthropicPromptCachingMiddleware,
            )
        except ImportError:
            logger.debug(
                "langchain_anthropic.middleware.AnthropicPromptCachingMiddleware "
                "unavailable; prompt caching disabled"
            )
            return None
        ttl = cast(Literal["5m", "1h"], self._prompt_cache_ttl)
        return AnthropicPromptCachingMiddleware(
            ttl=ttl,
            min_messages_to_cache=self._prompt_cache_min_messages,
            unsupported_model_behavior="ignore",
        )

    def _build_agent(
        self,
        *,
        chat: Any,
        tools: Sequence,
        system: Optional[str],
    ) -> Any:
        """Construct a LangChain agent with our middleware + tools.

        LangChain 1.0 accepts ``system_prompt`` as a ``create_agent`` kwarg.
        If a future/older version rejects it we fall back to prepending a
        system message at invoke time instead.
        """
        tools_list = _sorted_tools_for_cache(tools)
        middleware = self._build_middleware()
        try:
            return create_agent(
                model=chat,
                tools=tools_list,
                middleware=middleware,
                system_prompt=system,
            )
        except TypeError:  # pragma: no cover - defensive across LC versions
            logger.debug(
                "create_agent rejected system_prompt kwarg; falling back to message prepending"
            )
            return create_agent(
                model=chat,
                tools=tools_list,
                middleware=middleware,
            )

    def _collect_bind_kwargs(
        self,
        *,
        max_tokens: Optional[int],
        kw: dict,
    ) -> dict:
        """Peel off LLM-level kwargs and return the dict to feed ``chat.bind``.

        Mutates ``kw`` in place — removes any recognized LLM kwargs so they
        don't get forwarded as ``invoke`` kwargs. The explicit ``max_tokens``
        signature argument is included unconditionally when non-None so
        customers passing ``max_tokens=4096`` actually cap the model.

        Gemini's chat model expects ``max_output_tokens`` instead of
        ``max_tokens`` — we alias automatically when targeting
        ``google_genai`` and ``max_output_tokens`` wasn't supplied directly.
        """
        bind_kwargs: dict = {}
        for key in list(kw):
            if key in _LLM_BIND_KWARGS or key in _LLM_CONSTRUCTOR_KWARGS:
                bind_kwargs[key] = kw.pop(key)
        if max_tokens is not None:
            # Explicit signature param always reflects customer intent.
            bind_kwargs["max_tokens"] = max_tokens
        if self._provider == "google_genai":
            if "max_tokens" in bind_kwargs and "max_output_tokens" not in bind_kwargs:
                bind_kwargs["max_output_tokens"] = bind_kwargs.pop("max_tokens")
        return bind_kwargs

    def _resolve_model(self, model: Optional[str]) -> str:
        """Resolve the effective model name from call-site + default.

        Call-site ``model`` wins; otherwise we fall back to the constructor
        default. If neither is set we raise a clear migration error.
        """
        effective = model or self._default_model_name
        if not effective:
            raise CompresrError(
                "model is required — set it on the client "
                "(llm='anthropic:claude-...') or pass it to "
                ".create(model='...')."
            )
        return effective

    def _prepare_messages(
        self,
        *,
        agent: Any,
        messages: list,
        system: Optional[str],
    ) -> list:
        """Prepend a system message if ``create_agent`` couldn't take it.

        We can't easily introspect whether the agent already has a system
        prompt baked in, so we only prepend when the caller asked for one
        and the message list doesn't already start with a system entry.
        """
        if not system:
            return list(messages)
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return list(messages)
        # Heuristic: if the agent compiled successfully with `system_prompt`,
        # it owns the system message itself — don't double-up. We detect the
        # fallback path by checking the agent's invoke schema, but that's
        # fragile. Simpler: rely on the `system_prompt` try path above and
        # only reach here when the TypeError fallback ran, in which case
        # `system` will be threaded via state. Subclassing belongs in the
        # facades layer; for now just keep the original messages.
        return list(messages)

    def run(
        self,
        *,
        messages: list,
        tools: Sequence = (),
        system: Optional[str] = None,
        max_tokens: Optional[int] = 4096,
        model: Optional[str] = None,
        **kw: Any,
    ) -> NormalizedResult:
        """Synchronously invoke the agent and normalize the response.

        ``model`` overrides the constructor default. If neither is set,
        a :class:`CompresrError` is raised.

        Per-call LLM kwargs (``temperature``, ``top_p``, ``max_tokens``,
        ``stop_sequences``, etc.) are peeled off and forwarded to the
        underlying chat model via ``chat.bind(...)`` so they reach the
        provider without mutating the cached chat model.

        When ``tools`` is exactly one Compresr ``WebSearchTool``, the call
        is auto-routed through the deep-research loop — see
        :data:`_WEB_SEARCH_TOOL_NAMES` and :meth:`_run_via_research_loop`.
        """
        effective_model = self._resolve_model(model)
        bind_kwargs = self._collect_bind_kwargs(max_tokens=max_tokens, kw=kw)
        if _is_solo_web_search(tools):
            return self._run_via_research_loop(
                messages=messages,
                search_tool=tools[0],
                system=system,
                model=effective_model,
                bind_kwargs=bind_kwargs or None,
            )
        chat = self._get_or_build_chat(effective_model, extra_kwargs=bind_kwargs or None)
        agent = self._build_agent(chat=chat, tools=tools, system=system)
        prepared = self._prepare_messages(agent=agent, messages=messages, system=system)
        try:
            result = agent.invoke({"messages": prepared}, config=kw.pop("config", {}))
        except Exception as e:
            raise CompresrError(f"Agent execution failed: {e}") from e
        return self._normalize(result)

    async def arun(
        self,
        *,
        messages: list,
        tools: Sequence = (),
        system: Optional[str] = None,
        max_tokens: Optional[int] = 4096,
        model: Optional[str] = None,
        **kw: Any,
    ) -> NormalizedResult:
        """Async version of :meth:`run`.

        Solo-WebSearchTool calls are auto-routed through the research loop;
        because the loop itself is synchronous (LangChain ``bind_tools``
        + ``invoke``), the async path runs it via :func:`asyncio.to_thread`
        to avoid blocking the event loop.
        """
        effective_model = self._resolve_model(model)
        bind_kwargs = self._collect_bind_kwargs(max_tokens=max_tokens, kw=kw)
        if _is_solo_web_search(tools):
            import asyncio

            return await asyncio.to_thread(
                self._run_via_research_loop,
                messages=messages,
                search_tool=tools[0],
                system=system,
                model=effective_model,
                bind_kwargs=bind_kwargs or None,
            )
        chat = self._get_or_build_chat(effective_model, extra_kwargs=bind_kwargs or None)
        agent = self._build_agent(chat=chat, tools=tools, system=system)
        prepared = self._prepare_messages(agent=agent, messages=messages, system=system)
        try:
            result = await agent.ainvoke({"messages": prepared}, config=kw.pop("config", {}))
        except Exception as e:
            raise CompresrError(f"Agent execution failed: {e}") from e
        return self._normalize(result)

    def _run_via_research_loop(
        self,
        *,
        messages: Sequence,
        search_tool: Any,
        system: Optional[Any],
        model: str,
        bind_kwargs: Optional[dict],
    ) -> NormalizedResult:
        """Auto-route handler for solo-WebSearchTool calls.

        Builds a :class:`ResearchAgent`, runs its strict-output loop against
        the last user message, then feeds the resulting message chain through
        :meth:`_normalize` so each facade still gets its expected shape.

        ``system`` from the caller wins when supplied (string); otherwise
        the research-agent default prompt is used.
        """
        from compresr.agents.research.agent import ResearchAgent

        question = _extract_last_user_text(messages)
        system_prompt = system if isinstance(system, str) and system.strip() else None
        agent = ResearchAgent(
            engine=self,
            search_tool=search_tool,
            system_prompt=system_prompt,
        )
        state = agent._run_loop(
            question,
            model=model,
            extra_chat_kwargs=bind_kwargs,
        )
        return self._normalize({"messages": state.messages})

    def _normalize(self, agent_result: Any) -> NormalizedResult:
        """Reshape ``{"messages": [...]}`` into a :class:`NormalizedResult`.

        ``usage`` is summed across every AIMessage in the conversation — the
        single-last-turn read this used to do under-counted any agent loop
        with tool use. The full message chain is preserved on ``messages``
        for downstream consumers that need the trajectory.
        """
        messages = (
            (agent_result or {}).get("messages", []) if isinstance(agent_result, dict) else []
        )
        msg = _last_ai_message(messages)
        if msg is None:
            return NormalizedResult(text="", raw=None, messages=list(messages))
        return NormalizedResult(
            text=_content_text(msg),
            content_blocks=_normalize_content_blocks(msg),
            tool_uses=_normalize_tool_uses(msg),
            citations=_extract_citations(msg),
            stop_reason=_normalize_stop_reason(msg),
            usage=_aggregate_usage(messages),
            compresr_stats=CompresrStats(),
            raw=msg,
            messages=list(messages),
        )


__all__ = ["_Engine", "NormalizedResult", "Citation", "CompresrStats"]
