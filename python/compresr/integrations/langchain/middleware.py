"""Agent middleware for LangChain 1.0+ ``create_agent``.

- :class:`CompresrToolMiddleware` — compresses each tool output as it
  returns from the tool node, before it enters agent state.
- :class:`CompresrSummarizationMiddleware` — when state grows past a token
  threshold, compresses the older block into a single summary message and
  replaces it in state (keeping the recent tail untouched). Mirrors
  LangChain's ``SummarizationMiddleware`` but uses Compresr instead of an
  LLM call. Designed to preserve upstream KV cache across turns: the
  summary is generated once at trigger time and persists in state,
  so prompt prefixes stay stable.

Both accept the unified Compresr query API:

    query: str | None                       # static query
    query_extractor: Callable               # custom extractor
    query_arg: str | None                   # name of the tool arg

Both fail open by default (``on_error="passthrough"``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

from .._shared import (
    DEFAULT_MIN_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    CompressionPolicy,
    ErrorPolicy,
    ToolOutputCompressor,
    acompress_safe,
    compress_safe,
    estimate_tokens,
    resolve_query,
)

logger = logging.getLogger(__name__)

try:
    from langchain.agents.middleware import AgentMiddleware  # type: ignore[import-not-found]
    from langchain_core.messages import (  # type: ignore[import-not-found]
        BaseMessage,
        HumanMessage,
        RemoveMessage,
        ToolMessage,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CompresrToolMiddleware requires langchain>=1.0. "
        "Install with: pip install compresr[langchain]"
    ) from exc

try:  # langgraph re-exports REMOVE_ALL_MESSAGES sentinel
    from langgraph.graph.message import REMOVE_ALL_MESSAGES  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    REMOVE_ALL_MESSAGES = "__remove_all__"  # type: ignore[assignment]


ToolQueryExtractor = Callable[[dict, list], Optional[str]]
SummaryQueryExtractor = Callable[[list], Optional[str]]


def _build_tool_message(content: str, original: "ToolMessage") -> "ToolMessage":
    return ToolMessage(
        content=content,
        tool_call_id=original.tool_call_id,
        name=original.name,
    )


class _CompresrMiddlewareBase(AgentMiddleware):
    def __init__(
        self,
        *,
        api_key: Optional[str],
        compression_model: str,
        target_compression_ratio: float,
        coarse: Optional[bool],
        on_error: ErrorPolicy,
        base_url: Optional[str],
        client: Any,
    ) -> None:
        super().__init__()
        from .._shared import build_client

        self._client = client or build_client(
            api_key=api_key, base_url=base_url, caller="Compresr middleware"
        )
        self._ratio = target_compression_ratio
        self._model = compression_model
        self._coarse = coarse
        self._on_error = on_error


class CompresrToolMiddleware(_CompresrMiddlewareBase):
    """Compress every eligible tool output before it enters agent state.

    Example::

        from langchain.agents import create_agent
        from compresr.integrations.langchain import CompresrToolMiddleware

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[CompresrToolMiddleware(
                api_key=os.environ["COMPRESR_API_KEY"],
                allow_tools={"search"},
                query_arg="query",
            )],
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        target_compression_ratio: float = DEFAULT_RATIO,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        compression_model: str = DEFAULT_MODEL,
        coarse: Optional[bool] = None,
        allow_tools: Optional[Iterable[str]] = None,
        ignore_tools: Optional[Iterable[str]] = None,
        query: Optional[str] = None,
        query_extractor: Optional[ToolQueryExtractor] = None,
        query_arg: Optional[str] = None,
        on_error: ErrorPolicy = DEFAULT_POLICY,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            compression_model=compression_model,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            on_error=on_error,
            base_url=base_url,
            client=client,
        )
        policy = CompressionPolicy(
            target_compression_ratio=target_compression_ratio,
            compression_model_name=compression_model,
            coarse=coarse,
            min_tokens=min_tokens,
            on_error=on_error,
            allow_tools=allow_tools,
            ignore_tools=ignore_tools,
        )
        self._compressor = ToolOutputCompressor(client=self._client, policy=policy)
        self._static_query = query
        self._query_extractor = query_extractor
        self._query_arg = query_arg

    def _resolve_query(self, tool_call: dict, messages: list) -> Optional[str]:
        return resolve_query(
            static=self._static_query,
            extractor=self._query_extractor,
            extractor_arg=(tool_call, messages) if self._query_extractor else None,
            args=(tool_call or {}).get("args"),
            args_key=self._query_arg,
            messages=messages,
            tool_call_id=(tool_call or {}).get("id"),
        )

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        result = handler(request)
        tool_call = getattr(request, "tool_call", None) or {}
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        messages = list(getattr(request, "messages", None) or [])
        tool_name = (tool_call or {}).get("name") or getattr(result, "name", None) or ""
        new = self._compressor.process(
            tool_name=tool_name,
            output=result.content,
            query=self._resolve_query(tool_call, messages),
        )
        return _build_tool_message(new, result) if new != result.content else result

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        result = await handler(request)
        tool_call = getattr(request, "tool_call", None) or {}
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        messages = list(getattr(request, "messages", None) or [])
        tool_name = (tool_call or {}).get("name") or getattr(result, "name", None) or ""
        new = await self._compressor.aprocess(
            tool_name=tool_name,
            output=result.content,
            query=self._resolve_query(tool_call, messages),
        )
        return _build_tool_message(new, result) if new != result.content else result


SUMMARY_PREFIX = "[Earlier conversation summary]\n\n"


def _msg_to_text(m: "BaseMessage") -> str:
    """Render a message as `role: content` for joining before compression."""
    role = type(m).__name__.replace("Message", "").lower() or "msg"
    if hasattr(m, "name") and m.name:
        role = f"{role}:{m.name}"
    content = m.content if isinstance(m.content, str) else str(m.content)
    return f"{role}: {content}"


class CompresrSummarizationMiddleware(_CompresrMiddlewareBase):
    """Compress old conversation history into one summary when state grows
    past a token threshold. Keeps the recent ``messages_to_keep`` messages
    untouched so the upstream KV cache stays warm.

    Mirrors LangChain's ``SummarizationMiddleware`` shape but uses Compresr
    for the summary instead of an LLM call — same goal (bounded prompt),
    different mechanism (token-level compression).

    Example::

        from langchain.agents import create_agent
        from compresr.integrations.langchain import CompresrSummarizationMiddleware

        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[CompresrSummarizationMiddleware(
                api_key=os.environ["COMPRESR_API_KEY"],
                max_tokens_before_summary=4_000,
                messages_to_keep=20,
            )],
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        max_tokens_before_summary: int = 4_000,
        messages_to_keep: int = 20,
        trigger: Optional[int] = None,
        keep: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        target_compression_ratio: float = DEFAULT_RATIO,
        compression_model: str = DEFAULT_MODEL,
        coarse: Optional[bool] = None,
        query: Optional[str] = None,
        query_extractor: Optional[SummaryQueryExtractor] = None,
        on_error: ErrorPolicy = DEFAULT_POLICY,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            compression_model=compression_model,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            on_error=on_error,
            base_url=base_url,
            client=client,
        )
        self._max_tokens = max(1, trigger if trigger is not None else max_tokens_before_summary)
        self._keep = max(1, keep if keep is not None else messages_to_keep)
        self._token_counter = token_counter or estimate_tokens
        self._static_query = query
        self._query_extractor = query_extractor

    def _resolve_query(self, messages: list) -> Optional[str]:
        return resolve_query(
            static=self._static_query,
            extractor=self._query_extractor,
            extractor_arg=messages if self._query_extractor else None,
            messages=messages,
        )

    def _is_already_summarized(self, messages: list) -> bool:
        if not messages:
            return False
        first = messages[0]
        return (
            isinstance(first, HumanMessage)
            and isinstance(first.content, str)
            and first.content.startswith(SUMMARY_PREFIX)
        )

    def before_model(self, state, runtime):  # type: ignore[no-untyped-def]
        messages = list(state.get("messages", []))
        if len(messages) <= self._keep:
            return None

        total = sum(self._token_counter(m.content) for m in messages if isinstance(m.content, str))
        if total < self._max_tokens:
            return None

        to_summarize = messages[: -self._keep]
        recent = messages[-self._keep :]
        if not to_summarize:
            return None

        # If the first message is already a Compresr summary, fold any new
        # old messages into a fresh summary alongside it — never produce
        # nested summary-of-summary.
        if self._is_already_summarized(to_summarize):
            preserved_summary = to_summarize[0].content[len(SUMMARY_PREFIX) :]
            extra_old = to_summarize[1:]
            joined = preserved_summary + "\n\n" + "\n".join(_msg_to_text(m) for m in extra_old)
        else:
            joined = "\n".join(_msg_to_text(m) for m in to_summarize)

        compressed = compress_safe(
            self._client,
            context=joined,
            query=self._resolve_query(messages),
            compression_model_name=self._model,
            target_compression_ratio=self._ratio,
            coarse=self._coarse,
            min_tokens=1,
            on_error=self._on_error,
            context_label="summary",
        )
        if compressed == joined:
            return None  # passthrough — backend returned the same content

        summary = HumanMessage(content=f"{SUMMARY_PREFIX}{compressed}")
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                summary,
                *recent,
            ]
        }


def _rebuild_with_content(msg: "BaseMessage", new_content: str) -> "BaseMessage":
    """Recreate a message of the same class with new string content."""
    kwargs: dict = {"content": new_content}
    for attr in ("tool_call_id", "name", "tool_calls", "id"):
        v = getattr(msg, attr, None)
        if v is not None:
            kwargs[attr] = v
    try:
        return msg.__class__(**kwargs)
    except Exception:  # noqa: BLE001 — defensive: fall back to original
        return msg


class CompresrPromptMiddleware(_CompresrMiddlewareBase):
    """Cap the outbound prompt at a token budget. Walks ``request.messages``
    largest-first and compresses each long body with Compresr until the
    total fits under ``max_tokens``. Mutates only the request handed to the
    model — agent state is unchanged.

    Intended as a last-mile safety net (e.g., to keep prompts inside a
    context window) rather than a primary compression strategy. Combine
    with :class:`CompresrSummarizationMiddleware` for steady-state savings.

    Example::

        from langchain.agents import create_agent
        from compresr.integrations.langchain import CompresrPromptMiddleware

        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[CompresrPromptMiddleware(
                api_key=os.environ["COMPRESR_API_KEY"],
                max_tokens=8_000,
            )],
        )
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        api_key: Optional[str] = None,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        target_compression_ratio: float = DEFAULT_RATIO,
        compression_model: str = DEFAULT_MODEL,
        coarse: Optional[bool] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        query: Optional[str] = None,
        query_extractor: Optional[SummaryQueryExtractor] = None,
        on_error: ErrorPolicy = DEFAULT_POLICY,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            compression_model=compression_model,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            on_error=on_error,
            base_url=base_url,
            client=client,
        )
        self._max_tokens = max(1, max_tokens)
        self._min_tokens = max(1, min_tokens)
        self._token_counter = token_counter or estimate_tokens
        self._static_query = query
        self._query_extractor = query_extractor

    def _resolve_query(self, messages: list) -> Optional[str]:
        return resolve_query(
            static=self._static_query,
            extractor=self._query_extractor,
            extractor_arg=messages if self._query_extractor else None,
            messages=messages,
        )

    def _total_tokens(self, messages: list) -> int:
        return sum(self._token_counter(m.content) for m in messages if isinstance(m.content, str))

    def _plan(self, messages: list) -> tuple[list[tuple[int, "BaseMessage"]], Optional[str]]:
        """Return (candidates sorted largest-first, resolved_query). Empty
        candidates list means no shrink work needed."""
        if self._total_tokens(messages) <= self._max_tokens:
            return [], None
        candidates = [
            (i, m)
            for i, m in enumerate(messages)
            if isinstance(m.content, str) and self._token_counter(m.content) >= self._min_tokens
        ]
        candidates.sort(key=lambda im: self._token_counter(im[1].content), reverse=True)
        return candidates, self._resolve_query(messages)

    def _target_ratio(self, msg: "BaseMessage", new: list) -> float:
        content: str = msg.content  # type: ignore[assignment]  # guarded by _plan
        current = self._token_counter(content)
        overshoot = self._total_tokens(new) - self._max_tokens
        target = max(self._min_tokens, current - max(1, overshoot))
        return max(current / max(1, target), 1.0)

    def _shrink(self, messages: list) -> list:
        candidates, query = self._plan(messages)
        if not candidates:
            return messages
        new = list(messages)
        for idx, msg in candidates:
            if self._total_tokens(new) <= self._max_tokens:
                break
            content: str = msg.content  # type: ignore[assignment]  # guarded by _plan
            compressed = compress_safe(
                self._client,
                context=content,
                query=query,
                compression_model_name=self._model,
                target_compression_ratio=self._target_ratio(msg, new),
                coarse=self._coarse,
                min_tokens=self._min_tokens,
                on_error=self._on_error,
                context_label="prompt_budget",
            )
            if compressed != content:
                new[idx] = _rebuild_with_content(msg, compressed)
        return new

    async def _ashrink(self, messages: list) -> list:
        candidates, query = self._plan(messages)
        if not candidates:
            return messages
        new = list(messages)
        for idx, msg in candidates:
            if self._total_tokens(new) <= self._max_tokens:
                break
            content: str = msg.content  # type: ignore[assignment]  # guarded by _plan
            compressed = await acompress_safe(
                self._client,
                context=content,
                query=query,
                compression_model_name=self._model,
                target_compression_ratio=self._target_ratio(msg, new),
                coarse=self._coarse,
                min_tokens=self._min_tokens,
                on_error=self._on_error,
                context_label="prompt_budget",
            )
            if compressed != content:
                new[idx] = _rebuild_with_content(msg, compressed)
        return new

    def wrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        try:
            original = list(request.messages)
            shrunk = self._shrink(original)
            if _list_diff(shrunk, request.messages):
                request = _rebuild_request(request, shrunk)
        except Exception:  # noqa: BLE001 — fail open: send the original prompt
            logger.exception("CompresrPromptMiddleware shrink failed; sending original prompt")
        return handler(request)

    async def awrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        try:
            original = list(request.messages)
            shrunk = await self._ashrink(original)
            if _list_diff(shrunk, request.messages):
                request = _rebuild_request(request, shrunk)
        except Exception:  # noqa: BLE001
            logger.exception("CompresrPromptMiddleware shrink failed; sending original prompt")
        return await handler(request)


def _list_diff(new: list, old: list) -> bool:
    """Element-identity comparison — only True when ``_shrink`` actually
    rebuilt at least one message. A fresh ``list(...)`` of unchanged
    ``BaseMessage`` instances doesn't count as a diff."""
    if len(new) != len(old):
        return True
    for a, b in zip(new, old):
        if a is not b:
            return True
    return False


def _rebuild_request(request: Any, messages: list) -> Any:
    """Return a new request object with the messages list swapped in.

    Avoids mutating the caller-owned ``request`` so the middleware stays
    side-effect free. Falls back to mutation only if neither the Pydantic
    ``model_copy`` path nor the ``__dict__`` copy path is available.
    """
    model_copy = getattr(request, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update={"messages": messages})
        except Exception:  # noqa: BLE001 — fall through to dict path
            pass
    cls = type(request)
    try:
        clone = cls.__new__(cls)  # type: ignore[call-overload]
        clone.__dict__.update(request.__dict__)
        clone.messages = messages
        return clone
    except Exception:  # noqa: BLE001 — last-resort: mutate in place.
        request.messages = messages
        return request
