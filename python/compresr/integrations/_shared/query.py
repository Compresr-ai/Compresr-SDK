"""Extract a query string for ``latte_v1`` query-aware compression.

Three resolution layers, in priority order:

1. **Static** — user supplies a fixed ``query="..."`` (same for every call).
2. **Custom extractor** — user supplies a ``Callable`` that returns the query
   from some context (tool args, state dict, message list, ...).
3. **Smart default** — pick from common arg keys (``query``, ``question``,
   ...), then last human/user message, then a fallback string.

Pure Python, no I/O, no framework imports. Every integration calls
``resolve_query()`` so the user-facing API is identical across LangChain,
LangGraph, and LlamaIndex.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

DEFAULT_FALLBACK = "summarize"

COMMON_QUERY_KEYS: tuple[str, ...] = (
    "query",
    "question",
    "search_query",
    "q",
    "prompt",
    "input",
    "text",
)


def _is_ai_message(msg: Any) -> bool:
    name = type(msg).__name__
    if name in {"AIMessage", "AIMessageChunk"}:
        return True
    role = getattr(msg, "role", None) or _dict_get(msg, "role")
    return role in {"assistant", "ai"}


def _is_human_message(msg: Any) -> bool:
    name = type(msg).__name__
    if name in {"HumanMessage", "HumanMessageChunk"}:
        return True
    role = getattr(msg, "role", None) or _dict_get(msg, "role")
    return role in {"user", "human"}


def _dict_get(msg: Any, key: str) -> Any:
    if isinstance(msg, dict):
        return msg.get(key)
    return None


def _get_content(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg.get("content")
    return getattr(msg, "content", None)


def _get_tool_calls(msg: Any) -> Sequence[dict]:
    calls = getattr(msg, "tool_calls", None)
    if calls is None and isinstance(msg, dict):
        calls = msg.get("tool_calls")
    return calls or []


def extract_query_from_messages(
    messages: Sequence[Any],
    current_tool_call_id: Optional[str] = None,
    *,
    fallback: str = DEFAULT_FALLBACK,
) -> str:
    """Pick the best query from a message history.

    Priority:
        1. The args of the AI tool-call matching ``current_tool_call_id``.
        2. The most recent human/user message content.
        3. ``fallback``.
    """
    if current_tool_call_id is not None:
        for prev in reversed(messages):
            if not _is_ai_message(prev):
                continue
            for call in _get_tool_calls(prev):
                if not isinstance(call, dict):
                    continue
                if call.get("id") == current_tool_call_id:
                    q = extract_query_from_args(call.get("args") or {})
                    if q:
                        return q

    for prev in reversed(messages):
        if not _is_human_message(prev):
            continue
        content = _get_content(prev)
        if isinstance(content, str) and content.strip():
            return content

    return fallback


def extract_query_from_args(
    args: dict,
    *,
    preferred_key: Optional[str] = None,
    candidates: Iterable[str] = COMMON_QUERY_KEYS,
) -> Optional[str]:
    """Pick a likely-query value from tool call ``args``.

    1. If ``preferred_key`` is given and present, use it.
    2. Try ``candidates`` (default: COMMON_QUERY_KEYS) in order.
    3. Return ``None`` if nothing matches — the caller decides the fallback.
    """
    if not isinstance(args, dict) or not args:
        return None

    if preferred_key is not None:
        v = args.get(preferred_key)
        if isinstance(v, str) and v.strip():
            return v
        return None  # user named a key; don't silently fall back

    for key in candidates:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v

    return None


_MISSING = object()


def resolve_query(
    *,
    static: Optional[str] = None,
    extractor: Optional[Callable[..., Optional[str]]] = None,
    extractor_arg: Any = _MISSING,
    args: Optional[dict] = None,
    args_key: Optional[str] = None,
    messages: Optional[Sequence[Any]] = None,
    tool_call_id: Optional[str] = None,
    fallback: str = DEFAULT_FALLBACK,
) -> str:
    """Resolve a query string with consistent priority across integrations.

    Priority (first non-empty wins):
        1. ``static`` — fixed query supplied by the user.
        2. ``extractor(extractor_arg)`` — user callable. If ``extractor_arg``
           is omitted (sentinel), the extractor is called with no args.
        3. ``args[args_key]`` — explicit arg name on a tool call.
        4. ``args`` smart-picked via ``COMMON_QUERY_KEYS``.
        5. Most recent human/user message in ``messages``.
        6. ``fallback`` (default ``"summarize"``).
    """
    if isinstance(static, str) and static.strip():
        return static

    if extractor is not None:
        try:
            value = extractor() if extractor_arg is _MISSING else extractor(extractor_arg)
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value

    if args is not None:
        v = extract_query_from_args(args, preferred_key=args_key)
        if v:
            return v

    if messages:
        return extract_query_from_messages(messages, tool_call_id, fallback=fallback)

    return fallback


__all__ = [
    "COMMON_QUERY_KEYS",
    "DEFAULT_FALLBACK",
    "extract_query_from_args",
    "extract_query_from_messages",
    "resolve_query",
]
