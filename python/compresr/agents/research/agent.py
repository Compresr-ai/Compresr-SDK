"""``ResearchAgent`` — strict-output ReAct loop with per-snippet compression.

Bypasses ``_Engine.run`` so we can control ``tool_choice`` per step (forces
the model to commit on the final step) and compress each tool result against
the live query before it enters the conversation. Cache control is stamped
manually for Anthropic since ``create_agent`` is not in the call path.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .parser import parse_research_output
from .prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT
from .types import Citation, ResearchResult, ResearchUsage, Step

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTEXT_TOKENS = 120_000
_DEFAULT_MAX_STEPS = 10
_DEFAULT_MIN_COMPRESS_TOKENS = 100


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class _LoopState:
    """Internal result of :meth:`ResearchAgent._run_loop`.

    Exposed to the engine's solo-WebSearchTool fast-path so it can feed the
    raw message chain to :meth:`_Engine._normalize` while the public
    :meth:`run` keeps producing a typed :class:`ResearchResult`.
    """

    messages: list
    ai_history: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    last_ai: Optional[Any] = None


class ResearchAgent:
    """Multi-step web-research agent with on-the-fly compression.

    Args:
        engine: A live :class:`compresr.agents.engine._Engine`.
        search_tool: LangChain ``BaseTool`` taking ``query: str``.
        max_steps: Hard cap on LLM turns. On the final turn ``tool_choice``
            is set to ``"none"`` so the model has to commit.
        system_prompt: Override the default research prompt.
        compress_snippets: Compress each tool result via
            ``client.compress(query=live_query)`` before appending.
        compression_model: Compresr model name (default ``"latte_v1"``).
        min_compress_tokens: Skip compression for tool results below this size.
        max_context_tokens: Drop the oldest tool-result pair when context
            exceeds this cap.
    """

    def __init__(
        self,
        *,
        engine: Any,
        search_tool: Any,
        max_steps: int = _DEFAULT_MAX_STEPS,
        system_prompt: Optional[str] = None,
        compress_snippets: bool = True,
        compression_model: str = "latte_v1",
        min_compress_tokens: int = _DEFAULT_MIN_COMPRESS_TOKENS,
        max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> None:
        self._engine = engine
        self._search_tool = search_tool
        self._max_steps = max(1, int(max_steps))
        self._system_prompt = system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT
        self._compress_snippets = bool(compress_snippets)
        self._compression_model = compression_model
        self._min_compress_tokens = int(min_compress_tokens)
        self._max_context_tokens = int(max_context_tokens)

    def run(self, question: str, *, model: Optional[str] = None) -> ResearchResult:
        state = self._run_loop(question, model=model)
        usage = self._aggregate_usage(
            state.ai_history,
            search_calls=sum(1 for s in state.trajectory if s.type == "search"),
        )
        final_text = self._stringify(state.last_ai.content) if state.last_ai is not None else ""
        parsed = parse_research_output(final_text)
        citations = self._collect_citations(parsed.citation_urls, state.trajectory)

        return ResearchResult(
            answer=parsed.answer,
            explanation=parsed.explanation,
            confidence=parsed.confidence,
            text=final_text,
            citations=citations,
            trajectory=state.trajectory,
            usage=usage,
            raw=state.last_ai,
        )

    def _run_loop(
        self,
        question: str,
        *,
        model: Optional[str] = None,
        extra_chat_kwargs: Optional[dict] = None,
    ) -> _LoopState:
        """Execute the strict-output ReAct loop and return raw state.

        Shared by :meth:`run` (which parses ``ResearchResult``) and the
        engine's solo-WebSearchTool fast-path (which feeds ``state.messages``
        into :meth:`_Engine._normalize`).

        ``extra_chat_kwargs`` lets the engine path bake per-call LLM knobs
        (``max_tokens``, ``temperature``…) into the cached chat model so
        they survive ``bind_tools``.
        """
        from langchain_core.messages import (  # type: ignore[import-not-found]
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        effective_model = self._engine._resolve_model(model)
        chat = self._engine._get_or_build_chat(effective_model, extra_kwargs=extra_chat_kwargs)

        messages: list = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=question),
        ]
        ai_history: list = []
        trajectory: list[Step] = []
        last_ai: Optional[Any] = None

        for step_idx in range(self._max_steps):
            self._maybe_truncate(messages)

            tool_choice = "none" if step_idx == self._max_steps - 1 else "auto"
            bound = chat.bind_tools([self._search_tool], tool_choice=tool_choice)
            invoke_messages = self._apply_cache_control(messages)

            t0 = time.time()
            try:
                response = bound.invoke(invoke_messages)
            except Exception as exc:  # noqa: BLE001
                trajectory.append(
                    Step(type="error", text=str(exc), latency_s=round(time.time() - t0, 3))
                )
                break
            step_latency = round(time.time() - t0, 3)

            ai_history.append(response)
            last_ai = response
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                trajectory.append(
                    Step(
                        type="answer",
                        text=self._stringify(response.content),
                        latency_s=step_latency,
                    )
                )
                break

            for tc in tool_calls:
                args = tc.get("args") if isinstance(tc, dict) else {}
                args = args if isinstance(args, dict) else {}
                query = args.get("query")
                trajectory.append(Step(type="search", query=query, latency_s=step_latency))

                raw_result = self._invoke_search(args)
                final_result = (
                    self._compress(raw_result, query or question)
                    if self._compress_snippets
                    else raw_result
                )
                trajectory.append(
                    Step(
                        type="tool_result",
                        chars=len(final_result),
                        text=final_result,
                    )
                )

                tool_call_id = tc.get("id") if isinstance(tc, dict) else None
                messages.append(
                    ToolMessage(
                        content=final_result,
                        tool_call_id=tool_call_id or "",
                        name=getattr(self._search_tool, "name", "search_web"),
                    )
                )

        return _LoopState(
            messages=messages,
            ai_history=ai_history,
            trajectory=trajectory,
            last_ai=last_ai,
        )

    def _apply_cache_control(self, messages: list) -> list:
        """Stamp Anthropic ephemeral cache_control on the last message.

        Mirrors ``AnthropicPromptCachingMiddleware`` for the bypass path —
        the research agent does not go through ``create_agent`` so the
        engine's middleware never runs. No-op for non-Anthropic providers.
        """
        provider = getattr(self._engine, "_provider", None)
        if provider != "anthropic":
            return messages
        if not getattr(self._engine, "_enable_prompt_cache", False):
            return messages
        min_msgs = getattr(self._engine, "_prompt_cache_min_messages", 2)
        if len(messages) < min_msgs:
            return messages

        ttl = getattr(self._engine, "_prompt_cache_ttl", "5m")
        marker = {"type": "ephemeral", "ttl": ttl}
        last = messages[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str):
            new_content: list = [{"type": "text", "text": content, "cache_control": marker}]
        elif isinstance(content, list) and content:
            new_content = list(content)
            tail = new_content[-1]
            if isinstance(tail, dict):
                tail = dict(tail)
                tail["cache_control"] = marker
                new_content[-1] = tail
            else:
                new_content.append({"type": "text", "text": str(tail), "cache_control": marker})
        else:
            return messages

        try:
            patched = last.model_copy(update={"content": new_content})
        except Exception:  # noqa: BLE001
            patched = type(last)(content=new_content)
        return [*messages[:-1], patched]

    def _invoke_search(self, args: dict) -> str:
        try:
            out = self._search_tool.invoke(args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("research: search tool raised %s; agent will continue", exc)
            return f"Search error: {exc}"
        return self._stringify(out)

    def _compress(self, text: str, query: str) -> str:
        if not text or _estimate_tokens(text) < self._min_compress_tokens:
            return text
        client = self._engine._compresr_client
        try:
            resp = client.compress(
                context=text,
                query=query,
                compression_model_name=self._compression_model,
            )
            data = getattr(resp, "data", None)
            compressed = getattr(data, "compressed_context", None) if data is not None else None
            return compressed if isinstance(compressed, str) and compressed else text
        except Exception as exc:  # noqa: BLE001
            logger.debug("research: compression failed (%s); using raw snippet", exc)
            return text

    def _maybe_truncate(self, messages: list) -> None:
        if not self._max_context_tokens:
            return
        total = sum(_estimate_tokens(self._stringify(getattr(m, "content", ""))) for m in messages)
        if total <= self._max_context_tokens:
            return
        guard = 0
        while total > self._max_context_tokens and len(messages) >= 4 and guard < 4:
            del messages[2:4]
            total = sum(
                _estimate_tokens(self._stringify(getattr(m, "content", ""))) for m in messages
            )
            guard += 1

    def _aggregate_usage(
        self,
        ai_messages: Sequence[Any],
        *,
        search_calls: int = 0,
    ) -> ResearchUsage:
        from compresr.agents.engine import _aggregate_usage

        agg = _aggregate_usage(list(ai_messages))
        return ResearchUsage(
            input_tokens=int(agg.get("input_tokens", 0) or 0),
            output_tokens=int(agg.get("output_tokens", 0) or 0),
            cache_read_tokens=int(agg.get("cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(agg.get("cache_creation_input_tokens", 0) or 0),
            calls=len(ai_messages),
            search_calls=int(search_calls or 0),
        )

    def _collect_citations(
        self, parsed_urls: Sequence[str], trajectory: Sequence[Step]
    ) -> list[Citation]:
        urls: list[str] = []
        for u in parsed_urls:
            if u not in urls:
                urls.append(u)
        for step in trajectory:
            if step.type != "tool_result" or not step.text:
                continue
            for url in re.findall(r"https?://[^\s,;<>\"')]+", step.text):
                if url not in urls:
                    urls.append(url)
        return [Citation(url=u) for u in urls]

    @staticmethod
    def _stringify(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content) if content is not None else ""
