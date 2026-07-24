"""``compresr_handoff_tool`` — supervisor → subagent handoff tool with
compression baked in.

Supervisor agents in LangGraph 1.x emit a ``Command(goto=..., update=...)``
to hand control to another agent. The handoff payload typically carries:

- a ``task_description`` (free-form instructions, can be long),
- a ``context`` snippet (relevant excerpts to give the subagent a head
  start, can be very long for RAG handoffs).

This tool compresses both fields before forwarding them, so the subagent's
prompt starts smaller without changing the supervisor's reasoning.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from .._shared import (
    DEFAULT_MIN_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    ErrorPolicy,
    build_client,
    compress_safe,
)

try:
    from langchain_core.messages import ToolMessage  # type: ignore[import-not-found]
    from langchain_core.tools import (  # type: ignore[import-not-found]
        InjectedToolCallId,
        tool,
    )
    from langgraph.types import Command  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "compresr_handoff_tool requires langgraph + langchain-core. "
        "Install with: pip install compresr[langgraph]"
    ) from exc


def compresr_handoff_tool(
    agent_name: str,
    *,
    description: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Any = None,
    compression_model: str = DEFAULT_MODEL,
    target_compression_ratio: float = DEFAULT_RATIO,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    coarse: Optional[bool] = None,
    on_error: ErrorPolicy = DEFAULT_POLICY,
    base_url: Optional[str] = None,
) -> Any:
    """Build a LangChain handoff tool that compresses the payload before
    routing to ``agent_name``.

    Returns a ``BaseTool`` you can register with the supervisor agent.

    Example::

        from langchain.agents import create_agent
        from compresr.integrations.langgraph import compresr_handoff_tool

        researcher = create_agent(...)
        writer = create_agent(...)
        supervisor = create_agent(
            model=model,
            tools=[
                compresr_handoff_tool("researcher", api_key=KEY),
                compresr_handoff_tool("writer", api_key=KEY),
            ],
        )
    """
    compresr = client or build_client(
        api_key=api_key, base_url=base_url, caller="compresr_handoff_tool"
    )
    tool_name = f"transfer_to_{agent_name}"
    tool_description = description or (
        f"Hand off the task to the `{agent_name}` agent. "
        "Provide a clear task_description and any context the agent needs."
    )

    def _maybe_compress(text: str, *, query: str, label: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return compress_safe(
            compresr,
            context=text,
            query=query,
            compression_model_name=compression_model,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            min_tokens=min_tokens,
            on_error=on_error,
            context_label=label,
        )

    @tool(tool_name, description=tool_description)
    def handoff(
        task_description: str,
        context: str = "",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        # latte_v1 requires a query. The task description is the natural
        # one for the context excerpt (what the subagent is being asked to
        # do with it); the task description itself uses the agent name as
        # a fallback hint when shrinking itself.
        compressed_task = _maybe_compress(
            task_description, query=f"task for {agent_name}", label="handoff:task"
        )
        compressed_ctx = (
            _maybe_compress(
                context,
                query=task_description or f"context for {agent_name}",
                label="handoff:context",
            )
            if context
            else ""
        )

        ack = ToolMessage(
            content=f"Successfully transferred to {agent_name}.",
            tool_call_id=tool_call_id,
            name=tool_name,
        )
        return Command(
            goto=agent_name,
            update={
                "messages": [ack],
                "task_description": compressed_task,
                "context": compressed_ctx,
            },
            graph=Command.PARENT,
        )

    return handoff
