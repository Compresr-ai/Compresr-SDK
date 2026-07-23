"""Drop-in compression node for a custom ``StateGraph``.

Reads a string field from state, compresses it, writes it back to the
same key. Query resolution follows the unified Compresr API:

    query: str | None                            # static query
    query_key: str | None                        # state key holding the query
    query_extractor: Callable[[State], str]      # custom extractor over state

Example::

    from langgraph.graph import StateGraph
    from compresr.integrations.langgraph import make_compresr_node

    graph.add_node(
        "compress",
        make_compresr_node(
            api_key=os.environ["COMPRESR_API_KEY"],
            context_key="retrieved_text",
            query_key="user_question",
        ),
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .._shared import (
    DEFAULT_MIN_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    ErrorPolicy,
    build_client,
    compress_safe,
    resolve_query,
)

logger = logging.getLogger(__name__)


def make_compresr_node(
    *,
    api_key: Optional[str] = None,
    context_key: str,
    compression_model: str = DEFAULT_MODEL,
    target_compression_ratio: float = DEFAULT_RATIO,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    coarse: Optional[bool] = None,
    query: Optional[str] = None,
    query_key: Optional[str] = None,
    query_extractor: Optional[Callable[[dict], Optional[str]]] = None,
    on_error: ErrorPolicy = DEFAULT_POLICY,
    base_url: Optional[str] = None,
    client: Any = None,
) -> Callable[[dict], dict]:
    """Build a state-graph node that compresses ``state[context_key]``."""
    compresr = client or build_client(
        api_key=api_key, base_url=base_url, caller="make_compresr_node"
    )

    def _resolve(state: dict) -> Optional[str]:
        args_dict = None
        if query_key is not None:
            v = state.get(query_key)
            if isinstance(v, str) and v.strip():
                args_dict = {query_key: v}
        return resolve_query(
            static=query,
            extractor=query_extractor,
            extractor_arg=state if query_extractor is not None else None,
            args=args_dict,
            args_key=query_key,
        )

    def node(state: dict) -> dict:
        ctx = state.get(context_key)
        if not isinstance(ctx, str) or not ctx.strip():
            return {}
        new = compress_safe(
            compresr,
            context=ctx,
            query=_resolve(state),
            compression_model_name=compression_model,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            min_tokens=min_tokens,
            on_error=on_error,
            context_label=f"node:{context_key}",
        )
        return {context_key: new} if new != ctx else {}

    return node
