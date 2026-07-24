"""``BaseNodePostprocessor`` that compresses retrieved node content.

The query is supplied by the query engine via ``QueryBundle.query_str``;
``query=...`` overrides it (rarely needed).

Example::

    from llama_index.core import VectorStoreIndex
    from compresr.integrations.llamaindex import CompresrNodePostprocessor

    pp = CompresrNodePostprocessor(api_key=os.environ["COMPRESR_API_KEY"])
    query_engine = index.as_query_engine(node_postprocessors=[pp])
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from .._shared import (
    BATCH_LIMIT,
    DEFAULT_MIN_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_POLICY,
    DEFAULT_RATIO,
    ErrorPolicy,
    build_client,
    estimate_tokens,
)

logger = logging.getLogger(__name__)

try:
    from llama_index.core.bridge.pydantic import Field  # type: ignore[import-not-found]
    from llama_index.core.postprocessor.types import (  # type: ignore[import-not-found]
        BaseNodePostprocessor,
    )
    from llama_index.core.schema import NodeWithScore, QueryBundle  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CompresrNodePostprocessor requires llama-index-core. "
        "Install with: pip install compresr[llamaindex]"
    ) from exc


class CompresrNodePostprocessor(BaseNodePostprocessor):
    """Compress retrieved nodes with query-aware compression.

    Eligible nodes (content >= ``min_tokens``) are compressed in a single
    batch call per ``BATCH_LIMIT`` slice. Compressed text replaces
    ``node.text`` on a *copy* — the original index is untouched.
    """

    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    compression_model: str = Field(default=DEFAULT_MODEL)
    target_compression_ratio: float = Field(default=DEFAULT_RATIO)
    target_token: Optional[int] = Field(
        default=None,
        description=(
            "Absolute output token budget per node. When set, overrides "
            "target_compression_ratio with `ratio = max(estimate_tokens(ctx) / "
            "target_token, 1.0)`. Approximate (chars/4 estimator)."
        ),
    )
    min_tokens: int = Field(default=DEFAULT_MIN_TOKENS)
    coarse: Optional[bool] = Field(default=None)
    on_error: ErrorPolicy = Field(default=DEFAULT_POLICY)
    query: Optional[str] = Field(
        default=None,
        description="Override the query passed by the query engine.",
    )
    client: Any = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def class_name(cls) -> str:
        return "CompresrNodePostprocessor"

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        self.client = build_client(
            api_key=self.api_key,
            base_url=self.base_url,
            caller="CompresrNodePostprocessor",
        )
        return self.client

    def _resolve_query(self, query_bundle: Optional["QueryBundle"]) -> Optional[str]:
        if isinstance(self.query, str) and self.query.strip():
            return self.query
        if query_bundle is not None:
            q = getattr(query_bundle, "query_str", None)
            if isinstance(q, str) and q.strip():
                return q
        return None

    def _partition(self, nodes: List["NodeWithScore"]) -> tuple[list[int], list[str]]:
        indices: list[int] = []
        contents: list[str] = []
        for i, n in enumerate(nodes):
            text = n.node.get_content()
            if isinstance(text, str) and estimate_tokens(text) >= self.min_tokens:
                indices.append(i)
                contents.append(text)
        return indices, contents

    def _apply_results(
        self,
        nodes: List["NodeWithScore"],
        indices: list[int],
        results: list,
        slice_start: int,
        chunk_len: int,
    ) -> None:
        for offset, item in enumerate(results[:chunk_len]):
            global_idx = indices[slice_start + offset]
            new = getattr(item, "compressed_context", None)
            if isinstance(new, str) and new:
                _set_node_text(nodes[global_idx], new)

    def _effective_ratio(self, chunk: List[str]) -> float:
        """Per-chunk compression ratio.

        Backend accepts only ``target_compression_ratio`` on the wire. When
        ``target_token`` is set, translate it client-side using the average
        per-context token estimate for the chunk."""
        if not self.target_token or self.target_token <= 0:
            return self.target_compression_ratio
        avg = max(1.0, sum(estimate_tokens(c) for c in chunk) / max(1, len(chunk)))
        return max(avg / self.target_token, 1.0)

    def _postprocess_nodes(
        self,
        nodes: List["NodeWithScore"],
        query_bundle: Optional["QueryBundle"] = None,
    ) -> List["NodeWithScore"]:
        if not nodes:
            return nodes

        indices, contents = self._partition(nodes)
        if not contents:
            return nodes

        query = self._resolve_query(query_bundle)
        if not query:
            logger.warning(
                "CompresrNodePostprocessor: no query resolved; passthrough. "
                "Provide via QueryBundle.query_str or `query=` on the postprocessor."
            )
            return nodes

        client = self._get_client()
        out = [_clone_node_with_score(n) for n in nodes]

        for start in range(0, len(contents), BATCH_LIMIT):
            chunk = contents[start : start + BATCH_LIMIT]
            ratio = self._effective_ratio(chunk)
            try:
                resp = client.compress_batch(
                    contexts=chunk,
                    queries=query,
                    compression_model_name=self.compression_model,
                    target_compression_ratio=ratio,
                    coarse=self.coarse,
                )
            except Exception as exc:  # noqa: BLE001
                if self.on_error == "raise":
                    raise
                logger.warning(
                    "compresr: postprocessor batch failed (%s); passthrough %d nodes.",
                    exc,
                    len(chunk),
                )
                continue

            results = (resp.data.results if resp.data else None) or []
            self._apply_results(out, indices, results, start, len(chunk))

        return out


def _clone_node_with_score(nws: "NodeWithScore") -> "NodeWithScore":
    if hasattr(nws.node, "model_copy"):
        new_node = nws.node.model_copy()
    elif hasattr(nws.node, "copy"):
        new_node = nws.node.copy()
    else:
        logger.warning(
            "CompresrNodePostprocessor: node type %s lacks model_copy/copy; "
            "downstream mutations will affect the original.",
            type(nws.node).__name__,
        )
        new_node = nws.node
    return NodeWithScore(node=new_node, score=nws.score)


def _set_node_text(nws: "NodeWithScore", new_text: str) -> None:
    node = nws.node
    if hasattr(node, "set_content"):
        try:
            node.set_content(new_text)
            return
        except Exception:
            pass
    if hasattr(node, "text"):
        try:
            node.text = new_text
            return
        except Exception:
            pass
    md = getattr(node, "metadata", None)
    if isinstance(md, dict):
        md["compresr_compressed"] = new_text
    logger.warning(
        "CompresrNodePostprocessor: node type %s has no writable text field; "
        "compressed text stored in metadata[compresr_compressed] but the "
        "synthesizer will see the original text via get_content().",
        type(node).__name__,
    )
