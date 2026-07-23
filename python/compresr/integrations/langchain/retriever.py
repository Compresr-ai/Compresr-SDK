"""``BaseDocumentCompressor`` for ``ContextualCompressionRetriever``.

Drop-in replacement for ``LLMChainExtractor`` — one Compresr batch call
replaces N LLM extraction calls.

Example::

    from langchain.retrievers import ContextualCompressionRetriever
    from compresr.integrations.langchain import CompresrExtractor

    compressor = CompresrExtractor(api_key=os.environ["COMPRESR_API_KEY"])
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    )
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

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
    from langchain_core.callbacks.manager import Callbacks  # type: ignore[import-not-found]
    from langchain_core.documents import Document  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CompresrExtractor requires langchain-core. "
        "Install with: pip install compresr[langchain]"
    ) from exc

# `BaseDocumentCompressor` lives in `langchain_core.documents` through
# LangChain 1.x; some 1.x distributions also re-export it from
# `langchain_classic`. Prefer `langchain_core` to avoid pulling the heavy
# `langchain_classic` import chain (transformers/keras) when not needed.
try:
    from langchain_core.documents import (  # type: ignore[import-not-found]
        BaseDocumentCompressor,
    )
except ImportError:  # pragma: no cover
    try:
        from langchain_classic.retrievers.document_compressors.base import (  # type: ignore[import-not-found]
            BaseDocumentCompressor,
        )
    except ImportError as exc:
        raise ImportError(
            "CompresrExtractor requires `BaseDocumentCompressor`. "
            "Install `langchain-core>=0.3` or `langchain-classic`."
        ) from exc


class CompresrExtractor(BaseDocumentCompressor):
    """Compress retrieved ``Document`` content with query-aware compression."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    compression_model: str = DEFAULT_MODEL
    target_compression_ratio: float = DEFAULT_RATIO
    min_tokens: int = DEFAULT_MIN_TOKENS
    coarse: Optional[bool] = None
    on_error: ErrorPolicy = DEFAULT_POLICY
    drop_below_min: bool = False
    client: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        self.client = build_client(
            api_key=self.api_key,
            base_url=self.base_url,
            caller="CompresrExtractor",
        )
        return self.client

    def _partition(self, documents: Sequence[Document]) -> tuple[list[int], list[Document]]:
        indices: list[int] = []
        eligible: list[Document] = []
        for i, doc in enumerate(documents):
            if (
                isinstance(doc.page_content, str)
                and estimate_tokens(doc.page_content) >= self.min_tokens
            ):
                indices.append(i)
                eligible.append(doc)
        return indices, eligible

    def _emit(self, doc: Document, new_content: str) -> Document:
        return Document(
            page_content=new_content,
            metadata={**doc.metadata, "compresr": True},
        )

    def _apply_results(
        self,
        out: List[Document],
        documents: Sequence[Document],
        indices: list[int],
        results: list,
        slice_start: int,
        chunk_len: int,
    ) -> None:
        for offset, item in enumerate(results[:chunk_len]):
            global_idx = indices[slice_start + offset]
            new = getattr(item, "compressed_context", None)
            if isinstance(new, str) and new:
                out[global_idx] = self._emit(documents[global_idx], new)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        if not documents:
            return []

        indices, eligible = self._partition(documents)
        if not eligible:
            return list(documents)

        client = self._get_client()
        out: List[Document] = list(documents)

        for start in range(0, len(eligible), BATCH_LIMIT):
            chunk = eligible[start : start + BATCH_LIMIT]
            try:
                resp = client.compress_batch(
                    contexts=[d.page_content for d in chunk],
                    queries=query,
                    compression_model_name=self.compression_model,
                    target_compression_ratio=self.target_compression_ratio,
                    coarse=self.coarse,
                )
            except Exception as exc:  # noqa: BLE001
                if self.on_error == "raise":
                    raise
                logger.warning(
                    "compresr: batch compress failed (%s); passthrough %d docs.",
                    exc,
                    len(chunk),
                )
                continue

            results = (resp.data.results if resp.data else None) or []
            self._apply_results(out, documents, indices, results, start, len(chunk))

        if self.drop_below_min:
            return [d for d in out if d.page_content and d.page_content.strip()]
        return out

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> List[Document]:
        if not documents:
            return []

        indices, eligible = self._partition(documents)
        if not eligible:
            return list(documents)

        client = self._get_client()
        out: List[Document] = list(documents)

        for start in range(0, len(eligible), BATCH_LIMIT):
            chunk = eligible[start : start + BATCH_LIMIT]
            try:
                resp = await client.compress_batch_async(
                    contexts=[d.page_content for d in chunk],
                    queries=query,
                    compression_model_name=self.compression_model,
                    target_compression_ratio=self.target_compression_ratio,
                    coarse=self.coarse,
                )
            except Exception as exc:  # noqa: BLE001
                if self.on_error == "raise":
                    raise
                logger.warning(
                    "compresr: async batch compress failed (%s); passthrough %d docs.",
                    exc,
                    len(chunk),
                )
                continue

            results = (resp.data.results if resp.data else None) or []
            self._apply_results(out, documents, indices, results, start, len(chunk))

        if self.drop_below_min:
            return [d for d in out if d.page_content and d.page_content.strip()]
        return out
