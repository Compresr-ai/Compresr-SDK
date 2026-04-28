"""
CompressionClient - Token-level context compression.

Compresses context text to reduce token count before sending to LLMs.
Supports both agnostic (no query) and query-specific compression.

Models:
    - espresso_v1 (default): Agnostic compression, no query needed
    - latte_v1: Query-specific compression, query REQUIRED

Endpoints:
    Single:
        /compress/question-agnostic/     - context: str (no query)
        /compress/question-specific/     - context: str, query: str
    Batch:
        /compress/question-agnostic/batch - inputs: List[{context}]
        /compress/question-specific/batch - inputs: List[{context, query}]
    Stream:
        /compress/question-agnostic/stream - context: str
        /compress/question-specific/stream - context: str, query: str
"""

from typing import Generator, List, Optional, Union

from ..config import ENDPOINTS
from ..exceptions import ValidationError
from ..schemas import (
    AgnosticBatchInput,
    AgnosticBatchRequest,
    CompressBatchInput,
    CompressBatchRequest,
    CompressBatchResponse,
    CompressResponse,
    StreamChunk,
)
from .base import BaseCompressionClient


class CompressionClient(BaseCompressionClient):
    """
    Token-level compression client - compress context to reduce token costs.

    Models:
    - espresso_v1 (default): Agnostic compression, no query needed
    - latte_v1: Query-specific compression, query REQUIRED

    Args:
        api_key: Your Compresr API key (required) - "cmp_..."
        base_url: API base URL (optional) - defaults to https://api.compresr.ai
                  Use for on-prem deployments, e.g., "http://localhost:8000"
        timeout: Request timeout in seconds (optional)

    Example:
        from compresr import CompressionClient

        client = CompressionClient(api_key="cmp_...")

        # Single agnostic compression
        response = client.compress(context="Your long context text...")

        # Single query-specific compression
        response = client.compress(
            context="Your long context text...",
            query="What is the main conclusion?",
            compression_model_name="latte_v1",
        )

        # Batch agnostic compression
        response = client.compress_batch(
            contexts=["Doc 1...", "Doc 2...", "Doc 3..."],
        )

        # Batch query-specific compression
        response = client.compress_batch(
            contexts=["Doc 1...", "Doc 2...", "Doc 3..."],
            queries="What are the key points?",
            compression_model_name="latte_v1",
        )
    """

    # ==================== Single Compression ====================

    def compress(
        self,
        context: str,
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressResponse:
        """
        Compress a single context (sync).

        For multiple contexts, use compress_batch().

        Args:
            context: Context text to compress (single string)
            compression_model_name: Compression model to use
                - "espresso_v1" (default): No query needed
                - "latte_v1": Query REQUIRED
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Ratio 0.1-0.9 (percentage to REMOVE)
            coarse: Paragraph-level compression (only for query-specific with latte_v1).
                    - True: faster, paragraph-level (default for latte_v1)
                    - False: slower, token-level (finer grained)
                    - None: use backend default
                    Ignored for agnostic compression (no query).

        Returns:
            CompressResponse with compressed context and metrics
        """
        req = self._build_request(
            context, compression_model_name, query, target_compression_ratio, coarse
        )
        endpoint, _ = self._resolve_endpoints(compression_model_name, query)
        return self._do_request(endpoint, req)

    async def compress_async(
        self,
        context: str,
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressResponse:
        """
        Compress a single context (async).

        For multiple contexts, use compress_batch_async().

        Args:
            context: Context text to compress (single string)
            compression_model_name: Compression model to use
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Target ratio (optional)
            coarse: Paragraph-level compression (only for query-specific with latte_v1).
                    Ignored for agnostic compression (no query).

        Returns:
            CompressResponse with compressed context and metrics
        """
        req = self._build_request(
            context, compression_model_name, query, target_compression_ratio, coarse
        )
        endpoint, _ = self._resolve_endpoints(compression_model_name, query)
        return await self._do_request_async(endpoint, req)

    def compress_stream(
        self,
        context: str,
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream compression (sync).

        Args:
            context: Context text to compress (single string)
            compression_model_name: Compression model to use
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Target ratio (optional)
            coarse: Paragraph-level compression (only for query-specific with latte_v1).
                    Ignored for agnostic compression (no query).

        Yields:
            StreamChunk objects with compressed content
        """
        req = self._build_request(
            context, compression_model_name, query, target_compression_ratio, coarse
        )
        _, stream_endpoint = self._resolve_endpoints(compression_model_name, query)
        yield from self._do_stream(stream_endpoint, req)

    # ==================== Batch Compression ====================

    def compress_batch(
        self,
        contexts: List[str],
        queries: Optional[Union[str, List[str]]] = None,
        compression_model_name: str = "espresso_v1",
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressBatchResponse:
        """
        Batch compress multiple contexts (sync).

        - If queries is None: uses agnostic endpoint (no queries required)
        - If queries is provided: uses query-specific endpoint

        Args:
            contexts: List of context strings to compress (1-100 items)
            queries: Either:
                - None: agnostic compression (no queries)
                - Single query string (same for all contexts)
                - List of queries (one per context, must match contexts length)
            compression_model_name: Compression model to use
            target_compression_ratio: Target ratio (optional): 0-1 or >1 for Nx
            coarse: Paragraph-level compression (only for query-specific batch).
                    Ignored for agnostic batch (queries=None).

        Returns:
            CompressBatchResponse with results for each context and aggregated metrics

        Example - agnostic batch:
            response = client.compress_batch(
                contexts=["Doc 1...", "Doc 2...", "Doc 3..."],
            )

        Example - query-specific batch (same query):
            response = client.compress_batch(
                contexts=["Doc 1...", "Doc 2...", "Doc 3..."],
                queries="What are the key points?",
                compression_model_name="latte_v1",
            )

        Example - query-specific batch (different queries):
            response = client.compress_batch(
                contexts=["ML doc...", "NLP doc...", "CV doc..."],
                queries=["What is ML?", "What is NLP?", "What is CV?"],
                compression_model_name="latte_v1",
            )
        """
        if queries is None:
            # Agnostic batch (no queries)
            agnostic_inputs = [AgnosticBatchInput(context=ctx) for ctx in contexts]
            agnostic_req = AgnosticBatchRequest(
                inputs=agnostic_inputs,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
            data = self.post(
                ENDPOINTS.COMPRESS_AGNOSTIC_BATCH, agnostic_req.model_dump(exclude_none=True)
            )
        else:
            # Query-specific batch
            if isinstance(queries, str):
                query_list = [queries] * len(contexts)
            else:
                if len(queries) != len(contexts):
                    raise ValidationError(
                        f"Number of queries ({len(queries)}) must match number of contexts ({len(contexts)})"
                    )
                query_list = queries

            qs_inputs = [
                CompressBatchInput(context=ctx, query=q) for ctx, q in zip(contexts, query_list)
            ]
            qs_req = CompressBatchRequest(
                inputs=qs_inputs,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
                coarse=coarse,
            )
            data = self.post(ENDPOINTS.COMPRESS_QS_BATCH, qs_req.model_dump(exclude_none=True))

        return CompressBatchResponse.model_validate(data)

    async def compress_batch_async(
        self,
        contexts: List[str],
        queries: Optional[Union[str, List[str]]] = None,
        compression_model_name: str = "espresso_v1",
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressBatchResponse:
        """
        Batch compress multiple contexts (async).

        - If queries is None: uses agnostic endpoint (no queries required)
        - If queries is provided: uses query-specific endpoint

        Args:
            contexts: List of context strings to compress (1-100 items)
            queries: Either:
                - None: agnostic compression (no queries)
                - Single query string (same for all contexts)
                - List of queries (one per context, must match contexts length)
            compression_model_name: Compression model to use
            target_compression_ratio: Target ratio (optional): 0-1 or >1 for Nx
            coarse: Paragraph-level compression (only for query-specific batch).
                    Ignored for agnostic batch (queries=None).

        Returns:
            CompressBatchResponse with results for each context and aggregated metrics
        """
        if queries is None:
            # Agnostic batch (no queries)
            agnostic_inputs = [AgnosticBatchInput(context=ctx) for ctx in contexts]
            agnostic_req = AgnosticBatchRequest(
                inputs=agnostic_inputs,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
            data = await self.post_async(
                ENDPOINTS.COMPRESS_AGNOSTIC_BATCH, agnostic_req.model_dump(exclude_none=True)
            )
        else:
            # Query-specific batch
            if isinstance(queries, str):
                query_list = [queries] * len(contexts)
            else:
                if len(queries) != len(contexts):
                    raise ValidationError(
                        f"Number of queries ({len(queries)}) must match number of contexts ({len(contexts)})"
                    )
                query_list = queries

            qs_inputs = [
                CompressBatchInput(context=ctx, query=q) for ctx, q in zip(contexts, query_list)
            ]
            qs_req = CompressBatchRequest(
                inputs=qs_inputs,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
                coarse=coarse,
            )
            data = await self.post_async(
                ENDPOINTS.COMPRESS_QS_BATCH, qs_req.model_dump(exclude_none=True)
            )

        return CompressBatchResponse.model_validate(data)
