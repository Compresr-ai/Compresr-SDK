"""
CompressionClient - Token-level context compression.

Compresses context text to reduce token count before sending to LLMs.
Supports both agnostic (no query) and query-specific compression.

Models:
    - espresso_v1 (default): Agnostic compression, no query needed
    - latte_v1: Query-specific compression, query REQUIRED
"""

from typing import Generator, List, Optional, Union

from ..config import ALLOWED_COMPRESSION_MODELS, ENDPOINTS, QUERY_REQUIRED_MODELS
from ..exceptions import ValidationError
from ..schemas import (
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

    Supports two models:
    - espresso_v1 (default): Agnostic compression, no query needed
    - latte_v1: Query-specific compression, query REQUIRED

    Args:
        api_key: Your Compresr API key (required) - "cmp_..."
        timeout: Request timeout in seconds (optional)

    Example:
        from compresr import CompressionClient, MODELS

        client = CompressionClient(api_key="cmp_...")

        # Agnostic compression (espresso_v1)
        response = client.compress(context="Your long context text...")
        print(response.data.compressed_context)

        # Query-specific compression (latte_v1)
        response = client.compress(
            context="Your long context text...",
            query="What is the main conclusion?",
            compression_model_name="latte_v1",
        )
    """

    ALLOWED_MODELS = ALLOWED_COMPRESSION_MODELS

    def _validate_query_for_model(self, model_name: str, query: Optional[str]) -> None:
        """Validate query parameter against model requirements."""
        if model_name in QUERY_REQUIRED_MODELS and not query:
            raise ValidationError(f"Model '{model_name}' requires a 'query' parameter.")
        if model_name not in QUERY_REQUIRED_MODELS and query is not None:
            raise ValidationError(
                f"Model '{model_name}' does not accept a 'query' parameter. "
                f"Remove the query or use 'latte_v1' for query-specific compression."
            )

    # ==================== Sync ====================

    def compress(
        self,
        context: Union[str, List[str]],
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressResponse:
        """
        Compress context(s) (sync).

        Args:
            context: Context to compress - single string or list of strings
                - str -> compressed_context is str
                - List[str] -> compressed_context is List[str]
            compression_model_name: Compression model to use
                - "espresso_v1" (default): No query needed
                - "latte_v1": Query REQUIRED
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Ratio 0.1-0.9 (percentage to REMOVE)
            coarse: True for paragraph-level (faster), False/None for token-level (default).
                    Only applicable for latte_v1.

        Returns:
            CompressResponse with compressed context and metrics
        """
        self._validate_query_for_model(compression_model_name, query)
        req = self._build_request(
            context, compression_model_name, query, target_compression_ratio, coarse
        )
        endpoint, _ = self._resolve_endpoints(compression_model_name)
        return self._do_request(endpoint, req)

    def compress_stream(
        self,
        context: str,
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream compression (sync). Only supports single string context.

        Args:
            context: Context text to compress (string only)
            compression_model_name: Compression model to use
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Target ratio (optional)
            coarse: True for paragraph-level (faster), False/None for token-level (default).
                    Only applicable for latte_v1.

        Yields:
            StreamChunk objects with compressed content
        """
        self._validate_query_for_model(compression_model_name, query)
        req = self._build_request(
            context, compression_model_name, query, target_compression_ratio, coarse
        )
        _, stream_endpoint = self._resolve_endpoints(compression_model_name)
        yield from self._do_stream(stream_endpoint, req)

    # ==================== Async ====================

    async def compress_async(
        self,
        context: Union[str, List[str]],
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressResponse:
        """
        Compress context(s) (async).

        Args:
            context: Context to compress - single string or list of strings
            compression_model_name: Compression model to use
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Target ratio (optional)
            coarse: True for paragraph-level (faster), False/None for token-level (default).
                    Only applicable for latte_v1.

        Returns:
            CompressResponse with compressed context and metrics
        """
        self._validate_query_for_model(compression_model_name, query)
        req = self._build_request(
            context, compression_model_name, query, target_compression_ratio, coarse
        )
        endpoint, _ = self._resolve_endpoints(compression_model_name)
        return await self._do_request_async(endpoint, req)

    # ==================== Batch ====================

    def compress_batch(
        self,
        contexts: List[str],
        queries: Union[str, List[str]],
        compression_model_name: str = "latte_v1",
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressBatchResponse:
        """
        Batch compress multiple contexts with queries (sync).

        This is more efficient than calling compress() multiple times
        when you have many documents to compress.

        Args:
            contexts: List of context strings to compress (1-100 items)
            queries: Either:
                - Single query string (same for all contexts)
                - List of queries (one per context, must match contexts length)
            compression_model_name: Compression model (only "latte_v1" supported)
            target_compression_ratio: Target ratio (optional): 0-1 or >1 for Nx
            coarse: True for paragraph-level (faster), False/None for token-level

        Returns:
            CompressBatchResponse with results for each context and aggregated metrics

        Example with same query:
            response = client.compress_batch(
                contexts=["Doc 1...", "Doc 2...", "Doc 3..."],
                queries="What are the key points?",
            )

        Example with different queries:
            response = client.compress_batch(
                contexts=["Doc about ML...", "Doc about NLP...", "Doc about CV..."],
                queries=["What is ML?", "What is NLP?", "What is CV?"],
            )
        """
        if compression_model_name not in QUERY_REQUIRED_MODELS:
            raise ValidationError(
                f"Batch compression only supports query-specific models: {list(QUERY_REQUIRED_MODELS)}"
            )
        self._validate_model(compression_model_name)

        # Handle single query vs list of queries
        if isinstance(queries, str):
            query_list = [queries] * len(contexts)
        else:
            if len(queries) != len(contexts):
                raise ValidationError(
                    f"Number of queries ({len(queries)}) must match number of contexts ({len(contexts)})"
                )
            query_list = queries

        inputs = [CompressBatchInput(context=ctx, query=q) for ctx, q in zip(contexts, query_list)]
        req = CompressBatchRequest(
            inputs=inputs,
            compression_model_name=compression_model_name,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
        )
        data = self.post(ENDPOINTS.COMPRESS_QS_BATCH, req.model_dump(exclude_none=True))
        return CompressBatchResponse.model_validate(data)

    async def compress_batch_async(
        self,
        contexts: List[str],
        queries: Union[str, List[str]],
        compression_model_name: str = "latte_v1",
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressBatchResponse:
        """
        Batch compress multiple contexts with queries (async).

        Args:
            contexts: List of context strings to compress (1-100 items)
            queries: Either:
                - Single query string (same for all contexts)
                - List of queries (one per context, must match contexts length)
            compression_model_name: Compression model (only "latte_v1" supported)
            target_compression_ratio: Target ratio (optional): 0-1 or >1 for Nx
            coarse: True for paragraph-level (faster), False/None for token-level

        Returns:
            CompressBatchResponse with results for each context and aggregated metrics
        """
        if compression_model_name not in QUERY_REQUIRED_MODELS:
            raise ValidationError(
                f"Batch compression only supports query-specific models: {list(QUERY_REQUIRED_MODELS)}"
            )
        self._validate_model(compression_model_name)

        # Handle single query vs list of queries
        if isinstance(queries, str):
            query_list = [queries] * len(contexts)
        else:
            if len(queries) != len(contexts):
                raise ValidationError(
                    f"Number of queries ({len(queries)}) must match number of contexts ({len(contexts)})"
                )
            query_list = queries

        inputs = [CompressBatchInput(context=ctx, query=q) for ctx, q in zip(contexts, query_list)]
        req = CompressBatchRequest(
            inputs=inputs,
            compression_model_name=compression_model_name,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
        )
        data = await self.post_async(ENDPOINTS.COMPRESS_QS_BATCH, req.model_dump(exclude_none=True))
        return CompressBatchResponse.model_validate(data)
