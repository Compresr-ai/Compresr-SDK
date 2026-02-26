"""
CompressionClient - Token-level context compression.

Compresses context text to reduce token count before sending to LLMs.
Supports both agnostic (no query) and query-specific compression.

Models:
    - espresso_v1 (default): Agnostic compression, no query needed
    - latte_v1: Query-specific compression, query REQUIRED
"""

from typing import Generator, List, Optional, Union

from ..config import ALLOWED_COMPRESSION_MODELS, QUERY_REQUIRED_MODELS
from ..exceptions import ValidationError
from ..schemas import CompressResponse, StreamChunk
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
            raise ValidationError(
                f"Model '{model_name}' requires a 'query' parameter."
            )
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

        Returns:
            CompressResponse with compressed context and metrics
        """
        self._validate_query_for_model(compression_model_name, query)
        req = self._build_request(context, compression_model_name, query, target_compression_ratio)
        endpoint, _ = self._resolve_endpoints(compression_model_name)
        return self._do_request(endpoint, req)

    def compress_stream(
        self,
        context: str,
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream compression (sync). Only supports single string context.

        Args:
            context: Context text to compress (string only)
            compression_model_name: Compression model to use
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Target ratio (optional)

        Yields:
            StreamChunk objects with compressed content
        """
        self._validate_query_for_model(compression_model_name, query)
        req = self._build_request(context, compression_model_name, query, target_compression_ratio)
        _, stream_endpoint = self._resolve_endpoints(compression_model_name)
        yield from self._do_stream(stream_endpoint, req)

    # ==================== Async ====================

    async def compress_async(
        self,
        context: Union[str, List[str]],
        compression_model_name: str = "espresso_v1",
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
    ) -> CompressResponse:
        """
        Compress context(s) (async).

        Args:
            context: Context to compress - single string or list of strings
            compression_model_name: Compression model to use
            query: Query for query-specific compression (required for latte_v1)
            target_compression_ratio: Target ratio (optional)

        Returns:
            CompressResponse with compressed context and metrics
        """
        self._validate_query_for_model(compression_model_name, query)
        req = self._build_request(context, compression_model_name, query, target_compression_ratio)
        endpoint, _ = self._resolve_endpoints(compression_model_name)
        return await self._do_request_async(endpoint, req)
