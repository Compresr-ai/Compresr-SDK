"""
FilterClient - Coarse-grained chunk-level filtering.

Keeps or drops entire chunks based on query relevance.
For pre-chunked content where you want to select relevant chunks.

Models:
    - coldbrew_v1 (default): Query REQUIRED, no compression_ratio
"""

from typing import Generator, List, Optional

from ..config import ALLOWED_FILTER_MODELS
from ..exceptions import ValidationError
from ..schemas import CompressResponse, StreamChunk
from .base import BaseCompressionClient


class FilterClient(BaseCompressionClient):
    """
    Chunk-level filter client - select relevant chunks from pre-chunked content.

    Keeps or drops entire chunks based on query relevance.
    Query is always required. Does not support compression_ratio.

    Args:
        api_key: Your Compresr API key (required) - "cmp_..."
        timeout: Request timeout in seconds (optional)

    Available Models:
        - coldbrew_v1 (default): Chunk-level filter, query REQUIRED

    Example:
        from compresr import FilterClient, MODELS

        client = FilterClient(api_key="cmp_...")

        response = client.filter(
            chunks=["Chunk about Python...", "Chunk about Java...", "Chunk about ML..."],
            query="What is Python?",
        )
        print(response.data.compressed_context)  # List[str] of relevant chunks
    """

    ALLOWED_MODELS = ALLOWED_FILTER_MODELS

    def _validate_filter_params(
        self, model_name: str, target_compression_ratio: Optional[float]
    ) -> None:
        """Validate that filter models don't receive compression_ratio."""
        if target_compression_ratio is not None:
            raise ValidationError(
                f"Model '{model_name}' is a chunk-level filter and does not support "
                f"'target_compression_ratio'. Remove this parameter for filter models."
            )

    # ==================== Sync ====================

    def filter(
        self,
        chunks: List[str],
        query: str,
        compression_model_name: str = "coldbrew_v1",
    ) -> CompressResponse:
        """
        Filter chunks by query relevance (sync).

        Args:
            chunks: List of text chunks to filter
            query: Query to filter by (REQUIRED)
            compression_model_name: Filter model to use (default: coldbrew_v1)

        Returns:
            CompressResponse with filtered chunks and metrics
        """
        self._validate_filter_params(compression_model_name, None)
        req = self._build_request(chunks, compression_model_name, query, None)
        endpoint, _ = self._resolve_endpoints(compression_model_name)
        return self._do_request(endpoint, req)

    def filter_stream(
        self,
        chunks: List[str],
        query: str,
        compression_model_name: str = "coldbrew_v1",
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream filtered chunks (sync).

        Args:
            chunks: List of text chunks to filter
            query: Query to filter by (REQUIRED)
            compression_model_name: Filter model to use (default: coldbrew_v1)

        Yields:
            StreamChunk objects with filtered content
        """
        self._validate_filter_params(compression_model_name, None)
        req = self._build_request(chunks, compression_model_name, query, None)
        _, stream_endpoint = self._resolve_endpoints(compression_model_name)
        yield from self._do_stream(stream_endpoint, req)

    # ==================== Async ====================

    async def filter_async(
        self,
        chunks: List[str],
        query: str,
        compression_model_name: str = "coldbrew_v1",
    ) -> CompressResponse:
        """
        Filter chunks by query relevance (async).

        Args:
            chunks: List of text chunks to filter
            query: Query to filter by (REQUIRED)
            compression_model_name: Filter model to use (default: coldbrew_v1)

        Returns:
            CompressResponse with filtered chunks and metrics
        """
        self._validate_filter_params(compression_model_name, None)
        req = self._build_request(chunks, compression_model_name, query, None)
        endpoint, _ = self._resolve_endpoints(compression_model_name)
        return await self._do_request_async(endpoint, req)
