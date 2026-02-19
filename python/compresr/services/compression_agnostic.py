"""
CompressionClient - Agnostic Context Compression Service

Compresses context text to reduce token count before sending to LLMs.
Does NOT require a question - use for general context compression.

For question-specific compression, use QSCompressionClient instead.
"""

from typing import Generator, List, Optional

from ..schemas import (
    BatchCompressResponse,
    BatchInput,
    CompressResponse,
    StreamChunk,
)
from .base import BaseCompressionClient


class CompressionClient(BaseCompressionClient):
    """
    Agnostic compression client - compress context to reduce token costs.

    Does NOT require a question parameter. Use for general context compression.
    For question-specific compression, use QSCompressionClient instead.

    Args:
        api_key: Your Compresr API key (required) - "cmp_..."
        base_url: API base URL (optional)
        timeout: Request timeout in seconds (optional)

    Available Models:
        - A_CMPRSR_V1: LLM-based abstractive compression (Qwen3-4B) - default
        - A_CMPRSR_V1_FLASH: Fast extractive compression (LLMLingua-2)

    Example:
        from compresr import CompressionClient, MODELS

        client = CompressionClient(api_key="cmp_...")

        # Default model (A_CMPRSR_V1)
        response = client.compress(context="Your long context text...")

        # Fast model
        response = client.compress(
            context="Your long context text...",
            compression_model_name=MODELS.FAST
        )
        print(response.data.compressed_context)
    """

    # ==================== Sync ====================

    def compress(  # type: ignore[override]
        self,
        context: str,
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> CompressResponse:
        """
        Compress a context (sync).

        Args:
            context: Context text to compress
            compression_model_name: Compression model to use
                - "A_CMPRSR_V1" (default): LLM-based abstractive compression
                - "A_CMPRSR_V1_FLASH": Fast extractive compression
            target_compression_ratio: Compression ratio 0.1-0.9 (percentage to REMOVE)

        Returns:
            CompressResponse with compressed context and metrics
        """
        req = self._build_request(context, compression_model_name, None, target_compression_ratio)
        return self._do_compress(req)

    def compress_batch(  # type: ignore[override]
        self,
        contexts: List[str],
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> BatchCompressResponse:
        """
        Batch compression (sync).

        Args:
            contexts: List of contexts to compress (max 100)
            compression_model_name: Compression model to use
            target_compression_ratio: Target ratio (optional)

        Returns:
            BatchCompressResponse with all results and aggregated metrics
        """
        # Convert contexts to BatchInput format (no question for agnostic)
        inputs = [BatchInput(context=ctx, question=None) for ctx in contexts]
        req = self._build_batch_request(inputs, compression_model_name, target_compression_ratio)
        return self._do_compress_batch(req)

    def compress_stream(  # type: ignore[override]
        self,
        context: str,
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream compression (sync).

        Args:
            context: Context text to compress
            compression_model_name: Compression model to use
            target_compression_ratio: Target ratio (optional)

        Yields:
            StreamChunk objects with compressed content
        """
        req = self._build_request(context, compression_model_name, None, target_compression_ratio)
        yield from self._do_compress_stream(req)

    # ==================== Async ====================

    async def compress_async(  # type: ignore[override]
        self,
        context: str,
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> CompressResponse:
        """
        Compress a context (async).

        Args:
            context: Context text to compress
            compression_model_name: Compression model to use
            target_compression_ratio: Target ratio (optional)

        Returns:
            CompressResponse with compressed context and metrics
        """
        req = self._build_request(context, compression_model_name, None, target_compression_ratio)
        return await self._do_compress_async(req)

    async def compress_batch_async(  # type: ignore[override]
        self,
        contexts: List[str],
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> BatchCompressResponse:
        """
        Batch compression (async).

        Args:
            contexts: List of contexts to compress (max 100)
            compression_model_name: Compression model to use
            target_compression_ratio: Target ratio (optional)

        Returns:
            BatchCompressResponse with all results and aggregated metrics
        """
        inputs = [BatchInput(context=ctx, question=None) for ctx in contexts]
        req = self._build_batch_request(inputs, compression_model_name, target_compression_ratio)
        return await self._do_compress_batch_async(req)
