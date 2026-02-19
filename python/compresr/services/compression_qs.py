"""
QSCompressionClient - Question-Specific Context Compression Service

Compresses context text while preserving information relevant to a specific question.
REQUIRES a question parameter for all compression methods.

For general context compression (no question), use CompressionClient instead.
"""

from typing import Generator, List, Optional, Union

from ..schemas import (
    BatchCompressResponse,
    BatchInput,
    CompressResponse,
    StreamChunk,
)
from .base import BaseCompressionClient


class QSCompressionClient(BaseCompressionClient):
    """
    Question-specific compression client - compress context while preserving question-relevant info.

    REQUIRES a question parameter for all compression methods.
    For general compression without a question, use CompressionClient instead.

    Args:
        api_key: Your Compresr API key (required) - "cmp_..."
        base_url: API base URL (optional)
        timeout: Request timeout in seconds (optional)

    Available Models:
        - QS_CMPRSR_V1: Question-specific compression (default)
        - QSR_CMPRSR_V1: Question-specific with reranking
        - QSLF_CMPRSR_V1: Question-specific with longformer

    Example:
        from compresr import QSCompressionClient, MODELS

        client = QSCompressionClient(api_key="cmp_...")

        # Question-specific compression
        response = client.compress(
            context="Your long document with multiple topics...",
            question="What is the main conclusion?"  # REQUIRED
        )
        print(response.data.compressed_context)
    """

    # ==================== Sync ====================

    def compress(  # type: ignore[override]
        self,
        context: str,
        question: str,
        compression_model_name: str = "QS_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> CompressResponse:
        """
        Question-specific compression (sync).

        Compresses context while preserving information relevant to the question.

        Args:
            context: Context text to compress
            question: Question to preserve relevance for (REQUIRED)
            compression_model_name: Question-specific model to use
                - "QS_CMPRSR_V1" (default): Question-specific compression
                - "QSR_CMPRSR_V1": Question-specific with reranking
                - "QSLF_CMPRSR_V1": Question-specific with longformer
            target_compression_ratio: Compression ratio 0.1-0.9 (percentage to REMOVE)

        Returns:
            CompressResponse with compressed context and metrics
        """
        req = self._build_request(
            context, compression_model_name, question, target_compression_ratio
        )
        return self._do_compress(req)

    def compress_batch(  # type: ignore[override]
        self,
        inputs: List[Union[BatchInput, dict]],
        compression_model_name: str = "QS_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> BatchCompressResponse:
        """
        Question-specific batch compression (sync).

        Args:
            inputs: List of inputs to compress, each with context and question.
                    Can be BatchInput objects or dicts: {"context": "...", "question": "..."}
            compression_model_name: Question-specific model to use
            target_compression_ratio: Target ratio (optional)

        Returns:
            BatchCompressResponse with all results and aggregated metrics

        Example:
            response = client.compress_batch([
                {"context": "Machine learning is...", "question": "What is ML?"},
                {"context": "Python was created...", "question": "Who made Python?"},
            ])
        """
        # Convert dicts to BatchInput if needed
        batch_inputs = [BatchInput(**inp) if isinstance(inp, dict) else inp for inp in inputs]
        req = self._build_batch_request(
            batch_inputs, compression_model_name, target_compression_ratio
        )
        return self._do_compress_batch(req)

    def compress_stream(  # type: ignore[override]
        self,
        context: str,
        question: str,
        compression_model_name: str = "QS_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Question-specific stream compression (sync).

        Args:
            context: Context text to compress
            question: Question to preserve relevance for (REQUIRED)
            compression_model_name: Question-specific model to use
            target_compression_ratio: Target ratio (optional)

        Yields:
            StreamChunk objects with compressed content
        """
        req = self._build_request(
            context, compression_model_name, question, target_compression_ratio
        )
        yield from self._do_compress_stream(req)

    # ==================== Async ====================

    async def compress_async(  # type: ignore[override]
        self,
        context: str,
        question: str,
        compression_model_name: str = "QS_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> CompressResponse:
        """
        Question-specific compression (async).

        Args:
            context: Context text to compress
            question: Question to preserve relevance for (REQUIRED)
            compression_model_name: Question-specific model to use
            target_compression_ratio: Target ratio (optional)

        Returns:
            CompressResponse with compressed context and metrics
        """
        req = self._build_request(
            context, compression_model_name, question, target_compression_ratio
        )
        return await self._do_compress_async(req)

    async def compress_batch_async(  # type: ignore[override]
        self,
        inputs: List[Union[BatchInput, dict]],
        compression_model_name: str = "QS_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> BatchCompressResponse:
        """
        Question-specific batch compression (async).

        Args:
            inputs: List of inputs to compress, each with context and question.
                    Can be BatchInput objects or dicts: {"context": "...", "question": "..."}
            compression_model_name: Question-specific model to use
            target_compression_ratio: Target ratio (optional)

        Returns:
            BatchCompressResponse with all results and aggregated metrics
        """
        batch_inputs = [BatchInput(**inp) if isinstance(inp, dict) else inp for inp in inputs]
        req = self._build_batch_request(
            batch_inputs, compression_model_name, target_compression_ratio
        )
        return await self._do_compress_batch_async(req)
