"""
CompressionClient - Context Compression Service

Compresses context text to reduce token count before sending to LLMs.
Use when you want to save costs on long contexts.
"""

from typing import Generator, List, Optional

from pydantic import ValidationError as PydanticValidationError

from ..config import ENDPOINTS
from ..exceptions import ValidationError
from ..schemas import (
    BatchCompressRequest,
    BatchCompressResponse,
    CompressRequest,
    CompressResponse,
    StreamChunk,
)
from .proxy import HTTPClient


class CompressionClient(HTTPClient):
    """
    Compression client - compress context to reduce token costs.

    Args:
        api_key: Your Compresr API key (required) - "cmp_..."
        base_url: API base URL (optional)
        timeout: Request timeout in seconds (optional)

    Example:
        client = CompressionClient(api_key="cmp_...")

        response = client.compress(
            context="Your long context text...",
            compression_model_name="A_CMPRSR_V1"
        )
        print(response.data.compressed_context)
    """

    # ==================== Sync ====================

    def compress(
        self,
        context: str,
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> CompressResponse:
        """
        Compress a context (sync).

        Args:
            context: Context text to compress
            compression_model_name: Compression model to use (default: "A_CMPRSR_V1")
            target_compression_ratio: Compression ratio 0.1-0.9 (percentage to REMOVE - e.g., 0.3 removes 30%)

        Returns:
            CompressResponse with compressed context and metrics
        """
        try:
            req = CompressRequest(
                context=context,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

        data = self.post(ENDPOINTS.COMPRESS, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)

    def compress_batch(
        self,
        contexts: List[str],
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> BatchCompressResponse:
        """
        Batch compression (sync).

        Args:
            contexts: List of contexts to compress (max 100)
            compression_model_name: Compression model to use (default: "A_CMPRSR_V1")
            target_compression_ratio: Target ratio (optional)

        Returns:
            BatchCompressResponse with all results and aggregated metrics
        """
        try:
            req = BatchCompressRequest(
                contexts=contexts,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

        data = self.post(ENDPOINTS.COMPRESS_BATCH, req.model_dump(exclude_none=True))
        return BatchCompressResponse.model_validate(data)

    def compress_stream(
        self,
        context: str,
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream compression (sync).

        Args:
            context: Context text to compress
            compression_model_name: Compression model to use (default: "A_CMPRSR_V1")
            target_compression_ratio: Target ratio (optional)

        Yields:
            StreamChunk objects with compressed content
        """
        try:
            req = CompressRequest(
                context=context,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

        for content in self.stream(ENDPOINTS.COMPRESS_STREAM, req.model_dump(exclude_none=True)):
            yield StreamChunk(content=content, done=False)
        yield StreamChunk(content="", done=True)

    # ==================== Async ====================

    async def compress_async(
        self,
        context: str,
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> CompressResponse:
        """
        Compress a context (async).

        Args:
            context: Context text to compress
            compression_model_name: Compression model to use (default: "A_CMPRSR_V1")
            target_compression_ratio: Target ratio (optional)

        Returns:
            CompressResponse with compressed context and metrics
        """
        try:
            req = CompressRequest(
                context=context,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

        data = await self.post_async(ENDPOINTS.COMPRESS, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)

    async def compress_batch_async(
        self,
        contexts: List[str],
        compression_model_name: str = "A_CMPRSR_V1",
        target_compression_ratio: Optional[float] = None,
    ) -> BatchCompressResponse:
        """
        Batch compression (async).

        Args:
            contexts: List of contexts to compress (max 100)
            compression_model_name: Compression model to use (default: "A_CMPRSR_V1")
            target_compression_ratio: Target ratio (optional)

        Returns:
            BatchCompressResponse with all results and aggregated metrics
        """
        try:
            req = BatchCompressRequest(
                contexts=contexts,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

        data = await self.post_async(ENDPOINTS.COMPRESS_BATCH, req.model_dump(exclude_none=True))
        return BatchCompressResponse.model_validate(data)
