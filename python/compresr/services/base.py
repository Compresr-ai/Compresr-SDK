"""
BaseCompressionClient - Abstract base class for compression services.

Contains shared logic for HTTP calls and response handling.
Do not use directly - use CompressionClient or QSCompressionClient.
"""

from abc import ABC, abstractmethod
from typing import Any, Generator, List, Optional

from pydantic import ValidationError as PydanticValidationError

from ..config import ENDPOINTS
from ..exceptions import ValidationError
from ..schemas import (
    BatchCompressRequest,
    BatchCompressResponse,
    BatchInput,
    CompressRequest,
    CompressResponse,
    StreamChunk,
)
from .proxy import HTTPClient


class BaseCompressionClient(HTTPClient, ABC):
    """
    Abstract base class for compression clients.

    Provides shared HTTP functionality and response handling.
    Subclasses implement compress(), compress_batch(), compress_stream() methods.
    """

    def _build_request(
        self,
        context: str,
        compression_model_name: str,
        question: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
    ) -> CompressRequest:
        """Build and validate a compression request."""
        try:
            return CompressRequest(
                context=context,
                compression_model_name=compression_model_name,
                question=question,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

    def _build_batch_request(
        self,
        inputs: List[BatchInput],
        compression_model_name: str,
        target_compression_ratio: Optional[float] = None,
    ) -> BatchCompressRequest:
        """Build and validate a batch compression request."""
        try:
            return BatchCompressRequest(
                inputs=inputs,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

    def _do_compress(self, req: CompressRequest) -> CompressResponse:
        """Execute compression request (sync)."""
        data = self.post(ENDPOINTS.COMPRESS, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)

    def _do_compress_batch(self, req: BatchCompressRequest) -> BatchCompressResponse:
        """Execute batch compression request (sync)."""
        data = self.post(ENDPOINTS.COMPRESS_BATCH, req.model_dump(exclude_none=True))
        return BatchCompressResponse.model_validate(data)

    def _do_compress_stream(self, req: CompressRequest) -> Generator[StreamChunk, None, None]:
        """Execute stream compression request (sync)."""
        for content in self.stream(ENDPOINTS.COMPRESS_STREAM, req.model_dump(exclude_none=True)):
            yield StreamChunk(content=content, done=False)
        yield StreamChunk(content="", done=True)

    async def _do_compress_async(self, req: CompressRequest) -> CompressResponse:
        """Execute compression request (async)."""
        data = await self.post_async(ENDPOINTS.COMPRESS, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)

    async def _do_compress_batch_async(self, req: BatchCompressRequest) -> BatchCompressResponse:
        """Execute batch compression request (async)."""
        data = await self.post_async(ENDPOINTS.COMPRESS_BATCH, req.model_dump(exclude_none=True))
        return BatchCompressResponse.model_validate(data)

    # Abstract methods - subclasses must implement
    @abstractmethod
    def compress(self, context: str, **kwargs: Any) -> CompressResponse:
        """Compress a context."""
        pass

    @abstractmethod
    def compress_batch(self, contexts: List[str], **kwargs: Any) -> BatchCompressResponse:
        """Batch compress multiple contexts."""
        pass

    @abstractmethod
    def compress_stream(self, context: str, **kwargs: Any) -> Generator[StreamChunk, None, None]:
        """Stream compression."""
        pass

    @abstractmethod
    async def compress_async(self, context: str, **kwargs: Any) -> CompressResponse:
        """Compress a context (async)."""
        pass

    @abstractmethod
    async def compress_batch_async(
        self, contexts: List[str], **kwargs: Any
    ) -> BatchCompressResponse:
        """Batch compress multiple contexts (async)."""
        pass
