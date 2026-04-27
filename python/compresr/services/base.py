"""
BaseCompressionClient - Abstract base class for compression services.

Contains shared logic for HTTP calls and response handling.
Do not use directly - use CompressionClient or FilterClient.
"""

from typing import Generator, Optional, Tuple

from pydantic import ValidationError as PydanticValidationError

from ..config import ENDPOINTS
from ..exceptions import ValidationError
from ..schemas import (
    CompressRequest,
    CompressResponse,
    StreamChunk,
)
from .proxy import HTTPClient


class BaseCompressionClient(HTTPClient):
    """
    Abstract base class for compression clients.

    Provides shared HTTP functionality and response handling.
    Subclasses (CompressionClient, FilterClient) implement user-facing methods.
    """

    def _build_request(
        self,
        context: str,
        compression_model_name: str,
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
    ) -> CompressRequest:
        """Build and validate a compression request.

        Note: coarse is only included when query is provided (QS endpoint).
        Agnostic endpoint doesn't support coarse parameter.
        """
        try:
            # Only include coarse when using query-specific endpoint
            # Agnostic endpoint doesn't support coarse parameter
            effective_coarse = coarse if query is not None else None

            return CompressRequest(
                context=context,
                compression_model_name=compression_model_name,
                query=query,
                target_compression_ratio=target_compression_ratio,
                coarse=effective_coarse,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

    def _resolve_endpoints(self, model_name: str, query: Optional[str] = None) -> Tuple[str, str]:
        """Resolve base and stream endpoints based on whether query is provided.

        Returns:
            (base_endpoint, stream_endpoint) tuple
        """
        if query is not None:
            return ENDPOINTS.COMPRESS_QS, ENDPOINTS.COMPRESS_QS_STREAM
        else:
            return ENDPOINTS.COMPRESS_AGNOSTIC, ENDPOINTS.COMPRESS_AGNOSTIC_STREAM

    def _do_request(self, endpoint: str, req: CompressRequest) -> CompressResponse:
        """Execute compression request (sync)."""
        data = self.post(endpoint, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)

    def _do_stream(self, endpoint: str, req: CompressRequest) -> Generator[StreamChunk, None, None]:
        """Execute stream compression request (sync)."""
        for content in self.stream(endpoint, req.model_dump(exclude_none=True)):
            yield StreamChunk(content=content, done=False)
        yield StreamChunk(content="", done=True)

    async def _do_request_async(self, endpoint: str, req: CompressRequest) -> CompressResponse:
        """Execute compression request (async)."""
        data = await self.post_async(endpoint, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)
