"""
BaseCompressionClient - Abstract base class for compression services.

Contains shared logic for HTTP calls and response handling.
Do not use directly - use CompressionClient or FilterClient.
"""

from typing import ClassVar, FrozenSet, Generator, List, Optional, Tuple, Union

from pydantic import ValidationError as PydanticValidationError

from ..config import AGNOSTIC_ENDPOINT_MODELS, ENDPOINTS, QS_ENDPOINT_MODELS
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

    # Subclasses should override this with their allowed models
    ALLOWED_MODELS: ClassVar[FrozenSet[str]] = frozenset()

    def _validate_model(self, model_name: str) -> None:
        """Validate that the model is allowed for this client type."""
        if self.ALLOWED_MODELS and model_name not in self.ALLOWED_MODELS:
            allowed = ", ".join(sorted(self.ALLOWED_MODELS))
            raise ValidationError(
                f"Model '{model_name}' is not valid for {self.__class__.__name__}. "
                f"Allowed models: {allowed}"
            )

    def _build_request(
        self,
        context: Union[str, List[str]],
        compression_model_name: str,
        query: Optional[str] = None,
        target_compression_ratio: Optional[float] = None,
    ) -> CompressRequest:
        """Build and validate a compression request."""
        self._validate_model(compression_model_name)
        try:
            return CompressRequest(
                context=context,
                compression_model_name=compression_model_name,
                query=query,
                target_compression_ratio=target_compression_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

    def _resolve_endpoints(self, model_name: str) -> Tuple[str, str]:
        """Resolve base and stream endpoints based on model name.

        Returns:
            (base_endpoint, stream_endpoint) tuple
        """
        if model_name in AGNOSTIC_ENDPOINT_MODELS:
            return ENDPOINTS.COMPRESS_AGNOSTIC, ENDPOINTS.COMPRESS_AGNOSTIC_STREAM
        elif model_name in QS_ENDPOINT_MODELS:
            return ENDPOINTS.COMPRESS_QS, ENDPOINTS.COMPRESS_QS_STREAM
        else:
            raise ValidationError(f"Model '{model_name}' does not map to a known endpoint.")

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
