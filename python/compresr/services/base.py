"""Shared base for compression clients."""

from typing import Generator, Optional

from pydantic import ValidationError as PydanticValidationError

from ..config import ENDPOINTS
from ..exceptions import ValidationError
from ..schemas import CompressRequest, CompressResponse, StreamChunk
from .proxy import HTTPClient


class BaseCompressionClient(HTTPClient):
    def _build_request(
        self,
        context: str,
        query: Optional[str] = None,
        compression_model_name: str = "latte_v1",
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
        heuristic_chunking: Optional[bool] = None,
        disable_placeholders: Optional[bool] = None,
        dynamic: Optional[bool] = None,
        dynamic_min_ratio: Optional[float] = None,
        dynamic_max_ratio: Optional[float] = None,
    ) -> CompressRequest:
        try:
            return CompressRequest(
                context=context,
                query=query,
                compression_model_name=compression_model_name,
                target_compression_ratio=target_compression_ratio,
                coarse=coarse,
                heuristic_chunking=heuristic_chunking,
                disable_placeholders=disable_placeholders,
                dynamic=dynamic,
                dynamic_min_ratio=dynamic_min_ratio,
                dynamic_max_ratio=dynamic_max_ratio,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

    def _do_request(self, req: CompressRequest) -> CompressResponse:
        data = self.post(ENDPOINTS.COMPRESS, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)

    def _do_stream(self, req: CompressRequest) -> Generator[StreamChunk, None, None]:
        for content in self.stream(ENDPOINTS.COMPRESS_STREAM, req.model_dump(exclude_none=True)):
            yield StreamChunk(content=content, done=False)
        yield StreamChunk(content="", done=True)

    async def _do_compress_async(self, req: CompressRequest) -> CompressResponse:
        data = await self.post_async(ENDPOINTS.COMPRESS, req.model_dump(exclude_none=True))
        return CompressResponse.model_validate(data)
