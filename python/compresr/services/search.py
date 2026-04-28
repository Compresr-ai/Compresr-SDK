"""
SearchClient - Agentic search over pre-indexed knowledge bases.

Uses multi-round LLM reasoning (Opie backbone) to find relevant chunks
from indexed documents based on natural language queries.

Models:
    - macchiato_v1 (default): Agentic search, query + index_name REQUIRED
"""

import json
import time
from typing import Any, ClassVar, Dict, FrozenSet, Generator

from pydantic import ValidationError as PydanticValidationError

from ..config import ALLOWED_SEARCH_MODELS, ENDPOINTS
from ..exceptions import CompresrError, ValidationError
from ..schemas.agentic_search import SearchRequest, SearchResponse
from ..schemas.inference import StreamChunk
from .proxy import HTTPClient


class SearchClient(HTTPClient):
    """
    Agentic search client - search pre-indexed knowledge bases.

    Uses multi-round LLM reasoning to find the most relevant chunks
    from indexed documents based on a natural language query.

    Args:
        api_key: Your Compresr API key (required) - "cmp_..."
        base_url: API base URL (optional) - defaults to https://api.compresr.ai
                  Use for on-prem deployments, e.g., "http://localhost:8000"
        timeout: Request timeout in seconds (optional)

    Available Models:
        - macchiato_v1 (default): Agentic search, query + index_name REQUIRED

    Example:
        from compresr import SearchClient

        # Cloud (default)
        client = SearchClient(api_key="cmp_...")

        # On-prem
        client = SearchClient(api_key="cmp_...", base_url="http://localhost:8000")

        response = client.search(
            query="What is machine learning?",
            index_name="my-knowledge-base",
        )
        print(response.data.chunks)       # List[str] of relevant chunks
        print(response.data.chunks_count)  # Number of chunks found
    """

    ALLOWED_MODELS: ClassVar[FrozenSet[str]] = ALLOWED_SEARCH_MODELS

    def _validate_model(self, model_name: str) -> None:
        """Validate that the model is allowed for search."""
        if model_name not in self.ALLOWED_MODELS:
            allowed = ", ".join(sorted(self.ALLOWED_MODELS))
            raise ValidationError(
                f"Model '{model_name}' is not valid for SearchClient. " f"Allowed models: {allowed}"
            )

    def _build_search_request(
        self,
        query: str,
        index_name: str,
        compression_model_name: str,
        max_time_s: float,
    ) -> SearchRequest:
        """Build and validate a search request."""
        self._validate_model(compression_model_name)
        try:
            return SearchRequest(
                query=query,
                index_name=index_name,
                compression_model_name=compression_model_name,
                max_time_s=max_time_s,
            )
        except PydanticValidationError as e:
            raise ValidationError(str(e)) from e

    # ==================== Sync ====================

    def search(
        self,
        query: str,
        index_name: str,
        compression_model_name: str = "macchiato_v1",
        max_time_s: float = 4.5,
    ) -> SearchResponse:
        """
        Search a pre-indexed knowledge base (sync).

        Args:
            query: Natural language search question (REQUIRED)
            index_name: Name of pre-built index to search (REQUIRED)
            compression_model_name: Search model to use (default: macchiato_v1)
            max_time_s: Maximum search time in seconds (default: 4.5)

        Returns:
            SearchResponse with relevant chunks and metrics
        """
        req = self._build_search_request(query, index_name, compression_model_name, max_time_s)
        data = self.post(ENDPOINTS.SEARCH, req.model_dump(exclude_none=True))
        return SearchResponse.model_validate(data)

    def search_stream(
        self,
        query: str,
        index_name: str,
        compression_model_name: str = "macchiato_v1",
        max_time_s: float = 4.5,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream search results (sync).

        Args:
            query: Natural language search question (REQUIRED)
            index_name: Name of pre-built index to search (REQUIRED)
            compression_model_name: Search model to use (default: macchiato_v1)
            max_time_s: Maximum search time in seconds (default: 4.5)

        Yields:
            StreamChunk objects with search results
        """
        req = self._build_search_request(query, index_name, compression_model_name, max_time_s)
        for content in self.stream(ENDPOINTS.SEARCH_STREAM, req.model_dump(exclude_none=True)):
            yield StreamChunk(content=content, done=False)
        yield StreamChunk(content="", done=True)

    # ==================== Async ====================

    async def search_async(
        self,
        query: str,
        index_name: str,
        compression_model_name: str = "macchiato_v1",
        max_time_s: float = 4.5,
    ) -> SearchResponse:
        """
        Search a pre-indexed knowledge base (async).

        Args:
            query: Natural language search question (REQUIRED)
            index_name: Name of pre-built index to search (REQUIRED)
            compression_model_name: Search model to use (default: macchiato_v1)
            max_time_s: Maximum search time in seconds (default: 4.5)

        Returns:
            SearchResponse with relevant chunks and metrics
        """
        req = self._build_search_request(query, index_name, compression_model_name, max_time_s)
        data = await self.post_async(ENDPOINTS.SEARCH, req.model_dump(exclude_none=True))
        return SearchResponse.model_validate(data)

    # ==================== Health ====================

    def health(self) -> Dict[str, Any]:
        """Check if the agentic search API is healthy.

        Returns:
            Dict with health status
        """
        return self.get(ENDPOINTS.SEARCH_HEALTH)

    # ==================== Index Management (Sync) ====================

    def create_index(self, index_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new search index.

        Args:
            index_name: Name for the new index
            data: Dict with chunks, candidate_questions, source_docs, etc.

        Returns:
            Dict with task_id and status
        """
        endpoint = ENDPOINTS.SEARCH_INDEX_CREATE.format(index_name=index_name)
        payload_bytes = json.dumps(data).encode()
        files = {"data": ("data.json", payload_bytes, "application/json")}
        return self.post_multipart(endpoint, files=files)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of an index build task.

        Args:
            task_id: Task ID returned from create_index

        Returns:
            Dict with status, index_name, num_chunks, etc.
        """
        endpoint = ENDPOINTS.SEARCH_INDEX_TASK.format(task_id=task_id)
        return self.get(endpoint)

    def wait_for_index(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Wait for an index build to complete.

        Args:
            task_id: Task ID returned from create_index
            poll_interval: Seconds between status checks (default: 2.0)
            timeout: Maximum wait time in seconds (default: 300.0)

        Returns:
            Final task status dict

        Raises:
            CompresrError: If build fails or times out
        """
        start = time.time()
        while True:
            status = self.get_task_status(task_id)
            if status.get("status") == "done":
                return status
            if status.get("status") == "failed":
                raise CompresrError(f"Index build failed: {status.get('error', 'unknown')}")
            if time.time() - start > timeout:
                raise CompresrError(f"Index build timed out after {timeout}s")
            time.sleep(poll_interval)

    def delete_index(self, index_name: str) -> Dict[str, Any]:
        """
        Delete an existing search index.

        Args:
            index_name: Name of the index to delete

        Returns:
            Dict with success status
        """
        endpoint = ENDPOINTS.SEARCH_INDEX_DELETE.format(index_name=index_name)
        return self.delete(endpoint)

    # ==================== Index Management (Async) ====================

    async def create_index_async(self, index_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new search index (async). See create_index for details."""
        # Async multipart not yet supported — delegate to sync in thread
        import asyncio

        return await asyncio.to_thread(self.create_index, index_name, data)

    async def get_task_status_async(self, task_id: str) -> Dict[str, Any]:
        """Get the status of an index build task (async)."""
        endpoint = ENDPOINTS.SEARCH_INDEX_TASK.format(task_id=task_id)
        return await self.get_async(endpoint)

    async def wait_for_index_async(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Wait for an index build to complete (async). See wait_for_index for details."""
        import asyncio

        start = time.time()
        while True:
            status = await self.get_task_status_async(task_id)
            if status.get("status") == "done":
                return status
            if status.get("status") == "failed":
                raise CompresrError(f"Index build failed: {status.get('error', 'unknown')}")
            if time.time() - start > timeout:
                raise CompresrError(f"Index build timed out after {timeout}s")
            await asyncio.sleep(poll_interval)

    async def delete_index_async(self, index_name: str) -> Dict[str, Any]:
        """Delete an existing search index (async). See delete_index for details."""
        endpoint = ENDPOINTS.SEARCH_INDEX_DELETE.format(index_name=index_name)
        return await self.delete_async(endpoint)
