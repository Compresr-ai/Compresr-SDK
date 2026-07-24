"""``CompressionClient`` — query-aware context compression.

Only the question-specific endpoints are exposed; pass any
``compression_model_name`` you've enabled on the backend.
"""

from typing import Any, Dict, Generator, List, Optional, Union

from ..config import ENDPOINTS
from ..exceptions import ValidationError
from ..retry import RetryConfig
from ..schemas import (
    CompressBatchInput,
    CompressBatchRequest,
    CompressBatchResponse,
    CompressResponse,
    StreamChunk,
)
from .base import BaseCompressionClient

BatchInput = Union[CompressBatchInput, Dict[str, Any]]


class CompressionClient(BaseCompressionClient):
    """Compresr compression client.

    Args:
        api_key: ``cmp_...`` API key.
        base_url: API base URL (defaults to ``https://api.compresr.ai``).
        timeout: Request timeout in seconds.
        llm: Optional provider selector to opt into the provider-shape
            facades (``.messages``, ``.chat``, ``.run``). Use just the
            provider (``"anthropic"``) and pass ``model=`` at the call
            site, or pin a default with ``"anthropic:claude-haiku-4-5"``.
        llm_api_key: Optional provider API key forwarded to the LLM.
        compression: Optional dict of :class:`CompressionPolicy` kwargs
            (e.g. ``{"target_compression_ratio": 0.7, "min_tokens": 1000}``).
        enable_prompt_cache: Provider-aware prompt-cache control. Default ``True``.
            Anthropic → wires ``AnthropicPromptCachingMiddleware``.
            OpenAI → attaches ``prompt_cache_key`` (when set) and maps
            ``prompt_cache_ttl="1h"`` to ``prompt_cache_retention="24h"``.
            Gemini → no-op for now (implicit caching always on at the API).
        prompt_cache_ttl: ``"5m"`` or ``"1h"``. Anthropic uses it as the
            ephemeral cache TTL. OpenAI maps ``"1h"`` to the 24h retention tier.
        prompt_cache_min_messages: Skip Anthropic cache stamping until the
            conversation has at least this many messages.
        openai_prompt_cache_key: Optional routing key for OpenAI prompt caching
            (improves hit rate when multiple clients share a backend). Ignored
            for non-OpenAI providers.
        llm_http_client: Optional ``httpx.Client`` for the LLM provider transport
            (Anthropic/OpenAI). Use for corporate proxies, custom CA bundles, mTLS,
            or ``verify`` settings — e.g. ``httpx.Client(verify="/path/corp-ca.pem")``.
            This is for the downstream LLM calls, not the Compresr API transport.
            Requires langchain-anthropic with ``http_client`` support; OpenAI
            already supports it.
        llm_http_async_client: Optional ``httpx.AsyncClient``, the async counterpart
            to ``llm_http_client``.

    Example::

        from compresr import CompressionClient
        client = CompressionClient(api_key="cmp_...")
        result = client.compress(
            context="Long passage...",
            query="What is the main conclusion?",
        )

        # Opt into the Anthropic-shaped facade — model lives at the call site:
        agent = CompressionClient(
            api_key="cmp_...",
            llm="anthropic",
            llm_api_key="sk-ant-...",
        )
        msg = agent.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "hi"}],
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        *,
        retry_config: Optional[RetryConfig] = None,
        llm: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        compression: Optional[Dict[str, Any]] = None,
        enable_prompt_cache: bool = True,
        prompt_cache_ttl: str = "5m",
        prompt_cache_min_messages: int = 2,
        openai_prompt_cache_key: Optional[str] = None,
        llm_http_client: Any = None,
        llm_http_async_client: Any = None,
    ):
        super().__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, retry_config=retry_config
        )
        self._engine = None
        if llm is not None:
            # Lazy import: ``CompressionClient(api_key=...)`` without ``llm=``
            # must keep working when langchain isn't installed.
            from compresr.agents.engine import _Engine
            from compresr.integrations._shared import CompressionPolicy

            policy = CompressionPolicy(**(compression or {}))
            self._engine = _Engine(
                compresr_client=self,
                llm=llm,
                llm_api_key=llm_api_key,
                policy=policy,
                enable_prompt_cache=enable_prompt_cache,
                prompt_cache_ttl=prompt_cache_ttl,
                prompt_cache_min_messages=prompt_cache_min_messages,
                openai_prompt_cache_key=openai_prompt_cache_key,
                llm_http_client=llm_http_client,
                llm_http_async_client=llm_http_async_client,
            )
        self._research_facade: Optional[Any] = None

    @property
    def research(self) -> Any:
        """Research facade: ``client.research.run("question")``. Requires ``llm=``."""
        if self._research_facade is None:
            if self._engine is None:
                from ..exceptions import CompresrError

                raise CompresrError(
                    "client.research requires an LLM provider. "
                    "Construct with CompressionClient(api_key=..., llm='anthropic:...', llm_api_key=...).",
                    code="missing_llm",
                )
            from ..agents.research.facade import ResearchFacade

            self._research_facade = ResearchFacade(self._engine)
        return self._research_facade

    def compress(
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
    ) -> CompressResponse:
        req = self._build_request(
            context,
            query,
            compression_model_name,
            target_compression_ratio,
            coarse,
            heuristic_chunking,
            disable_placeholders,
            dynamic,
            dynamic_min_ratio,
            dynamic_max_ratio,
        )
        return self._do_request(req)

    async def compress_async(
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
    ) -> CompressResponse:
        req = self._build_request(
            context,
            query,
            compression_model_name,
            target_compression_ratio,
            coarse,
            heuristic_chunking,
            disable_placeholders,
            dynamic,
            dynamic_min_ratio,
            dynamic_max_ratio,
        )
        return await self._do_compress_async(req)

    def compress_stream(
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
    ) -> Generator[StreamChunk, None, None]:
        req = self._build_request(
            context,
            query,
            compression_model_name,
            target_compression_ratio,
            coarse,
            heuristic_chunking,
            disable_placeholders,
            dynamic,
            dynamic_min_ratio,
            dynamic_max_ratio,
        )
        yield from self._do_stream(req)

    def compress_batch(
        self,
        contexts: Optional[List[str]] = None,
        queries: Optional[Union[str, List[str]]] = None,
        inputs: Optional[List[BatchInput]] = None,
        compression_model_name: str = "latte_v1",
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
        heuristic_chunking: Optional[bool] = None,
        disable_placeholders: Optional[bool] = None,
        dynamic: Optional[bool] = None,
        dynamic_min_ratio: Optional[float] = None,
        dynamic_max_ratio: Optional[float] = None,
    ) -> CompressBatchResponse:
        """Batch compress up to 100 contexts.

        Two equivalent input forms:

        - **Convenience form**: ``contexts=[...]`` plus ``queries`` as a single
          string (applied to all contexts) or a list (one query per context).
        - **Pair form** (matches the wire format): ``inputs=[{"context": ...,
          "query": ...}, ...]``.

        Pass exactly one of ``inputs`` or ``contexts``.
        """
        items = self._build_batch_inputs(contexts, queries, inputs)
        req = CompressBatchRequest(
            inputs=items,
            compression_model_name=compression_model_name,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            heuristic_chunking=heuristic_chunking,
            disable_placeholders=disable_placeholders,
            dynamic=dynamic,
            dynamic_min_ratio=dynamic_min_ratio,
            dynamic_max_ratio=dynamic_max_ratio,
        )
        data = self.post(ENDPOINTS.COMPRESS_BATCH, req.model_dump(exclude_none=True))
        return CompressBatchResponse.model_validate(data)

    async def compress_batch_async(
        self,
        contexts: Optional[List[str]] = None,
        queries: Optional[Union[str, List[str]]] = None,
        inputs: Optional[List[BatchInput]] = None,
        compression_model_name: str = "latte_v1",
        target_compression_ratio: Optional[float] = None,
        coarse: Optional[bool] = None,
        heuristic_chunking: Optional[bool] = None,
        disable_placeholders: Optional[bool] = None,
        dynamic: Optional[bool] = None,
        dynamic_min_ratio: Optional[float] = None,
        dynamic_max_ratio: Optional[float] = None,
    ) -> CompressBatchResponse:
        items = self._build_batch_inputs(contexts, queries, inputs)
        req = CompressBatchRequest(
            inputs=items,
            compression_model_name=compression_model_name,
            target_compression_ratio=target_compression_ratio,
            coarse=coarse,
            heuristic_chunking=heuristic_chunking,
            disable_placeholders=disable_placeholders,
            dynamic=dynamic,
            dynamic_min_ratio=dynamic_min_ratio,
            dynamic_max_ratio=dynamic_max_ratio,
        )
        data = await self.post_async(ENDPOINTS.COMPRESS_BATCH, req.model_dump(exclude_none=True))
        return CompressBatchResponse.model_validate(data)

    @staticmethod
    def _build_batch_inputs(
        contexts: Optional[List[str]],
        queries: Optional[Union[str, List[str]]],
        inputs: Optional[List[BatchInput]],
    ) -> List[CompressBatchInput]:
        if inputs is not None and contexts is not None:
            raise ValidationError("Pass `inputs` OR `contexts`, not both.")
        if inputs is not None:
            return [
                item if isinstance(item, CompressBatchInput) else CompressBatchInput(**item)
                for item in inputs
            ]
        if contexts is None:
            raise ValidationError("Must provide either `inputs` or `contexts`.")
        query_list = CompressionClient._resolve_query_list(contexts, queries)
        return [CompressBatchInput(context=ctx, query=q) for ctx, q in zip(contexts, query_list)]

    @staticmethod
    def _resolve_query_list(
        contexts: List[str],
        queries: Optional[Union[str, List[str]]],
    ) -> List[Optional[str]]:
        if queries is None:
            return [None] * len(contexts)
        if isinstance(queries, str):
            return [queries] * len(contexts)
        if len(queries) != len(contexts):
            raise ValidationError(
                f"Number of queries ({len(queries)}) must match contexts ({len(contexts)})"
            )
        return list(queries)

    @property
    def messages(self) -> Any:
        """Anthropic-shaped facade: ``client.messages.create(...)``."""
        self._require_engine("messages.create")
        if not hasattr(self, "_anthropic_facade"):
            from compresr.agents.facades.anthropic import _Anthropic

            self._anthropic_facade = _Anthropic(self._engine)
        return self._anthropic_facade.messages

    @property
    def chat(self) -> Any:
        """OpenAI-shaped facade: ``client.chat.completions.create(...)``."""
        self._require_engine("chat.completions.create")
        if not hasattr(self, "_openai_facade"):
            from compresr.agents.facades.openai import _OpenAI

            self._openai_facade = _OpenAI(self._engine)
        return self._openai_facade.chat

    def run(
        self,
        *,
        prompt: str,
        tools: Optional[list] = None,
        system: Optional[Any] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        """Native facade — returns :class:`NormalizedResult` directly.

        ``model`` overrides the constructor default; if neither is set the
        engine raises a clear :class:`CompresrError`.
        """
        self._require_engine("run")
        if not hasattr(self, "_native_facade"):
            from compresr.agents.facades.native import _Native

            self._native_facade = _Native(self._engine)
        return self._native_facade(
            prompt=prompt,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            model=model,
            **kw,
        )

    async def arun(
        self,
        *,
        prompt: str,
        tools: Optional[list] = None,
        system: Optional[Any] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        **kw: Any,
    ) -> Any:
        """Async native facade — returns :class:`NormalizedResult` directly."""
        self._require_engine("arun")
        if not hasattr(self, "_native_facade"):
            from compresr.agents.facades.native import _Native

            self._native_facade = _Native(self._engine)
        return await self._native_facade.arun(
            prompt=prompt,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
            model=model,
            **kw,
        )

    def _require_engine(self, surface: str) -> None:
        """Guard the facade surfaces with a clear opt-in error."""
        if self._engine is None:
            from compresr.exceptions import CompresrError

            raise CompresrError(
                f"CompressionClient.{surface} requires an LLM provider. "
                f"Construct with CompressionClient(api_key='cmp_...', "
                f"llm='anthropic', llm_api_key='sk-ant-...')."
            )

    async def aclose(self) -> None:
        """Close the underlying async HTTP client and release its pool."""
        await super().aclose()

    async def __aenter__(self) -> "CompressionClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
