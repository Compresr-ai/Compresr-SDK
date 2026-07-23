"""OpenAI-shaped facade — ``client.chat.completions.create(...)``.

Delegates the actual run to a private :class:`_Engine` and remaps the
returned :class:`NormalizedResult` to an OpenAI ``ChatCompletion`` shape.

Note: OpenAI puts the system prompt inside the ``messages`` list (with
``role="system"``). We pass ``messages`` through unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from ..schemas.openai import ChatCompletion, to_chat_completion


def _forward(kw: dict, **maybe: Any) -> dict:
    """Return ``kw`` with each non-``None`` ``maybe`` value added."""
    for key, value in maybe.items():
        if value is not None:
            kw[key] = value
    return kw


class _OpenAICompletions:
    """``client.chat.completions`` surface — exposes ``.create`` / ``.acreate``."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def create(
        self,
        *,
        model: Optional[str] = None,
        messages: list,
        max_tokens: int = 4096,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Any] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kw: Any,
    ) -> ChatCompletion:
        forwarded = _forward(
            dict(kw),
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            stop=stop,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        result = self._engine.run(
            messages=messages,
            tools=tuple(tools or ()),
            max_tokens=max_tokens,
            model=model,
            **forwarded,
        )
        effective_model = model or self._engine.default_model_name
        return to_chat_completion(result, model=effective_model)

    async def acreate(
        self,
        *,
        model: Optional[str] = None,
        messages: list,
        max_tokens: int = 4096,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Any] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kw: Any,
    ) -> ChatCompletion:
        forwarded = _forward(
            dict(kw),
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            stop=stop,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        result = await self._engine.arun(
            messages=messages,
            tools=tuple(tools or ()),
            max_tokens=max_tokens,
            model=model,
            **forwarded,
        )
        effective_model = model or self._engine.default_model_name
        return to_chat_completion(result, model=effective_model)


class _OpenAIChat:
    """``client.chat`` surface — exposes ``.completions``."""

    def __init__(self, engine: Any) -> None:
        self.completions = _OpenAICompletions(engine)


class _OpenAI:
    """``client`` surface — exposes ``.chat``."""

    def __init__(self, engine: Any) -> None:
        self.chat = _OpenAIChat(engine)


__all__ = ["_OpenAI", "_OpenAIChat", "_OpenAICompletions"]
