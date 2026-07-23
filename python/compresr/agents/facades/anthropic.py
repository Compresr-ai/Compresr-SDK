"""Anthropic-shaped facade — ``client.messages.create(...)``.

Delegates the actual run to a private :class:`_Engine` and remaps the
returned :class:`NormalizedResult` to an Anthropic ``Message`` shape.
"""

from __future__ import annotations

from typing import Any, Optional

from ..schemas.anthropic import AnthropicMessage, to_anthropic_message


def _forward(kw: dict, **maybe: Any) -> dict:
    """Return ``kw`` with each non-``None`` ``maybe`` value added.

    Used to keep explicit forwarding kwargs out of the engine call when
    the caller didn't set them, so engine-side defaults still apply.
    """
    for key, value in maybe.items():
        if value is not None:
            kw[key] = value
    return kw


class _AnthropicMessages:
    """``client.messages`` surface — exposes ``.create`` / ``.acreate``."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def create(
        self,
        *,
        model: Optional[str] = None,
        messages: list,
        max_tokens: int = 4096,
        tools: Optional[list] = None,
        system: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        **kw: Any,
    ) -> AnthropicMessage:
        forwarded = _forward(
            dict(kw),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )
        result = self._engine.run(
            messages=messages,
            tools=tuple(tools or ()),
            system=system,
            max_tokens=max_tokens,
            model=model,
            **forwarded,
        )
        # The engine raised if no effective model could be resolved, so
        # ``model or default`` is safe here for the response envelope.
        effective_model = model or self._engine.default_model_name
        return to_anthropic_message(result, model=effective_model)

    async def acreate(
        self,
        *,
        model: Optional[str] = None,
        messages: list,
        max_tokens: int = 4096,
        tools: Optional[list] = None,
        system: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        **kw: Any,
    ) -> AnthropicMessage:
        forwarded = _forward(
            dict(kw),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )
        result = await self._engine.arun(
            messages=messages,
            tools=tuple(tools or ()),
            system=system,
            max_tokens=max_tokens,
            model=model,
            **forwarded,
        )
        effective_model = model or self._engine.default_model_name
        return to_anthropic_message(result, model=effective_model)


class _Anthropic:
    """``client`` surface — exposes ``.messages``."""

    def __init__(self, engine: Any) -> None:
        self.messages = _AnthropicMessages(engine)


__all__ = ["_Anthropic", "_AnthropicMessages"]
