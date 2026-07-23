"""Native facade — ``client.run(prompt=..., tools=..., ...)``.

The native surface returns :class:`NormalizedResult` directly so callers
who don't need provider-specific shapes can skip the remappers entirely.
"""

from __future__ import annotations

from typing import Any, Optional, cast

from ..normalized import NormalizedResult


def _forward(kw: dict, **maybe: Any) -> dict:
    """Return ``kw`` with each non-``None`` ``maybe`` value added."""
    for key, value in maybe.items():
        if value is not None:
            kw[key] = value
    return kw


class _Native:
    """Callable native facade — invoke via ``__call__`` or ``arun``."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def __call__(
        self,
        *,
        prompt: str,
        tools: Optional[list] = None,
        system: Optional[Any] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        **kw: Any,
    ) -> NormalizedResult:
        forwarded = _forward(
            dict(kw),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )
        return cast(
            NormalizedResult,
            self._engine.run(
                messages=[{"role": "user", "content": prompt}],
                tools=tuple(tools or ()),
                system=system,
                max_tokens=max_tokens,
                model=model,
                **forwarded,
            ),
        )

    async def arun(
        self,
        *,
        prompt: str,
        tools: Optional[list] = None,
        system: Optional[Any] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        **kw: Any,
    ) -> NormalizedResult:
        forwarded = _forward(
            dict(kw),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )
        return cast(
            NormalizedResult,
            await self._engine.arun(
                messages=[{"role": "user", "content": prompt}],
                tools=tuple(tools or ()),
                system=system,
                max_tokens=max_tokens,
                model=model,
                **forwarded,
            ),
        )


__all__ = ["_Native"]
