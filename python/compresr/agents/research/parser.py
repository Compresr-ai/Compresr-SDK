"""Parse the strict-format final response into structured fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class _Parsed:
    answer: str
    explanation: str
    confidence: Optional[float]
    citation_urls: list[str]


_FIELD_PATTERNS = {
    "explanation": r"(?im)^\s*explanation\s*:\s*(.+?)(?=^\s*(?:exact\s+answer|confidence|citations)\s*:|\Z)",
    "answer": r"(?im)^\s*exact\s+answer\s*:\s*(.+?)(?=^\s*(?:explanation|confidence|citations)\s*:|\Z)",
    "confidence": r"(?im)^\s*confidence\s*:\s*(.+?)(?=^\s*(?:explanation|exact\s+answer|citations)\s*:|\Z)",
    "citations": r"(?im)^\s*citations\s*:\s*(.+?)\Z",
}


def parse_research_output(text: str) -> _Parsed:
    """Extract Explanation / Exact Answer / Confidence / Citations from ``text``.

    Missing fields default to empty / None. Confidence accepts ``"62"``,
    ``"62%"``, or ``"0.62"`` and normalizes to a 0–1 float.
    """
    if not text:
        return _Parsed(answer="", explanation="", confidence=None, citation_urls=[])

    def _grab(name: str) -> str:
        m = re.search(_FIELD_PATTERNS[name], text, re.DOTALL)
        return m.group(1).strip() if m else ""

    answer = _grab("answer")
    explanation = _grab("explanation")
    confidence_raw = _grab("confidence")
    citations_raw = _grab("citations")

    confidence: Optional[float] = None
    if confidence_raw:
        m = re.search(r"(\d+(?:\.\d+)?)", confidence_raw)
        if m:
            v = float(m.group(1))
            confidence = v / 100 if v > 1 else v

    citation_urls: list[str] = []
    if citations_raw:
        for url in re.findall(r"https?://[^\s,;<>\"']+", citations_raw):
            if url not in citation_urls:
                citation_urls.append(url)

    return _Parsed(
        answer=answer,
        explanation=explanation,
        confidence=confidence,
        citation_urls=citation_urls,
    )
