from __future__ import annotations

import html as _html
import re
from typing import Iterable

import requests

_WIKI_UA = "compresr-sdk-tutorial/1.0 (mailto:support@compresr.ai)"


def fetch_wikipedia(title: str) -> str:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exlimit": "1",
            "titles": title,
            "format": "json",
            "redirects": "1",
        },
        headers={"User-Agent": _WIKI_UA},
        timeout=60,
    )
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "") or ""


def fetch_corpus(titles: Iterable[str], *, separator: str = "\n\n# ") -> str:
    chunks: list[str] = []
    for t in titles:
        text = fetch_wikipedia(t)
        if text:
            chunks.append(f"{t}\n\n{text}")
    return separator.join(chunks) if chunks else ""


def compresr_diff_html(raw: str, compressed: str, *, max_display_chars: int = 20_000):
    from collections import Counter

    from IPython.display import HTML

    def _key(w: str) -> str:
        return re.sub(r"[^\w]", "", w).lower()

    full_counter: Counter[str] = Counter(k for k in (_key(w) for w in compressed.split()) if k)
    full_raw_count = 0
    full_kept = 0
    for w in raw.split():
        k = _key(w)
        if not k:
            continue
        full_raw_count += 1
        if full_counter[k] > 0:
            full_counter[k] -= 1
            full_kept += 1
    overall_pct = full_kept * 100 // max(1, full_raw_count)

    truncated = len(raw) > max_display_chars
    display_raw = raw[:max_display_chars]
    display_counter: Counter[str] = Counter(k for k in (_key(w) for w in compressed.split()) if k)

    raw_lines = display_raw.split("\n")
    rendered_lines: list[str] = []
    for line in raw_lines:
        words = line.split()
        if not words:
            rendered_lines.append("&nbsp;")
            continue
        spans = []
        for w in words:
            k = _key(w)
            kept = False
            if k and display_counter[k] > 0:
                display_counter[k] -= 1
                kept = True
            if kept:
                spans.append(
                    f'<span style="background:#d1f4d1;color:#0a6c1d;'
                    f'padding:0 2px;border-radius:2px">{_html.escape(w)}</span>'
                )
            else:
                spans.append(
                    f'<span style="background:#ffd7d5;color:#a01b1b;'
                    f"text-decoration:line-through;padding:0 2px;"
                    f'border-radius:2px;opacity:0.85">{_html.escape(w)}</span>'
                )
        rendered_lines.append(" ".join(spans))

    if truncated:
        rendered_lines.append(
            f'<div style="color:#57606a;font-style:italic;margin-top:8px">'
            f"… visual truncated; full corpus has {full_raw_count:,} words "
            f"(showing first {max_display_chars:,} chars).</div>"
        )

    header = (
        f'<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;'
        f'font-size:13px;margin:0 0 6px 0;color:#1f2328">'
        f"Overall: <b>{full_kept:,}</b> / <b>{full_raw_count:,}</b> words kept "
        f"(<b>{overall_pct}%</b>) — query-aware compression keeps the "
        f"answer-relevant tokens, drops the rest.&nbsp;&nbsp; "
        f'<span style="background:#d1f4d1;color:#0a6c1d;padding:0 4px;'
        f'border-radius:2px">kept</span>&nbsp; '
        f'<span style="background:#ffd7d5;color:#a01b1b;'
        f'text-decoration:line-through;padding:0 4px;border-radius:2px">'
        f"dropped</span>"
        f"</div>"
    )
    body = (
        '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'
        "monospace;font-size:11.5px;line-height:1.65;max-height:540px;"
        "overflow:auto;padding:12px 14px;background:#fafbfc;"
        'border:1px solid #d0d7de;border-radius:6px;white-space:normal">'
        + "<br>".join(rendered_lines)
        + "</div>"
    )
    return HTML(header + body)


def print_savings_table(raw_tokens: int, cmp_tokens: int, *, calls_per_day: int = 1_000) -> None:
    days = 30
    prices = {"gpt-4o-mini": 0.15, "gpt-4o": 2.50, "claude-sonnet-4-6": 3.00, "gpt-5": 1.25}
    print(f"{'Model':<20}{'Raw $/mo':>14}{'Compresr $/mo':>17}{'Saved $/mo':>14}")
    for name, price in prices.items():
        raw_cost = raw_tokens * calls_per_day * days * price / 1_000_000
        cmp_cost = cmp_tokens * calls_per_day * days * price / 1_000_000
        print(f"{name:<20}{raw_cost:>14,.2f}{cmp_cost:>17,.2f}{raw_cost - cmp_cost:>14,.2f}")
