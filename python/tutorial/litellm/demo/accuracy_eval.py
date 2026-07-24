#!/usr/bin/env python3
"""
Accuracy eval: does Compresr's query-aware compression preserve (or improve)
downstream answer accuracy while cutting prompt tokens?

Setup (a realistic RAG / web_search "context dump" the model must read):
  - A knowledge base of many fictional companies, each a verbose paragraph with
    several metrics. Some companies have confusingly-similar names. The target
    fact (net revenue retention) is buried mid-paragraph.
  - For each question we ask gpt-5-mini twice:
       FULL       = the entire KB as context
       COMPRESSED = the same KB compressed by Compresr with the QUESTION as the
                    query (latte_v2) — exactly what the LiteLLM guardrail does to
                    a tool output, just with the question as the per-target query.
  - Score = the model's answer contains the correct retention figure.

Long, noisy context degrades retrieval ("lost in the middle"); query-aware
compression shortens it to the relevant slice. We report accuracy + tokens for
both conditions with REAL model calls.

Usage:  python accuracy_eval.py
Requires: OPENAI_API_KEY and COMPRESR_API_KEY (read from ../.env if present).
"""

import os
import random
import sys
from pathlib import Path


def _load_env() -> None:
    """Load the first ``.env`` found walking up from this script (demo dir → repo root)."""
    here = Path(__file__).resolve().parent
    for base in (here, *list(here.parents)[:5]):
        env_file = base / ".env"
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        return


SECTORS = [
    "logistics",
    "cybersecurity",
    "fintech",
    "biotech",
    "ad-tech",
    "data-infrastructure",
    "robotics",
    "e-commerce",
]
CITIES = [
    "Austin",
    "Denver",
    "Boston",
    "Seattle",
    "Atlanta",
    "Chicago",
    "Toronto",
    "Dublin",
    "Berlin",
    "Singapore",
]

# Confusingly-similar name pairs + unique names -> adversarial distractors.
NAMES = [
    "Aldridge Systems",
    "Aldridge Software",
    "Northwind Logistics",
    "Northgate Logistics",
    "Veridian Labs",
    "Veridian Analytics",
    "Brightpath AI",
    "Brightpath Robotics",
    "Summit Payments",
    "Summit Securities",
    "Cobalt Health",
    "Cobalt Cloud",
    "Pinewood Mobility",
    "Pinecrest Mobility",
    "Halcyon Data",
    "Halcyon Media",
    "Orion Freight",
    "Orion Foods",
    "Meridian Retail",
    "Meridian Rentals",
    "Sterling Devices",
    "Sterling Diagnostics",
    "Larkspur Energy",
    "Larkspur Education",
]


def build_kb(seed: int = 7):
    rng = random.Random(seed)
    companies = []
    used_nrr = set()
    for i, name in enumerate(NAMES):
        # distinct net revenue retention, clustered to look similar
        while True:
            nrr = rng.randint(104, 138)
            if nrr not in used_nrr:
                used_nrr.add(nrr)
                break
        rev = round(rng.uniform(0.4, 18.0), 1)
        g = rng.randint(4, 41)
        gm = rng.randint(58, 86)
        om = rng.randint(6, 34)
        gr = rng.randint(88, 97)
        hc = rng.randint(120, 9000)
        cash = round(rng.uniform(0.1, 9.0), 1)
        rd = rng.randint(9, 28)
        sector = SECTORS[i % len(SECTORS)]
        city = CITIES[i % len(CITIES)]
        para = (
            f"[{i+1}] {name} is a {sector} company headquartered in {city}. "
            f"In FY2025 it reported revenue of ${rev}B (up {g}% year over year), "
            f"a gross margin of {gm}% and an operating margin of {om}%. "
            f"Its net revenue retention was {nrr}%, while gross retention held at "
            f"{gr}%. Headcount stood at {hc:,} and it ended the year with "
            f"${cash}B in cash and equivalents. R&D spend was {rd}% of revenue, "
            f"and management reiterated its focus on durable growth and margin "
            f"discipline heading into FY2026."
        )
        companies.append({"name": name, "nrr": nrr, "para": para})
    return companies


def main() -> int:
    _load_env()
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("COMPRESR_API_KEY"):
        print("Need OPENAI_API_KEY and COMPRESR_API_KEY in env or ../.env")
        return 1

    from openai import OpenAI

    from compresr import CompressionClient

    oai = OpenAI()
    cmp = CompressionClient(api_key=os.environ["COMPRESR_API_KEY"], timeout=30)

    companies = build_kb()
    kb_text = "COMPANY BRIEFS (FY2025):\n\n" + "\n\n".join(c["para"] for c in companies)

    # Ask about targets spread across the middle of the context (hardest zone),
    # including confusingly-similar names.
    target_idxs = [2, 3, 4, 5, 8, 9, 12, 13, 16, 17]
    targets = [companies[i] for i in target_idxs]

    def ask(context: str, question: str) -> str:
        # gpt-5-mini is a reasoning model: it rejects `max_tokens` (use
        # `max_completion_tokens`) and only supports the default temperature,
        # and it needs headroom for hidden reasoning tokens before the answer.
        r = oai.chat.completions.create(
            model="gpt-5-mini",
            max_completion_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": "Answer using ONLY the provided company briefs. "
                    "Reply with just the net revenue retention percentage, "
                    "e.g. '123%'. If unknown, say 'unknown'.",
                },
                {"role": "user", "content": f"{context}\n\nQUESTION: {question}"},
            ],
        )
        return r.choices[0].message.content.strip(), r.usage.prompt_tokens

    full_correct = comp_correct = 0
    full_tok = comp_tok = comp_chars = full_chars = 0
    rows = []
    for t in targets:
        q = f"What was {t['name']}'s net revenue retention in FY2025?"
        expected = f"{t['nrr']}%"

        # Compress the KB query-aware (same call the guardrail makes on a tool output).
        cres = cmp.compress(
            context=kb_text,
            query=q,
            compression_model_name="latte_v2",
            target_compression_ratio=0.5,
        )
        compressed = cres.data.compressed_context

        full_ans, ftok = ask(kb_text, q)
        comp_ans, ctok = ask(compressed, q)

        f_ok = expected in full_ans
        c_ok = expected in comp_ans
        full_correct += f_ok
        comp_correct += c_ok
        full_tok += ftok
        comp_tok += ctok
        full_chars += len(kb_text)
        comp_chars += len(compressed)
        rows.append((t["name"], expected, full_ans, f_ok, comp_ans, c_ok))

    n = len(targets)
    print("=" * 92)
    print(f"{'company':22} {'truth':6} {'FULL ans':12} {'ok':3} {'COMPRESSED ans':14} {'ok':3}")
    print("-" * 92)
    for name, exp, fa, fok, ca, cok in rows:
        print(f"{name:22} {exp:6} {fa:12} {'Y' if fok else 'N':3} {ca:14} {'Y' if cok else 'N':3}")
    print("=" * 92)
    print(f"KB size: {len(kb_text)} chars, {len(companies)} companies")
    print(
        f"ACCURACY   full={full_correct}/{n} ({100*full_correct/n:.0f}%)   "
        f"compressed={comp_correct}/{n} ({100*comp_correct/n:.0f}%)"
    )
    print(
        f"AVG PROMPT TOKENS   full={full_tok/n:.0f}   compressed={comp_tok/n:.0f}   "
        f"({100*(1-comp_tok/full_tok):.0f}% fewer)"
    )
    print(
        f"AVG CONTEXT CHARS   full={full_chars/n:.0f}   compressed={comp_chars/n:.0f}   "
        f"({100*(1-comp_chars/full_chars):.0f}% smaller)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
