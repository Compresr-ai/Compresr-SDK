#!/usr/bin/env python3
"""
Run the Compresr guardrail hook in-process on a request JSON and print, in one
view: the REQUEST, the per-target compression query, the compresr_stats
METADATA, and the COMPRESSED prompt that would be forwarded upstream.

This is the same async_pre_call_hook the proxy runs — here we call it directly
so we can surface the metadata the proxy stashes in request state.

Usage:  python show_compression.py <request.json>
"""

import asyncio
import json
import os
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


def _roles(msgs):
    return [m.get("role") for m in msgs]


async def main() -> int:
    _load_env()
    req_path = sys.argv[1] if len(sys.argv) > 1 else "websearch_request.json"
    data = json.loads(Path(req_path).read_text())
    original = json.loads(json.dumps(data))  # deep copy for before/after

    from litellm import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    from compresr.integrations.litellm import CompresrGuardrail

    guardrail = CompresrGuardrail(
        guardrail_name="compresr-websearch-demo",
        api_key=os.environ["COMPRESR_API_KEY"],
        api_base=os.environ.get("COMPRESR_BASE_URL"),
        default_on=True,
        event_hook="pre_call",
    )

    print("=" * 74)
    print("REQUEST (what the app sends to the gateway)")
    print("=" * 74)
    print("roles:", _roles(original["messages"]))
    for i, m in enumerate(original["messages"]):
        c = m.get("content")
        size = len(c) if isinstance(c, str) else f"(non-text:{type(c).__name__})"
        tag = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            fn = m["tool_calls"][0]["function"]
            tag = f"  tool_call -> {fn['name']}({fn['arguments']})"
        print(f"  [{i}] {m.get('role'):9} chars={size}{tag}")

    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    stats = result.get("metadata", {}).get("compresr_stats")
    print("\n" + "=" * 74)
    print("METADATA (compresr_stats stashed in request metadata)")
    print("=" * 74)
    print(json.dumps(stats, indent=2))

    # Locate the tool message before/after.
    def tool_msg(msgs):
        return next(m for m in msgs if m.get("role") == "tool")

    o_tool = tool_msg(original["messages"])["content"]
    f_tool = tool_msg(result["messages"])["content"]

    print("\n" + "=" * 74)
    print("COMPRESSION (the tool/web_search output, rewritten in flight)")
    print("=" * 74)
    print(f"  original  : {len(o_tool):>5} chars")
    print(f"  forwarded : {len(f_tool):>5} chars")
    print(f"  reduction : {100 * (1 - len(f_tool) / len(o_tool)):.1f}%")
    print(
        f"\n  user question preserved verbatim : "
        f"{original['messages'][1]['content'] == result['messages'][1]['content']}"
    )
    print(
        f"  system prompt preserved verbatim : "
        f"{original['messages'][0]['content'] == result['messages'][0]['content']}"
    )

    print("\n" + "-" * 74)
    print("COMPRESSED web_search RESULTS forwarded upstream:")
    print("-" * 74)
    print(f_tool)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
