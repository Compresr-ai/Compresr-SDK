# Compresr Python SDK

Query-aware LLM context compression — reduce LLM API costs by 30-70%.

## Install

```bash
pip install compresr
```

Get an API key at [compresr.ai](https://compresr.ai) → Dashboard → API Keys.

## Quick start

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

result = client.compress(
    context="Long passage to compress...",
    query="What is the main conclusion?",
    target_compression_ratio=0.5,
)

print(f"Original:   {result.data.original_tokens} tokens")
print(f"Compressed: {result.data.compressed_tokens} tokens")
print(f"Saved:      {result.data.tokens_saved} tokens")
print(result.data.compressed_context)
```

The default model is `latte_v2` (query-aware). Pass any other model name your
account has access to via `compression_model_name="..."` — the backend
validates.

## Adaptive (dynamic) compression — `latte_v2` only

`latte_v2` can pick the keep-ratio per document instead of holding a fixed
target. Useful for RAG / tool-output flows where chunk density varies a lot:
dense docs keep more, sparse docs compress hard.

```python
# Server adapts per-doc, capped between 1.5x and 10x by default
result = client.compress(
    context="...",
    query="...",
    compression_model_name="latte_v2",
    dynamic=True,
)

# Tighter band — never less than 2x, never more than 6x
result = client.compress(
    context="...",
    query="...",
    compression_model_name="latte_v2",
    dynamic=True,
    dynamic_min_ratio=2.0,
    dynamic_max_ratio=6.0,
)
```

When `dynamic=True`, `target_compression_ratio` is ignored. Sending
`dynamic=True` to a model that doesn't support it (e.g. `latte_v1`) returns
a 422 from the API.

## Batch

Compress up to 100 contexts in one call. Pass a single query (applied to all)
or a list of one query per context:

```python
batch = client.compress_batch(
    contexts=["Doc 1...", "Doc 2...", "Doc 3..."],
    queries="What is self-attention?",
    target_compression_ratio=0.5,
)

print(f"Total saved: {batch.data.total_tokens_saved} tokens")
```

## Async + streaming

```python
result = await client.compress_async(context="...", query="...")

for chunk in client.compress_stream(context="...", query="..."):
    print(chunk.content, end="")
```

## LLM-agnostic agent client

One `CompressionClient`, three provider-shape facades, one engine. Construct
the client with `llm=` and you get an agent surface where **every tool output
is compressed automatically** before the LLM sees it.

```python
import os
from compresr import CompressionClient, WebSearchTool

client = CompressionClient(
    api_key=os.environ["COMPRESR_API_KEY"],
    llm="anthropic",                        # or "openai", "google_genai"
    llm_api_key=os.environ["ANTHROPIC_API_KEY"],
    compression={"target_compression_ratio": 0.5, "min_tokens": 300},
)
```

Use `llm="anthropic:claude-haiku-4-5"` if you want a default — but the
call-site `model=` always wins.

Three equivalent surfaces sit on the same client — the model lives at
the call site, just like Anthropic's and OpenAI's own SDKs:

```python
# Anthropic shape
client.messages.create(model="claude-haiku-4-5", max_tokens=512,
                       messages=[...], tools=[...])

# OpenAI shape
client.chat.completions.create(model="gpt-5-mini", messages=[...], tools=[...])

# Native — returns a NormalizedResult
client.run(prompt="...", model="claude-haiku-4-5", tools=[...], max_tokens=512)
```

Behind all three sits LangChain 1.0's `create_agent` + `CompresrToolMiddleware`.
Tool outputs above `min_tokens` flow through `client.compress(...)` first.

### Built-in web search

```python
search = WebSearchTool.tavily(
    api_key=os.environ["TAVILY_API_KEY"],
    max_results=5,
    allowed_domains=["nytimes.com", "reuters.com"],   # optional
)
# Brave: WebSearchTool.brave(api_key=..., max_results=5)
```

#### Amazon Bedrock AgentCore

`WebSearchTool.agentcore(...)` reaches Amazon Bedrock AgentCore web search via a
Cognito OAuth handshake + an MCP streamable-HTTP session. Its runtime deps are
optional:

```bash
pip install compresr[agentcore]
```

Config resolves from explicit args first, then env vars (the
`AGENTCORE_`-prefixed name takes precedence, with the bare name as fallback):

| Field | Env var (fallback) |
|---|---|
| `gateway_url` | `AGENTCORE_GATEWAY_MCP_URL` (`GATEWAY_MCP_URL`) |
| `cognito_token_url` | `AGENTCORE_COGNITO_TOKEN_URL` (`COGNITO_TOKEN_URL`) |
| `client_id` | `AGENTCORE_COGNITO_CLIENT_ID` (`COGNITO_CLIENT_ID`) |
| `client_secret` | `AGENTCORE_COGNITO_CLIENT_SECRET` (`COGNITO_CLIENT_SECRET`) |
| `scope` | `AGENTCORE_COGNITO_SCOPE` (`COGNITO_SCOPE`) |

```python
# All five values from env vars:
search = WebSearchTool.agentcore(max_results=5)

# Or pass them explicitly:
search = WebSearchTool.agentcore(
    gateway_url="https://...gateway.../mcp",
    cognito_token_url="https://...oauth2/token",
    client_id="...",
    client_secret="...",          # never logged
    scope="gateway/invoke",
    max_results=5,                # clamped to 1–25
)
```

The tool emitted to the LLM is named `agentcore_web_search` and returns
plaintext. Domain filters (`allowed_domains` / `blocked_domains`) are not
supported by AgentCore — use Tavily for domain filtering.

### Bring your own tool

Any `@tool`-decorated function works — its string output is compressed for you:

```python
from langchain_core.tools import tool

@tool
def kb_lookup(topic: str) -> str:
    """Look up the internal policy on the given topic."""
    return INTERNAL_KB.get(topic, "Not found.")

client.messages.create(model="claude-haiku-4-5", max_tokens=256,
                       messages=[{"role": "user", "content": "Refund policy?"}],
                       tools=[kb_lookup])
```

Switch providers with one line: `llm="openai"` instead of
`llm="anthropic"` (then pass the model at the call site). Tools and
code are unchanged.

### Per-call LLM knobs

Pass `temperature`, `top_p`, `max_tokens`, `stop_sequences`,
`presence_penalty`, `frequency_penalty`, `seed`, etc. to any facade — they're
forwarded to the underlying chat model via `.bind(...)` per call, so the
cached chat model is never mutated:

```python
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    temperature=0.2,
    top_p=0.9,
    messages=[...],
)
```

Gemini's `max_output_tokens` is aliased automatically when targeting
`llm="google_genai:..."`.

**Why not provider-native server search?** Anthropic's `web_search_20250305`,
OpenAI's `web_search_preview`, and Gemini's `google_search` run server-side
and return encrypted/opaque content that Compresr cannot read or compress.
Use Tavily or Brave so the result is plaintext we can compress.

## Research agent

`client.research.run(question)` runs a multi-step web-research loop with
per-snippet `latte_v1` compression on tool results and multi-provider prompt
caching. Loop structure adapted from Perplexity `search_evals` (MIT).

```python
from compresr import CompressionClient

client = CompressionClient(
    api_key="cmp_...",
    llm="anthropic:claude-sonnet-4-6",
    llm_api_key="sk-ant-...",
)

result = client.research.run(
    "What was the latest stable Python version released in 2025?",
    search="tavily",          # "tavily" | "brave" | a LangChain BaseTool
    max_steps=10,
)

print(result.answer)              # parsed Exact Answer field
print(result.explanation)         # parsed Explanation field
print(result.confidence)          # parsed 0-1 confidence
print(result.citations)           # list[Citation(url=...)]
print(result.usage.cache_read_tokens, result.usage.calls)
```

Single-shot mode: `client.research.search(question)` runs one search + a
forced final answer (equivalent to `run(..., max_steps=2)`).

The agent respects all the prompt-cache options on `CompressionClient`
(`enable_prompt_cache`, `prompt_cache_ttl`, `openai_prompt_cache_key`).
Tavily / Brave keys are read from `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY`
(falls back to `BRAVE_API_KEY`).

## Compression options

| Param | Purpose |
|---|---|
| `query` | Question the LLM is trying to answer — drives `latte_v2` compression |
| `target_compression_ratio` | `0-1` strength (e.g. `0.5` = remove 50%) or `>1` for Nx factor (`4` = 4x). Backend max: 200 |
| `coarse` | `True` for paragraph-level (default, faster), `False` for token-level (fine-grained) |
| `heuristic_chunking` | Structure-preserving chunking |
| `disable_placeholders` | Disable placeholder tokens in output |

## Error handling

```python
from compresr.exceptions import (
    CompresrError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)

try:
    result = client.compress(context="...", query="...")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except ValidationError as e:
    print(f"Invalid request: {e}")
except CompresrError as e:
    print(f"API error: {e}")
```

## Framework integrations

The agents layer ships in the base install — `pip install compresr` is enough to get `CompressionClient`, all three provider chat models (Anthropic / OpenAI / Gemini), and both web search tools (Tavily + Brave).

Genuinely optional integrations beyond the agents layer:

| Extra | Pulls in |
|---|---|
| `compresr[langgraph]` | `langgraph` (LangGraph checkpoint serializer, store, handoff tool) |
| `compresr[llamaindex]` | `llama-index-core` (node postprocessor, memory block, tool wrapper) |
| `compresr[litellm]` | `litellm[proxy]` (LiteLLM proxy guardrail) |
| `compresr[agentcore]` | `mcp`, `nest-asyncio` (Amazon Bedrock AgentCore web search) |
| `compresr[all]` | all three above |

```bash
pip install "compresr[langgraph]"
```

Old `compresr[agents]` / `compresr[agents-anthropic]` / `compresr[agents-all]` / `compresr[langchain]` install commands still resolve (no-op extras kept for back-compat) — everything they used to pull in is now in the base install.

### LangChain — middleware + tool wrapper + retriever

```python
from langchain.agents import create_agent
from compresr.integrations.langchain import (
    CompresrToolMiddleware,
    wrap_tool_with_compression,
    CompresrExtractor,
)

agent = create_agent(
    model=model,
    tools=[web_search],
    middleware=[CompresrToolMiddleware(
        api_key=os.environ["COMPRESR_API_KEY"],
        query_arg="query",
    )],
)
```

### LangGraph — compression as a graph node

```python
from compresr.integrations.langgraph import make_compresr_node

graph.add_node("compress", make_compresr_node(
    api_key=os.environ["COMPRESR_API_KEY"],
    context_key="retrieved_text",
    query_key="user_question",
))
```

### LlamaIndex — node postprocessor for RAG

```python
from compresr.integrations.llamaindex import CompresrNodePostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[CompresrNodePostprocessor(
        api_key=os.environ["COMPRESR_API_KEY"],
    )],
)
```

### LiteLLM proxy — context compression as a guardrail

Run Compresr as a [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/guardrails)
`pre_call` guardrail — every request the proxy serves gets bulky tool/function
outputs rewritten with `latte_v2` compression keyed to the matching tool call's
intent, *before* the LLM sees them. No app code changes, no client wrapping.

Pick **one** of three install paths:

```bash
pip install 'compresr[litellm]'
export COMPRESR_API_KEY=cmp_...

# A — drop-in CLI wrapper (no site-packages writes)
compresr-litellm --config proxy_config.yaml --port 4000

# B — one-time shim install, then use plain `litellm`
install-compresr-shim
litellm --config proxy_config.yaml --port 4000

# C — Docker / ephemeral: install on first import
export COMPRESR_AUTO_INSTALL_SHIM=1
python -c "import compresr.integrations.litellm"
litellm --config proxy_config.yaml --port 4000
```

Minimal `proxy_config.yaml`:

```yaml
model_list:
  - model_name: gpt-5-mini
    litellm_params:
      model: openai/gpt-5-mini
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "compresr"
    litellm_params:
      guardrail: compresr
      mode: pre_call
      api_key: os.environ/COMPRESR_API_KEY
      default_on: true
      # All knobs are optional; defaults shown for reference
      compression_model_name: latte_v2     # latte_v1 also available
      target_compression_ratio: 0.5        # 0–1 = fraction removed, >1 = Nx factor
      coarse: true                         # paragraph-level (fast); false = token-level
      min_chars_to_compress: 500
      compress_tool_outputs: true          # ON by default
      compress_system: false               # opt-in
      compress_history: false              # opt-in
      compress_last_user: false            # opt-in
      target_ratio_by_role: {tool: 0.7, system: 0.3}
      fail_closed: false                   # fail-open by default
      cache_ttl: 300
```

Per-request override (no restart, no YAML edit):

```json
{
  "model": "gpt-5-mini",
  "messages": [...],
  "metadata": {
    "guardrail_config": {
      "compression_model_name": "latte_v1",
      "target_compression_ratio": 4,
      "coarse": false,
      "compress_history": true
    }
  }
}
```

**Defaults live in one place** — `compresr.integrations.litellm.DEFAULTS`
(frozen dataclass). Edit `compresr/integrations/litellm/defaults.py` to
change a default; it propagates to the runtime constructor and the YAML
schema together.

The guardrail fails open by default: if the Compresr API is unreachable,
the request is forwarded uncompressed and tagged `compresr:fail_open` in
the `x-litellm-applied-guardrails` response header. Set `fail_closed: true`
to harden. See `python/litellm_shim/README.md` for the on-prem deployment
matrix and admin-UI notes.

### Unified query API

Every integration that accepts a query exposes the same three knobs:

| Param | Purpose |
|---|---|
| `query` | Static query — same for every call |
| `query_extractor` | Callable that derives the query from the call context |
| `query_arg` / `query_key` | Name of the tool arg / state key to use as the query |

Priority: `query` > `query_extractor` > `query_arg`/`query_key` > smart-pick
from common arg keys (`query`, `question`, `search_query`, ...) > last human
message in history.

### Tutorials

Runnable Jupyter notebooks under `tutorial/`:

- `01_quickstart.ipynb` — core `CompressionClient`.
- `02_langchain.ipynb` — middleware + tool wrapper + retriever.
- `03_langgraph.ipynb` — compression node in a 3-node graph.
- `04_llamaindex.ipynb` — node postprocessor + tool wrapper.
- `05_compresr_agents.ipynb` — agent client (Anthropic/OpenAI/native shapes) with auto-compressed tool output.

## Requirements

- Python 3.9+
- `httpx >= 0.27.0`
- `pydantic >= 2.10.0`
- Optional: `langchain>=1.0`, `langgraph>=0.2`, `llama-index-core>=0.11`
  (install the matching extra)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Support

- Docs: [compresr.ai/docs](https://compresr.ai/docs/overview)
- Issues: [GitHub](https://github.com/Compresr-ai/Compresr-SDK/issues)
- Email: support@compresr.ai
