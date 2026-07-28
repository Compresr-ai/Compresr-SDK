# Compresr for Hermes Agent

Query-aware context + tool-output compression for
[Hermes Agent](https://github.com/NousResearch/hermes-agent), powered by
[Compresr](https://compresr.ai).

Two independent, opt-in features:

| Feature | What it does | Turn on with |
|---|---|---|
| **Context engine** | Compacts mid-conversation turns via Compresr's query-specific API instead of an auxiliary-LLM summary | `context.engine: compresr` |
| **Tool-output compression** | Shrinks large tool outputs as they arrive; the verbatim original stays recoverable from Hermes's cache via `read_file` | `compresr.tool_output_enabled: true` |

Both fail open: any API error falls back to Hermes's built-in behavior and
never drops context. Host tool calls are never broken.

## Install

```bash
hermes plugins install Compresr-ai/Compresr-SDK/hermes-plugin
pip install compresr           # same environment that runs Hermes
hermes plugins enable compresr
```

The installer prompts for `COMPRESR_API_KEY` (get one at
[compresr.ai](https://compresr.ai) → Dashboard → API Keys) and saves it to
`~/.hermes/.env`. `compresr-sdk login` works too.

Alternatively, pip-only (no guided setup): `pip install compresr` already
ships the plugin via the `hermes_agent.plugins` entry point — just
`hermes plugins enable compresr` and set the key yourself.

## Configuration

Environment variables take precedence over the `compresr:` block in
`~/.hermes/config.yaml`.

```yaml
context:
  engine: compresr             # activate the context engine

compresr:
  tool_output_enabled: true    # activate per-turn tool-output compression
  # base_url: https://api.compresr.ai
  # model: latte_v2            # context-engine model (latte_v1 | latte_v2)
  # target_ratio: 5            # context-engine override: 0-1 fraction or Nx factor
  # timeout: 60                # context-engine API timeout (s)
  # tool_output_model: toc_latte_v2
  # tool_output_min_tokens: 1500
  # tool_output_target_ratio: 2.0
  # tool_output_timeout: 30
  # tool_output_max_cache_mb: 256
```

| Env var | Config key | Default |
|---|---|---|
| `COMPRESR_API_KEY` | — (secrets live in `.env`) | — |
| `COMPRESR_BASE_URL` | `base_url` | `https://api.compresr.ai` |
| `COMPRESR_MODEL` | `model` | `latte_v2` |
| `COMPRESR_TARGET_RATIO` | `target_ratio` | derived from `compression.target_ratio` |
| `COMPRESR_TIMEOUT` | `timeout` | `60` |
| `COMPRESR_TOOL_OUTPUT_ENABLED` | `tool_output_enabled` | `false` |
| `COMPRESR_TOOL_OUTPUT_MODEL` | `tool_output_model` | `toc_latte_v2` |
| `COMPRESR_TOOL_OUTPUT_MIN_TOKENS` | `tool_output_min_tokens` | `1500` |
| `COMPRESR_TOOL_OUTPUT_TARGET_RATIO` | `tool_output_target_ratio` | `2.0` |
| `COMPRESR_TOOL_OUTPUT_TIMEOUT` | `tool_output_timeout` | `30` |
| `COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB` | `tool_output_max_cache_mb` | `256` |

## Usage

Everything is automatic once enabled. In a session:

- `/compresr` — live stats (calls, tokens saved, errors, recoveries).
- Compressed tool outputs end with a `[compresr:recover]` footer pointing at
  the cached original — the agent recovers exact bytes with `read_file`.

## How it works

This directory is a thin shim: all logic ships in the `compresr` PyPI package
(`compresr.integrations.hermes`), built on the official SDK client (retries,
typed errors, connection pooling). The shim exists so
`hermes plugins install` can offer guided setup.

## Support

- support@compresr.ai · https://compresr.ai/docs
