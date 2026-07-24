# Compresr SDK

[![PyPI](https://img.shields.io/pypi/v/compresr)](https://pypi.org/project/compresr/)
[![npm](https://img.shields.io/npm/v/compresr)](https://www.npmjs.com/package/compresr)
[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://pypi.org/project/compresr/)
[![TypeScript](https://img.shields.io/badge/TypeScript-%3E%3D5.0-blue)](https://www.npmjs.com/package/compresr)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Official SDKs for [Compresr](https://compresr.ai) — query-aware LLM context
compression. Reduce API costs by 30-70%.

## SDKs

| Language | Package | Documentation |
|----------|---------|---------------|
| Python | [![PyPI](https://img.shields.io/pypi/v/compresr)](https://pypi.org/project/compresr/) | [python/README.md](python/README.md) |
| TypeScript | [![npm](https://img.shields.io/npm/v/compresr)](https://www.npmjs.com/package/compresr) | [typescript/README.md](typescript/README.md) |
| curl / REST | — | [curl/README.md](curl/README.md) |

## Quick Start

### Python

```bash
pip install compresr
```

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

result = client.compress(
    context="Long passage to compress...",
    query="What is the main conclusion?",
    target_compression_ratio=0.5,
)

print(f"Saved {result.data.tokens_saved} tokens")
print(result.data.compressed_context)
```

### TypeScript

```bash
npm install @compresr/sdk
```

```typescript
import { CompressionClient } from '@compresr/sdk';

const client = new CompressionClient({ apiKey: 'cmp_your_api_key' });

const result = await client.compress({
  context: 'Long passage to compress...',
  query: 'What is the main conclusion?',
  targetCompressionRatio: 0.5,
});

console.log(`Saved ${result.data?.tokens_saved} tokens`);
```

### curl

```bash
curl -X POST https://api.compresr.ai/api/compress/question-specific/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Long passage to compress...",
    "query": "What is the main conclusion?",
    "compression_model_name": "latte_v2",
    "target_compression_ratio": 0.5
  }'
```

## Framework integrations

Both Python and TypeScript ship first-party integrations as optional
installs:

| Framework | Python import | TypeScript import |
|---|---|---|
| LangChain | `compresr.integrations.langchain` | `@compresr/sdk/integrations/langchain` |
| LangGraph | `compresr.integrations.langgraph` | `@compresr/sdk/integrations/langgraph` |
| LlamaIndex | `compresr.integrations.llamaindex` | `@compresr/sdk/integrations/llamaindex` |

Each integration exposes the same set of helpers: agent middleware, tool
wrappers, retriever/postprocessor adapters, and graph nodes. See the
language-specific READMEs for code snippets and the `tutorial/` directories
for runnable end-to-end examples.

## Getting an API key

1. Create an account at [compresr.ai](https://compresr.ai).
2. Navigate to Dashboard → API Keys.
3. Click "Create New Key" and copy it (shown only once).

## Features

- **Query-aware compression** (`latte_v2`) — keeps tokens relevant to the
  supplied query, drops the rest.
- **30-70% token reduction** on typical agent / RAG workloads.
- **Permissive model surface** — new compression models work without an SDK
  update; the backend is the authority.
- **Batch endpoint** — compress up to 100 contexts in one call.
- **Streaming + async** — full async/await + streaming for high-throughput
  pipelines.
- **First-party integrations** — LangChain, LangGraph, LlamaIndex (both
  Python and TypeScript).
- **Fail-open by default** — integrations log + passthrough on compression
  errors, opt-in to `raise`.

## Documentation

- [API Documentation](https://compresr.ai/docs)
- [Python SDK](python/README.md)
- [TypeScript SDK](typescript/README.md)
- [curl / REST Examples](curl/README.md)

## Run CI Locally

Test all workflows locally before pushing:

```bash
./test_ci_local.sh    # auto-installs `act` on first run
```

For tests requiring an API key, create `.secrets`:

```bash
echo "COMPRESR_API_KEY=your_key" > .secrets
```

Requires [GitHub CLI](https://cli.github.com/) and [act](https://github.com/nektos/act).

## Support

- Email: support@compresr.ai
- [API Documentation](https://compresr.ai/docs)
- [GitHub Discussions](https://github.com/Compresr-ai/Compresr-SDK/discussions)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
