# Compresr Python SDK

Intelligent context compression service to optimize LLM costs and performance. Reduce your LLM API costs by 30-70% through intelligent context compression.

## Installation

```bash
pip install compresr
```

## Quick Start

### API Key Setup

Get your API key from [compresr.ai](https://compresr.ai):

1. Create an account at [compresr.ai](https://compresr.ai)
2. Navigate to Dashboard → API Keys
3. Click "Create New Key" and copy it (shown only once!)

### Two Client Types

#### 1. CompressionClient — Token-Level Compression

Intelligently removes less important tokens while preserving meaning. Best for compressing raw, unstructured text like system prompts, documents, or retrieved passages before sending to an LLM. Supports a tunable `compression_ratio` to control how aggressively to compress.

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

# Agnostic compression (espresso_v1, default) — no query needed
result = client.compress(
    context="Your very long context that needs compression...",
)

print(f"Original: {result.data.original_tokens} tokens")
print(f"Compressed: {result.data.compressed_tokens} tokens")
print(f"Saved: {result.data.tokens_saved} tokens")

# Query-specific compression (latte_v1) — query REQUIRED
result = client.compress(
    context="Python was created in 1991. JavaScript in 1995. Java in 1995.",
    query="Who created Python?",
    compression_model_name="latte_v1",
)

print(f"Compressed (query-relevant): {result.data.compressed_context}")
```

#### 2. FilterClient — Chunk-Level Filtering

Keeps or drops entire chunks based on query relevance — no partial rewriting. Best for RAG pipelines where your retriever returns many chunks and you want to discard the irrelevant ones before stuffing them into a prompt. Fast and binary: a chunk is either in or out.

```python
from compresr import FilterClient

client = FilterClient(api_key="cmp_your_api_key")

# Query is always required
result = client.filter(
    chunks=[
        "Python is a programming language created by Guido van Rossum.",
        "JavaScript was created by Brendan Eich for web browsers.",
        "Java was developed by James Gosling at Sun Microsystems.",
    ],
    query="Who created Python?",
)

print(f"Filtered chunks: {result.data.compressed_context}")
print(f"Saved: {result.data.tokens_saved} tokens")
```

### Integration with OpenAI

**Agnostic compression:**

```python
from compresr import CompressionClient
from openai import OpenAI

compresr = CompressionClient(api_key="cmp_xxx")
openai_client = OpenAI(api_key="sk-xxx")

compressed = compresr.compress(
    context="Your long system prompt or document...",
)

response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": compressed.data.compressed_context},
        {"role": "user", "content": "Analyze this data..."}
    ]
)

print(f"Saved {compressed.data.tokens_saved} tokens!")
```

**Query-specific compression (RAG/QA):**

```python
from compresr import CompressionClient
from openai import OpenAI

compresr = CompressionClient(api_key="cmp_xxx")
openai_client = OpenAI(api_key="sk-xxx")

user_question = "What is machine learning?"

compressed = compresr.compress(
    context="Retrieved documents with lots of information...",
    query=user_question,
    compression_model_name="latte_v1",
)

response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": compressed.data.compressed_context},
        {"role": "user", "content": user_question}
    ]
)
```

**Chunk filtering for RAG pipelines:**

```python
from compresr import FilterClient
from openai import OpenAI

filter_client = FilterClient(api_key="cmp_xxx")
openai_client = OpenAI(api_key="sk-xxx")

user_question = "What is Python?"
retrieved_chunks = ["Chunk about Python...", "Chunk about Java...", "Chunk about ML..."]

filtered = filter_client.filter(
    chunks=retrieved_chunks,
    query=user_question,
)

response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "\n".join(filtered.data.compressed_context)},
        {"role": "user", "content": user_question}
    ]
)
```

## Streaming Support

Both clients support real-time streaming:

```python
from compresr import CompressionClient, FilterClient

# Compression streaming
client = CompressionClient(api_key="cmp_your_api_key")
for chunk in client.compress_stream(
    context="Your long context...",
):
    print(chunk.content, end="", flush=True)

# Query-specific compression streaming
for chunk in client.compress_stream(
    context="Your long context...",
    query="What is important here?",
    compression_model_name="latte_v1",
):
    print(chunk.content, end="", flush=True)

# Filter streaming
filter_client = FilterClient(api_key="cmp_your_api_key")
for chunk in filter_client.filter_stream(
    chunks=["Chunk 1...", "Chunk 2..."],
    query="What is relevant?",
):
    print(chunk.content, end="", flush=True)
```

## Async Support

Both clients support async/await:

```python
import asyncio
from compresr import CompressionClient, FilterClient

async def main():
    # Compression async
    client = CompressionClient(api_key="cmp_your_api_key")
    result = await client.compress_async(
        context="Your context...",
    )

    # Query-specific compression async
    result = await client.compress_async(
        context="Your context...",
        query="What matters here?",
        compression_model_name="latte_v1",
    )

    # Filter async
    filter_client = FilterClient(api_key="cmp_your_api_key")
    filtered = await filter_client.filter_async(
        chunks=["Chunk 1...", "Chunk 2..."],
        query="What is relevant?",
    )

    await client.close()
    await filter_client.close()

asyncio.run(main())
```

## API Reference

### Client Initialization

```python
from compresr import CompressionClient, FilterClient

# Token-level compression
client = CompressionClient(
    api_key="cmp_your_api_key",  # Required
    timeout=30                    # Optional: request timeout in seconds
)

# Chunk-level filtering
filter_client = FilterClient(
    api_key="cmp_your_api_key",  # Required
    timeout=30                    # Optional: request timeout in seconds
)
```

### CompressionClient Methods

| Method | Description |
|--------|-------------|
| `compress()` | Compress context (latte_v1 requires `query` param) |
| `compress_async()` | Async compress |
| `compress_stream()` | Stream compression |

### FilterClient Methods

| Method | Description |
|--------|-------------|
| `filter()` | Filter chunks by query relevance (query REQUIRED) |
| `filter_async()` | Async filter |
| `filter_stream()` | Stream filtered chunks |

### Response Structure

```python
# CompressResult (same for both clients)
result.data.original_context      # Original input
result.data.compressed_context    # Compressed/filtered output
result.data.original_tokens       # Token count before
result.data.compressed_tokens     # Token count after
result.data.actual_compression_ratio  # Achieved ratio (0-1)
result.data.tokens_saved          # Tokens saved
result.data.duration_ms           # Processing time
```

## Available Models

### Compression Models (CompressionClient)

| Model | Query | Compression Ratio | Best For |
|-------|-------|-------------------|----------|
| `espresso_v1` | Not needed | Supported | System prompts, documents, general context — when there is no specific question |
| `latte_v1` | **Required** | Supported | RAG / QA — compress retrieved text while preserving answer-relevant tokens |

### Filter Models (FilterClient)

| Model | Query | Compression Ratio | Best For |
|-------|-------|-------------------|----------|
| `coldbrew_v1` | **Required** | Not supported | RAG chunk selection — quickly drop irrelevant chunks from a retrieval set |

## Error Handling

```python
from compresr import CompressionClient, FilterClient
from compresr.exceptions import (
    CompresrError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)

client = CompressionClient(api_key="cmp_your_api_key")

try:
    result = client.compress(context="Your context...")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except ValidationError as e:
    print(f"Invalid request: {e}")
except CompresrError as e:
    print(f"API error: {e}")
```

## Requirements

- Python 3.9+
- `httpx >= 0.27.0`
- `pydantic >= 2.10.0`

## License

Proprietary License

## Support

- Documentation: [compresr.ai/docs/overview](https://compresr.ai/docs/overview)
- Support: [support@compresr.ai](mailto:support@compresr.ai)
- Issues: [GitHub Issues](https://github.com/compresr/sdk/issues)
- Website: [compresr.ai](https://compresr.ai)
