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

### CompressionClient — Token-Level Compression

Intelligently removes less important tokens while preserving meaning. Best for compressing raw, unstructured text like system prompts, documents, or retrieved passages before sending to an LLM.

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

# Agnostic compression (espresso_v1) — no query needed
result = client.compress(
    context="Your very long context that needs compression...",
    compression_model_name="espresso_v1",
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

### Compression Models

| Model | Query | Description |
|-------|-------|-------------|
| `espresso_v1` | Not needed | Agnostic compression — good for system prompts, documents |
| `latte_v1` | **Required** | Query-specific — preserves tokens relevant to your question |

### Compression Options

#### Target Compression Ratio

Control how aggressively to compress:

```python
# Keep 50% of tokens
result = client.compress(
    context="Your context...",
    compression_model_name="espresso_v1",
    target_compression_ratio=0.5,
)

# 4x compression (keep 25%)
result = client.compress(
    context="Your context...",
    query="What matters?",
    compression_model_name="latte_v1",
    target_compression_ratio=4,
)
```

| Value | Meaning |
|-------|---------|
| `0.5` | Keep 50% of tokens |
| `0.3` | Keep 30% of tokens (aggressive) |
| `2` | 2x compression = keep 50% |
| `4` | 4x compression = keep 25% |

#### Coarse Mode (Paragraph-Level)

For faster compression, use `coarse=True` to operate at paragraph level instead of token level:

```python
# Token-level (default) — more precise
result = client.compress(
    context="Your context...",
    query="What is ML?",
    compression_model_name="latte_v1",
    coarse=False,
)

# Paragraph-level — faster, keeps/removes entire paragraphs
result = client.compress(
    context="Your context...",
    query="What is ML?",
    compression_model_name="latte_v1",
    coarse=True,
)
```

## Batch Compression

Compress multiple contexts efficiently in a single API call:

```python
# Same query for all contexts
response = client.compress_batch(
    contexts=[
        "Document about machine learning...",
        "Document about deep learning...",
        "Document about NLP...",
    ],
    queries="What are the key concepts?",
    compression_model_name="latte_v1",
)

print(f"Compressed {response.data.count} documents")
print(f"Total tokens saved: {response.data.total_tokens_saved}")

# Different query per context
response = client.compress_batch(
    contexts=["ML doc...", "DL doc...", "NLP doc..."],
    queries=["What is ML?", "What is DL?", "What is NLP?"],
    compression_model_name="latte_v1",
)

for i, result in enumerate(response.data.results):
    print(f"Doc {i+1}: {result.original_tokens} → {result.compressed_tokens} tokens")
```

## Integration with OpenAI

**Agnostic compression:**

```python
from compresr import CompressionClient
from openai import OpenAI

compresr = CompressionClient(api_key="cmp_xxx")
openai_client = OpenAI(api_key="sk-xxx")

compressed = compresr.compress(
    context="Your long system prompt or document...",
    compression_model_name="espresso_v1",
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

## Streaming Support

Real-time streaming compression:

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

# Streaming compression
for chunk in client.compress_stream(
    context="Your long context...",
    compression_model_name="espresso_v1",
):
    print(chunk.content, end="", flush=True)

# Query-specific streaming
for chunk in client.compress_stream(
    context="Your long context...",
    query="What is important here?",
    compression_model_name="latte_v1",
):
    print(chunk.content, end="", flush=True)
```

## Async Support

Full async/await support:

```python
import asyncio
from compresr import CompressionClient

async def main():
    client = CompressionClient(api_key="cmp_your_api_key")

    # Async compression
    result = await client.compress_async(
        context="Your context...",
        compression_model_name="espresso_v1",
    )

    # Async query-specific compression
    result = await client.compress_async(
        context="Your context...",
        query="What matters here?",
        compression_model_name="latte_v1",
    )

    # Async batch compression
    batch_result = await client.compress_batch_async(
        contexts=["Doc 1...", "Doc 2...", "Doc 3..."],
        queries="Summarize key points",
        compression_model_name="latte_v1",
    )

    await client.close()

asyncio.run(main())
```

## API Reference

### Client Initialization

```python
from compresr import CompressionClient

client = CompressionClient(
    api_key="cmp_your_api_key",  # Required
    timeout=30                    # Optional: request timeout in seconds
)
```

### Methods

| Method | Description |
|--------|-------------|
| `compress()` | Compress context (sync) |
| `compress_async()` | Compress context (async) |
| `compress_stream()` | Stream compression chunks |
| `compress_batch()` | Batch compress multiple contexts (sync) |
| `compress_batch_async()` | Batch compress multiple contexts (async) |

### Response Structure

```python
# Single compression result
result.data.original_context      # Original input
result.data.compressed_context    # Compressed output
result.data.original_tokens       # Token count before
result.data.compressed_tokens     # Token count after
result.data.actual_compression_ratio  # Achieved ratio
result.data.tokens_saved          # Tokens saved
result.data.duration_ms           # Processing time

# Batch compression result
batch.data.results                # List of CompressBatchItemResult
batch.data.count                  # Number of items
batch.data.total_original_tokens  # Total tokens before
batch.data.total_compressed_tokens # Total tokens after
batch.data.total_tokens_saved     # Total tokens saved
batch.data.average_compression_ratio # Average ratio
```

## Error Handling

```python
from compresr import CompressionClient
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
