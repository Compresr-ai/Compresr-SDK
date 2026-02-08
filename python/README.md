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

```python
from compresr import CompressionClient

# Initialize with your API key
client = CompressionClient(api_key="cmp_your_api_key_here")
```

### Basic Compression

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

# Compress context
result = client.compress(
    context="Your very long context that needs compression...",
    compression_model_name="cmprsr_v1"
)

print(f"Original: {result.data.original_tokens} tokens")
print(f"Compressed: {result.data.compressed_tokens} tokens")
print(f"Compression ratio: {result.data.actual_compression_ratio:.2%}")
print(f"Tokens saved: {result.data.tokens_saved}")

# Use compressed context with your own LLM
compressed_context = result.data.compressed_context
```

### Integration with OpenAI

```python
from compresr import CompressionClient
from openai import OpenAI

compresr = CompressionClient(api_key="cmp_xxx")
openai_client = OpenAI(api_key="sk-xxx")

# Compress your long context
compressed = compresr.compress(
    context="Your long system prompt or document...",
    compression_model_name="cmprsr_v1"
)

# Use with OpenAI - saves tokens and money!
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": compressed.data.compressed_context},
        {"role": "user", "content": "Analyze this data..."}
    ]
)

print(f"Saved {compressed.data.tokens_saved} tokens!")
```

## Streaming Support

Real-time streaming compression:

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

for chunk in client.compress_stream(
    context="Your long context...",
    compression_model_name="cmprsr_v1"
):
    print(chunk.content, end="", flush=True)
```

## Async Support

All methods have async variants:

```python
import asyncio
from compresr import CompressionClient

async def main():
    client = CompressionClient(api_key="cmp_your_api_key")

    result = await client.compress_async(
        context="Your context...",
        compression_model_name="cmprsr_v1"
    )

    print(result.data.compressed_context)
    await client.close()

asyncio.run(main())
```

## Batch Processing

Process multiple contexts efficiently:

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")
results = client.compress_batch(
    contexts=[
        "First context to compress...",
        "Second context to compress..."
    ],
    compression_model_name="cmprsr_v1"
)

print(f"Total tokens saved: {results.data.total_tokens_saved}")
for result in results.data.results:
    print(f"  - Original: {result.original_tokens} → Compressed: {result.compressed_tokens}")
```

## API Reference

### Client Initialization

```python
from compresr import CompressionClient

client = CompressionClient(
    api_key="cmp_your_api_key",           # Required
    base_url="https://api.compresr.ai",   # Optional (default)
    timeout=30,                            # Optional: seconds
)
```

### Methods

| Method | Description |
|--------|-------------|
| `compress()` | Compress single context |
| `compress_async()` | Async compress |
| `compress_batch()` | Batch compress multiple contexts |
| `compress_stream()` | Stream compression |

### Response Structure

```python
# CompressionResult
result.data.original_context      # Original input
result.data.compressed_context    # Compressed output
result.data.original_tokens       # Token count before
result.data.compressed_tokens     # Token count after
result.data.actual_compression_ratio  # Achieved ratio (0-1)
result.data.tokens_saved          # Tokens saved
result.data.duration_ms           # Processing time

# BatchResult
results.data.total_original_tokens
results.data.total_compressed_tokens
results.data.total_tokens_saved
results.data.average_compression_ratio
results.data.count
results.data.results              # List of CompressionResult
```

## Key Features

- **🗜️ Intelligent Compression**: Reduce context lengths while preserving meaning
- **💰 Cost Optimization**: Save 30-70% on LLM API costs
- **⚡ Streaming Support**: Real-time response streaming
- **🔄 Async Ready**: Full async/await support
- **📦 Batch Processing**: Handle multiple contexts efficiently
- **📊 Usage Tracking**: Monitor compression metrics

## Available Models

| Model | Type | Description |
|-------|------|-------------|
| `cmprsr_v1` | Agnostic | Production compressor (default) - uses Qwen3-4B |
| `question_specific_cmprsr` | Question-Specific | Preserves question-relevant info (UCC format) |
| `question_specific_reranker` | Question-Specific | Selects chunks based on question using reranker |
| `history_cmprsr` | Agentic | Compresses conversation history into summaries |
| `tool_output_cmprsr` | Agentic | Compresses tool outputs with specialized prompts |
| `tool_output_openai` | Agentic | Compresses tool outputs using OpenAI GPT-5-mini |
| `tool_output_reranker` | Agentic | Selects tool output chunks using reranker |
| `tool_discovery_cmprsr` | Agentic | LLM-based tool discovery for selecting relevant tools |

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
    result = client.compress(
        context="Your context...",
        compression_model_name="cmprsr_v1"
    )
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

- 📖 Documentation: [docs.compresr.ai](https://docs.compresr.ai)
- 💬 Support: [hello@compresr.ai](mailto:hello@compresr.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/compresr/sdk/issues)
- 🌐 Website: [compresr.ai](https://compresr.ai)