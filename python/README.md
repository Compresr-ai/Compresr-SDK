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

### Two Types of Compression

#### 1. Agnostic Compression (No Question Needed)

Use `CompressionClient` for general-purpose compression:

```python
from compresr import CompressionClient

client = CompressionClient(api_key="cmp_your_api_key")

result = client.compress(
    context="Your very long context that needs compression...",
    compression_model_name="A_CMPRSR_V1"  # or "A_CMPRSR_V1_FLASH" for speed
)

print(f"Original: {result.data.original_tokens} tokens")
print(f"Compressed: {result.data.compressed_tokens} tokens")
print(f"Saved: {result.data.tokens_saved} tokens")
```

#### 2. Question-Specific Compression

Use `QSCompressionClient` to compress based on a specific question:

```python
from compresr import QSCompressionClient

client = QSCompressionClient(api_key="cmp_your_api_key")

result = client.compress(
    context="Python was created in 1991. JavaScript in 1995. Java in 1995.",
    question="Who created Python?",
    compression_model_name="QS_CMPRSR_V1"
)

print(f"Compressed (question-relevant): {result.data.compressed_context}")
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
    compression_model_name="A_CMPRSR_V1"
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

**Question-specific compression (RAG/QA):**

```python
from compresr import QSCompressionClient
from openai import OpenAI

compresr = QSCompressionClient(api_key="cmp_xxx")
openai_client = OpenAI(api_key="sk-xxx")

user_question = "What is machine learning?"

# Compress retrieval results based on the question
compressed = compresr.compress(
    context="Retrieved documents with lots of information...",
    question=user_question,
    compression_model_name="QS_CMPRSR_V1"
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

Both clients support real-time streaming:

```python
from compresr import CompressionClient, QSCompressionClient

# Agnostic streaming
client = CompressionClient(api_key="cmp_your_api_key")
for chunk in client.compress_stream(
    context="Your long context...",
    compression_model_name="A_CMPRSR_V1"
):
    print(chunk.content, end="", flush=True)

# Question-specific streaming
qs_client = QSCompressionClient(api_key="cmp_your_api_key")
for chunk in qs_client.compress_stream(
    context="Your long context...",
    question="What is important here?",
    compression_model_name="QS_CMPRSR_V1"
):
    print(chunk.content, end="", flush=True)
```

## Async Support

Both clients support async/await:

```python
import asyncio
from compresr import CompressionClient, QSCompressionClient

async def main():
    # Agnostic async
    client = CompressionClient(api_key="cmp_your_api_key")
    result = await client.compress_async(
        context="Your context...",
        compression_model_name="A_CMPRSR_V1"
    )
    
    # Question-specific async
    qs_client = QSCompressionClient(api_key="cmp_your_api_key")
    qs_result = await qs_client.compress_async(
        context="Your context...",
        question="What matters here?",
        compression_model_name="QS_CMPRSR_V1"
    )
    
    await client.close()
    await qs_client.close()

asyncio.run(main())
```

## Batch Processing

Both clients support batch processing:

```python
from compresr import CompressionClient, QSCompressionClient

# Agnostic batch
client = CompressionClient(api_key="cmp_your_api_key")
results = client.compress_batch(
    contexts=["First context...", "Second context..."],
    compression_model_name="A_CMPRSR_V1"
)

# Question-specific batch
qs_client = QSCompressionClient(api_key="cmp_your_api_key")
qs_results = qs_client.compress_batch(
    contexts=["Context 1...", "Context 2..."],
    questions=["Question 1?", "Question 2?"],
    compression_model_name="QS_CMPRSR_V1"
)

print(f"Total tokens saved: {results.data.total_tokens_saved}")
```

## API Reference

### Client Initialization

```python
from compresr import CompressionClient, QSCompressionClient

# Agnostic compression
client = CompressionClient(
    api_key="cmp_your_api_key",  # Required
    timeout=30                    # Optional: request timeout in seconds
)

# Question-specific compression
qs_client = QSCompressionClient(
    api_key="cmp_your_api_key",  # Required
    timeout=30                    # Optional: request timeout in seconds
)
```

**Note:** BASE_URL is fixed to `https://api.compresr.ai` and cannot be changed.

### Methods

Both `CompressionClient` and `QSCompressionClient` support:

| Method | Description |
|--------|-------------|
| `compress()` | Compress single context (QS requires `question` param) |
| `compress_async()` | Async compress |
| `compress_batch()` | Batch compress (QS requires `questions` list) |
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

## Available Models

### Agnostic Models (CompressionClient)

| Model | Description | Use Case |
|-------|-------------|----------|
| `A_CMPRSR_V1` | LLM-based abstractive compression (default) | General purpose, best quality |
| `A_CMPRSR_V1_FLASH` | Fast extractive compression | Speed-critical applications |

### Question-Specific Models (QSCompressionClient)

| Model | Description | Use Case |
|-------|-------------|----------|
| `QS_CMPRSR_V1` | Question-specific compression, Abstractive (default) | General purpose |
| `QSR_CMPRSR_V1` | Question-specific Extractive | General purpose |

## Error Handling

Both clients use the same exception handling:

```python
from compresr import CompressionClient, QSCompressionClient
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
        compression_model_name="A_CMPRSR_V1"
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

- Documentation: [compresr.ai/docs/overview](https://compresr.ai/docs/overview)
- Support: [support@compresr.ai](mailto:support@compresr.ai)
- Issues: [GitHub Issues](https://github.com/compresr/sdk/issues)
- Website: [compresr.ai](https://compresr.ai)