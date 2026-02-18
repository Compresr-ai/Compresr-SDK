# Compresr curl SDK

REST API examples using curl. Reduce your LLM API costs by 30-70% through intelligent context compression.

## Quick Start

### 1. Get Your API Key

Get your API key from [compresr.ai](https://compresr.ai):

1. Create an account at [compresr.ai](https://compresr.ai)
2. Navigate to Dashboard → API Keys
3. Click "Create New Key" and copy it (shown only once!)

### 2. Set Environment Variables

```bash
# Required: Your API key (in ../.env or export it)
export COMPRESR_API_KEY="cmp_your_api_key_here"
```

### 3. Agnostic Compression (No Question Needed)

```bash
curl -X POST "https://api.compresr.ai/api/compress/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Your long context that needs compression...",
    "compression_model_name": "A_CMPRSR_V1"
  }'
```

### 4. Question-Specific Compression

```bash
curl -X POST "https://api.compresr.ai/api/compress/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Your long context with multiple topics...",
    "question": "What is the main conclusion?",
    "compression_model_name": "QS_CMPRSR_V1"
  }'
```

## Available Scripts

### Agnostic Compression (no question needed)

| Script | Description |
|--------|-------------|
| `compress.sh` | Compress single context |
| `compress_batch.sh` | Batch compress multiple contexts |
| `compress_stream.sh` | Stream compression (real-time) |

### Question-Specific Compression (requires question)

| Script | Description |
|--------|-------------|
| `compress_qs.sh` | Question-specific compression |

### Utilities

| Script | Description |
|--------|-------------|
| `models.sh` | List available models |
| `tokens.sh` | Count tokens |
| `health.sh` | Health checks |

## Available Models

### Agnostic Models

| Model | Description |
|-------|-------------|
| `A_CMPRSR_V1` | LLM-based abstractive compression (default) |
| `A_CMPRSR_V1_FLASH` | Fast extractive compression (LLMLingua-2) |

### Question-Specific Models

| Model | Description |
|-------|-------------|
| `QS_CMPRSR_V1` | Question-specific compression (default) |
| `QSR_CMPRSR_V1` | Question-specific with reranking |
| `QSLF_CMPRSR_V1` | Question-specific with longformer |

## Usage

```bash
# Set your API key in ../.env or export it
export COMPRESR_API_KEY="cmp_your_api_key_here"

# Run agnostic compression
./compress.sh

# Run question-specific compression
./compress_qs.sh
```

## Response Format

```json
{
  "success": true,
  "data": {
    "original_context": "Your long context...",
    "compressed_context": "Compressed version...",
    "original_tokens": 150,
    "compressed_tokens": 75,
    "actual_compression_ratio": 0.5,
    "tokens_saved": 75,
    "duration_ms": 123
  }
}
```

## Documentation

- API Docs: [compresr.ai/docs](https://compresr.ai/docs)
- Python SDK: [../python](../python)
```

## Documentation

- API Docs: [compresr.ai/docs](https://compresr.ai/docs)
- Python SDK: [../python](../python)
