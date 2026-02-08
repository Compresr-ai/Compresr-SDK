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
# Required: Your API key
export COMPRESR_API_KEY="cmp_your_api_key_here"

# Optional: Base URL (defaults to production)
export COMPRESR_BASE_URL="https://api.compresr.ai"
```

### 3. Compress Your Context

```bash
curl -X POST "${COMPRESR_BASE_URL:-https://api.compresr.ai}/api/compress/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Your long context that needs compression...",
    "compression_model_name": "cmprsr_v1"
  }'
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `compress.sh` | Compress single context |
| `compress_batch.sh` | Batch compress multiple contexts |
| `compress_stream.sh` | Stream compression (real-time) |
| `models.sh` | List available models |
| `tokens.sh` | Count tokens |
| `health.sh` | Health checks |

## Usage

```bash
# Set your API key
export COMPRESR_API_KEY="cmp_your_api_key_here"

# Run any script
./compress.sh
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

## Local Development

```bash
export COMPRESR_BASE_URL="http://localhost:8000"
export COMPRESR_API_KEY="your_local_api_key"
./compress.sh
```

## Documentation

- API Docs: [compresr.ai/docs](https://compresr.ai/docs)
- Python SDK: [../python](../python)
