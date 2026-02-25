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

### 3. Agnostic Compression (No Query Needed)

Compress raw text (system prompts, documents) without a specific question. Removes less important tokens while preserving meaning.

```bash
curl -X POST "https://api.compresr.ai/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Your long context that needs compression...",
    "compression_model_name": "espresso_v1"
  }'
```

### 4. Query-Specific Compression

Compress text while preserving tokens relevant to a specific query. Ideal for RAG/QA pipelines where you want to shrink retrieved passages before sending to an LLM.

```bash
curl -X POST "https://api.compresr.ai/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Your long context with multiple topics...",
    "query": "What is the main conclusion?",
    "compression_model_name": "latte_v1"
  }'
```

### 5. Chunk-Level Filtering

Keep or drop entire chunks based on query relevance — no rewriting. Best when your retriever returns many chunks and you want to discard irrelevant ones before stuffing them into a prompt.

```bash
curl -X POST "https://api.compresr.ai/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": ["Chunk about Python...", "Chunk about Java...", "Chunk about ML..."],
    "query": "What is Python?",
    "compression_model_name": "coldbrew_v1"
  }'
```

## Available Scripts

### Agnostic Compression (no query needed)

| Script | Description |
|--------|-------------|
| `compress.sh` | Compress single context |
| `compress_batch.sh` | Compress multiple contexts (list input) |
| `compress_stream.sh` | Stream compression (real-time) |

### Query-Specific Compression (requires query)

| Script | Description |
|--------|-------------|
| `compress_qs.sh` | Query-specific compression |

### Utilities

| Script | Description |
|--------|-------------|
| `models.sh` | List available models |
| `tokens.sh` | Count tokens |
| `health.sh` | Health checks |

## Available Models

### Compression Models

| Model | Query | Best For |
|-------|-------|----------|
| `espresso_v1` | Not needed | System prompts, documents, general context |
| `latte_v1` | **Required** | RAG / QA — preserve answer-relevant tokens |

### Filter Models

| Model | Query | Best For |
|-------|-------|----------|
| `coldbrew_v1` | **Required** | RAG chunk selection — drop irrelevant chunks |

## Usage

```bash
# Set your API key in ../.env or export it
export COMPRESR_API_KEY="cmp_your_api_key_here"

# Run agnostic compression
./compress.sh

# Run query-specific compression
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
