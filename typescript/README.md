# Compresr TypeScript SDK

[![npm](https://img.shields.io/npm/v/compresr)](https://www.npmjs.com/package/compresr)
[![TypeScript](https://img.shields.io/badge/TypeScript-%3E%3D5.0-blue)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Official TypeScript SDK for [Compresr](https://compresr.ai) - Intelligent context compression to reduce LLM API costs by 30-70%.

## Installation

```bash
npm install compresr
# or
yarn add compresr
# or
pnpm add compresr
```

## Quick Start

### API Key Setup

Get your API key from [compresr.ai](https://compresr.ai):

1. Create an account at [compresr.ai](https://compresr.ai)
2. Navigate to Dashboard → API Keys
3. Click "Create New Key" and copy it (shown only once!)

### Basic Usage

```typescript
import { CompressionClient } from 'compresr';

const client = new CompressionClient({ apiKey: 'cmp_your_api_key' });

// Agnostic compression (espresso_v1) — no query needed
const result = await client.compress({
  context: 'Your very long context that needs compression...',
});

console.log(`Original: ${result.data.original_tokens} tokens`);
console.log(`Compressed: ${result.data.compressed_tokens} tokens`);
console.log(`Saved: ${result.data.tokens_saved} tokens`);
```

### Query-Specific Compression

```typescript
// Query-specific compression (latte_v1) — query REQUIRED
const result = await client.compress({
  context: 'Python was created in 1991. JavaScript in 1995. Java in 1995.',
  query: 'Who created Python?',
  compressionModelName: 'latte_v1',
});

console.log(result.data.compressed_context);
// Output preserves Python-relevant content
```

## Compression Models

| Model | Query | Description |
|-------|-------|-------------|
| `espresso_v1` | Not needed | Agnostic compression — good for system prompts, documents |
| `latte_v1` | **Required** | Query-specific — preserves tokens relevant to your question |

## API Reference

### `CompressionClient`

```typescript
import { CompressionClient } from 'compresr';

const client = new CompressionClient({
  apiKey: 'cmp_your_api_key',  // Required
  timeout: 60000,              // Optional: request timeout in ms (default: 60000)
  baseUrl: 'https://api.compresr.ai', // Optional: override API URL
});
```

### Methods

#### `compress(options): Promise<CompressResponse>`

Compress context text.

```typescript
const result = await client.compress({
  context: 'Your context...',              // Required: string or string[]
  compressionModelName: 'espresso_v1',     // Optional (default: 'espresso_v1')
  query: 'Your question?',                 // Required for latte_v1
  targetCompressionRatio: 0.5,             // Optional: 0-1 or >1 for Nx
  coarse: false,                           // Optional: paragraph-level (latte_v1 only)
});
```

#### `compressStream(options): AsyncGenerator<StreamChunk>`

Stream compression chunks in real-time.

```typescript
for await (const chunk of client.compressStream({
  context: 'Your context...',
})) {
  if (!chunk.done) {
    process.stdout.write(chunk.content);
  }
}
```

#### `compressBatch(options): Promise<CompressBatchResponse>`

Batch compress multiple contexts (more efficient than multiple `compress()` calls).

```typescript
// Same query for all contexts
const result = await client.compressBatch({
  contexts: ['Doc 1...', 'Doc 2...', 'Doc 3...'],
  queries: 'What are the key points?',
  compressionModelName: 'latte_v1',  // Optional (default for batch)
});

// Different query per context
const result = await client.compressBatch({
  contexts: ['ML doc...', 'NLP doc...'],
  queries: ['What is ML?', 'What is NLP?'],
});

console.log(`Total saved: ${result.data.total_tokens_saved} tokens`);
```

## Response Types

### CompressResponse

```typescript
interface CompressResponse {
  success: boolean;
  data: {
    original_context: string | string[];
    compressed_context: string | string[];
    original_tokens: number;
    compressed_tokens: number;
    actual_compression_ratio: number;
    tokens_saved: number;
    duration_ms: number;
  } | null;
}
```

### CompressBatchResponse

```typescript
interface CompressBatchResponse {
  success: boolean;
  data: {
    results: CompressBatchItemResult[];
    total_original_tokens: number;
    total_compressed_tokens: number;
    total_tokens_saved: number;
    average_compression_ratio: number;
    count: number;
  } | null;
}
```

## Integration with OpenAI

```typescript
import { CompressionClient } from 'compresr';
import OpenAI from 'openai';

const compresr = new CompressionClient({ apiKey: 'cmp_xxx' });
const openai = new OpenAI({ apiKey: 'sk-xxx' });

// Compress your context first
const compressed = await compresr.compress({
  context: 'Your very long system prompt or document...',
});

// Use compressed context with OpenAI
const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [
    { role: 'system', content: compressed.data.compressed_context },
    { role: 'user', content: 'Analyze this data...' },
  ],
});

console.log(`Saved ${compressed.data.tokens_saved} tokens!`);
```

## Error Handling

```typescript
import {
  CompressionClient,
  CompresrError,
  AuthenticationError,
  ValidationError,
  RateLimitError,
} from 'compresr';

try {
  const result = await client.compress({ context: 'Hello' });
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Invalid API key:', error.message);
  } else if (error instanceof ValidationError) {
    console.error('Invalid request:', error.message, 'Field:', error.field);
  } else if (error instanceof RateLimitError) {
    console.error('Rate limited. Retry after:', error.retryAfter, 'seconds');
  } else if (error instanceof CompresrError) {
    console.error('API error:', error.message, 'Code:', error.code);
  }
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `AuthenticationError` | Invalid or missing API key |
| `ValidationError` | Invalid request parameters |
| `RateLimitError` | Too many requests (includes `retryAfter`) |
| `ScopeError` | API key lacks required permissions |
| `ServerError` | Internal server error |
| `ConnectionError` | Network/connection failure |
| `NotFoundError` | Resource not found |

## Requirements

- Node.js 18+ (uses native `fetch`)
- TypeScript 5.0+ (optional, but recommended)

## Development

```bash
# Install dependencies
npm install

# Run unit tests
npm test

# Run integration tests (requires COMPRESR_API_KEY)
npm run test:integration

# Build
npm run build

# Lint
npm run lint
```

## Support

- Email: support@compresr.ai
- [API Documentation](https://compresr.ai/docs)
- [GitHub Issues](https://github.com/compresr/sdk/issues)

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
