/**
 * Compresr TypeScript SDK
 *
 * Intelligent context compression to reduce LLM API costs by 30-70%.
 *
 * @example
 * ```typescript
 * import { CompressionClient, MODELS } from 'compresr';
 *
 * const client = new CompressionClient({ apiKey: 'cmp_...' });
 *
 * // Agnostic compression (espresso_v1, default)
 * const result = await client.compress({
 *   context: 'Your long context...',
 * });
 * console.log(result.data.compressed_context);
 *
 * // Query-specific compression (latte_v1)
 * const result = await client.compress({
 *   context: 'Your long context...',
 *   query: 'What is the main conclusion?',
 *   compressionModelName: 'latte_v1',
 * });
 * ```
 *
 * @packageDocumentation
 */

// Clients
export {
  CompressionClient,
  type CompressOptions,
  type CompressBatchOptions,
} from './clients/index.js';

// Configuration
export { MODELS, type CompressionModel } from './config/index.js';

// Errors
export {
  CompresrError,
  AuthenticationError,
  ValidationError,
  RateLimitError,
  ScopeError,
  ServerError,
  ConnectionError,
  NotFoundError,
  type ErrorResponseData,
} from './errors/index.js';

// Schemas/Types
export {
  type CompressResponse,
  type CompressResult,
  type CompressBatchResponse,
  type CompressBatchResult,
  type CompressBatchItemResult,
  type StreamChunk,
} from './schemas/index.js';

// Version - re-export from version.ts
export { SDK_VERSION as VERSION } from './version.js';
