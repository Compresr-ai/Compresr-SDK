/**
 * Compresr TypeScript SDK — query-aware context compression for LLMs.
 *
 * @example
 * ```typescript
 * import { CompressionClient } from 'compresr';
 *
 * const client = new CompressionClient({ apiKey: 'cmp_...' });
 * const result = await client.compress({
 *   context: 'Your long context...',
 *   query: 'What is the main conclusion?',
 * });
 * console.log(result.data?.compressed_context);
 * ```
 *
 * @packageDocumentation
 */

export {
  CompressionClient,
  type CompressOptions,
  type CompressBatchOptions,
  type CompressionClientOptions,
} from './clients/index.js';

// Agents (lazy peer-dep — only resolves when consumer touches the surface)
export {
  WebSearchTool,
  createWebSearchTool,
  type Citation,
  type CompresrStats,
  type CompressionPolicyOptions,
  type NormalizedResult,
  type TavilyOptions,
  type BraveOptions,
} from './agents/index.js';

// Logger (pluggable; replace via setLogger to route SDK warnings)
export { setLogger, type CompresrLogger } from './logger.js';

export { MODELS, type Model } from './config/index.js';

export { type RetryConfig } from './http/index.js';

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

export {
  type CompressResponse,
  type CompressResult,
  type CompressBatchResponse,
  type CompressBatchResult,
  type CompressBatchItemResult,
  type StreamChunk,
} from './schemas/index.js';

export {
  ResearchAgent,
  ResearchFacade,
  parseResearchOutput,
  DEFAULT_RESEARCH_SYSTEM_PROMPT,
  type ResearchAgentOptions,
  type ResearchAgentRunOptions,
  type ResearchRunOptions,
  type ParsedResearch,
  type ResearchResult,
  type ResearchUsage,
  type Step,
  type StepKind,
} from './agents/research/index.js';

export { SDK_VERSION as VERSION } from './version.js';
