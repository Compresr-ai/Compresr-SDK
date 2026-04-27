/**
 * CompressionClient - Token-level context compression
 *
 * Compresses context text to reduce token count before sending to LLMs.
 * Supports both agnostic (no query) and query-specific compression.
 *
 * Models:
 * - espresso_v1 (default): Agnostic compression, no query needed
 * - latte_v1: Query-specific compression, query REQUIRED
 *
 * Endpoints:
 *   Single:
 *     /compress/question-agnostic/     - context: string (no query)
 *     /compress/question-specific/     - context: string, query: string
 *   Batch:
 *     /compress/question-agnostic/batch - inputs: [{context}]
 *     /compress/question-specific/batch - inputs: [{context, query}]
 *   Stream:
 *     /compress/question-agnostic/stream - context: string
 *     /compress/question-specific/stream - context: string, query: string
 */
import { ZodError } from 'zod';
import {
  ENDPOINTS,
  MODELS,
} from '../config/index.js';
import { ValidationError } from '../errors/index.js';
import { HttpClient, type HttpClientOptions } from '../http/client.js';
import {
  CompressRequestSchema,
  CompressResponseSchema,
  AgnosticBatchRequestSchema,
  CompressBatchRequestSchema,
  CompressBatchResponseSchema,
  type CompressResponse,
  type CompressBatchResponse,
  type StreamChunk,
} from '../schemas/index.js';

/**
 * Options for single compression
 */
export interface CompressOptions {
  /** Context to compress (single string) */
  context: string;
  /** Compression model (default: espresso_v1) */
  compressionModelName?: string;
  /** Query for query-specific models (required for latte_v1) */
  query?: string;
  /**
   * Target compression ratio:
   * - 0-1: percentage to keep (e.g., 0.5 = keep 50%)
   * - >1: Nx factor (e.g., 4 = 4x compression = keep 25%)
   */
  targetCompressionRatio?: number;
  /** 
   * Paragraph-level compression (only for query-specific with latte_v1).
   * - true: faster, coarser compression (default for latte_v1)
   * - false: slower, token-level compression
   * Ignored for agnostic compression (no query).
   */
  coarse?: boolean;
}

/**
 * Options for batch compression
 * - If queries is undefined: uses agnostic endpoint (no queries required)
 * - If queries is provided: uses query-specific endpoint
 */
export interface CompressBatchOptions {
  /** List of contexts to compress (1-100 items) */
  contexts: string[];
  /** Single query for all contexts, or one query per context. Omit for agnostic batch. */
  queries?: string | string[];
  /** Compression model (default: espresso_v1) */
  compressionModelName?: string;
  /** Target compression ratio */
  targetCompressionRatio?: number;
  /** 
   * Paragraph-level compression (only for query-specific batch).
   * Ignored for agnostic batch (queries undefined).
   */
  coarse?: boolean;
}

/**
 * Token-level compression client
 *
 * @example
 * ```typescript
 * import { CompressionClient } from 'compresr';
 *
 * const client = new CompressionClient({ apiKey: 'cmp_...' });
 *
 * // Single agnostic compression
 * const result = await client.compress({
 *   context: 'Your long context...',
 * });
 *
 * // Single query-specific compression
 * const result = await client.compress({
 *   context: 'Your long context...',
 *   query: 'What is the main conclusion?',
 *   compressionModelName: 'latte_v1',
 * });
 *
 * // Batch agnostic compression
 * const result = await client.compressBatch({
 *   contexts: ['Doc 1...', 'Doc 2...', 'Doc 3...'],
 * });
 *
 * // Batch query-specific compression
 * const result = await client.compressBatch({
 *   contexts: ['Doc 1...', 'Doc 2...', 'Doc 3...'],
 *   queries: 'What are the key points?',
 *   compressionModelName: 'latte_v1',
 * });
 * ```
 */
export class CompressionClient {
  private readonly http: HttpClient;

  constructor(options: HttpClientOptions) {
    this.http = new HttpClient(options);
  }

  private resolveEndpoints(query?: string): {
    base: string;
    stream: string;
  } {
    if (query !== undefined) {
      return {
        base: ENDPOINTS.COMPRESS_QS,
        stream: ENDPOINTS.COMPRESS_QS_STREAM,
      };
    }
    return {
      base: ENDPOINTS.COMPRESS_AGNOSTIC,
      stream: ENDPOINTS.COMPRESS_AGNOSTIC_STREAM,
    };
  }

  private buildRequest(options: CompressOptions): Record<string, unknown> {
    const modelName = options.compressionModelName ?? MODELS.ESPRESSO;

    // Only include coarse when using query-specific endpoint
    // Agnostic endpoint doesn't support coarse parameter
    const effectiveCoarse = options.query !== undefined ? options.coarse : undefined;

    try {
      const request = CompressRequestSchema.parse({
        context: options.context,
        compression_model_name: modelName,
        query: options.query,
        target_compression_ratio: options.targetCompressionRatio,
        coarse: effectiveCoarse,
      });
      return request;
    } catch (error) {
      if (error instanceof ZodError) {
        const firstError = error.errors[0];
        throw new ValidationError(
          firstError?.message ?? 'Validation failed',
          firstError?.path.join('.')
        );
      }
      throw error;
    }
  }

  // ==========================================================================
  // Single Compression
  // ==========================================================================

  /**
   * Compress a single context
   *
   * For multiple contexts, use compressBatch().
   *
   * @example
   * ```typescript
   * // Agnostic compression (no query)
   * const result = await client.compress({
   *   context: 'Your long context text...',
   * });
   *
   * // Query-specific compression (with query)
   * const result = await client.compress({
   *   context: 'Your long context text...',
   *   query: 'What is the main conclusion?',
   *   compressionModelName: 'latte_v1',
   * });
   *
   * console.log(`Saved ${result.data.tokens_saved} tokens!`);
   * ```
   */
  async compress(options: CompressOptions): Promise<CompressResponse> {
    const request = this.buildRequest(options);
    const { base } = this.resolveEndpoints(options.query);

    const response = await this.http.post<unknown>(base, request);
    return CompressResponseSchema.parse(response);
  }

  /**
   * Stream compression chunks
   *
   * @example
   * ```typescript
   * for await (const chunk of client.compressStream({
   *   context: 'Your long context...',
   * })) {
   *   process.stdout.write(chunk.content);
   * }
   * ```
   */
  async *compressStream(
    options: CompressOptions
  ): AsyncGenerator<StreamChunk, void, undefined> {
    const request = this.buildRequest(options);
    const { stream } = this.resolveEndpoints(options.query);

    for await (const content of this.http.stream(stream, request)) {
      yield { content, done: false };
    }
    yield { content: '', done: true };
  }

  // ==========================================================================
  // Batch Compression
  // ==========================================================================

  /**
   * Batch compress multiple contexts
   *
   * - If queries is undefined: uses agnostic endpoint (no queries required)
   * - If queries is provided: uses query-specific endpoint
   *
   * @example
   * ```typescript
   * // Agnostic batch (no queries)
   * const result = await client.compressBatch({
   *   contexts: ['Doc 1...', 'Doc 2...', 'Doc 3...'],
   * });
   *
   * // Query-specific batch (same query for all)
   * const result = await client.compressBatch({
   *   contexts: ['Doc 1...', 'Doc 2...', 'Doc 3...'],
   *   queries: 'What are the key points?',
   *   compressionModelName: 'latte_v1',
   * });
   *
   * // Query-specific batch (different queries)
   * const result = await client.compressBatch({
   *   contexts: ['ML doc...', 'NLP doc...'],
   *   queries: ['What is ML?', 'What is NLP?'],
   *   compressionModelName: 'latte_v1',
   * });
   *
   * console.log(`Saved ${result.data.total_tokens_saved} tokens!`);
   * ```
   */
  async compressBatch(
    options: CompressBatchOptions
  ): Promise<CompressBatchResponse> {
    const modelName = options.compressionModelName ?? MODELS.ESPRESSO;

    try {
      if (options.queries === undefined) {
        // Agnostic batch (no queries)
        const inputs = options.contexts.map((context) => ({ context }));
        const request = AgnosticBatchRequestSchema.parse({
          inputs,
          compression_model_name: modelName,
          target_compression_ratio: options.targetCompressionRatio,
        });

        const response = await this.http.post<unknown>(
          ENDPOINTS.COMPRESS_AGNOSTIC_BATCH,
          request
        );
        return CompressBatchResponseSchema.parse(response);
      } else {
        // Query-specific batch
        const queryList =
          typeof options.queries === 'string'
            ? Array(options.contexts.length).fill(options.queries)
            : options.queries;

        if (queryList.length !== options.contexts.length) {
          throw new ValidationError(
            `Number of queries (${queryList.length}) must match number of contexts (${options.contexts.length})`
          );
        }

        const inputs = options.contexts.map((context, i) => ({
          context,
          // Safe: queryList length validated above
          query: queryList[i] as string,
        }));

        const request = CompressBatchRequestSchema.parse({
          inputs,
          compression_model_name: modelName,
          target_compression_ratio: options.targetCompressionRatio,
          coarse: options.coarse,
        });

        const response = await this.http.post<unknown>(
          ENDPOINTS.COMPRESS_QS_BATCH,
          request
        );
        return CompressBatchResponseSchema.parse(response);
      }
    } catch (error) {
      if (error instanceof ZodError) {
        const firstError = error.errors[0];
        throw new ValidationError(
          firstError?.message ?? 'Validation failed',
          firstError?.path.join('.')
        );
      }
      throw error;
    }
  }
}
