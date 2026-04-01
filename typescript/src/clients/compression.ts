/**
 * CompressionClient - Token-level context compression
 *
 * Compresses context text to reduce token count before sending to LLMs.
 * Supports both agnostic (no query) and query-specific compression.
 *
 * Models:
 * - espresso_v1 (default): Agnostic compression, no query needed
 * - latte_v1: Query-specific compression, query REQUIRED
 */
import { ZodError } from 'zod';
import {
  ENDPOINTS,
  ALLOWED_COMPRESSION_MODELS,
  QUERY_REQUIRED_MODELS,
  AGNOSTIC_ENDPOINT_MODELS,
  MODELS,
} from '../config/index.js';
import { ValidationError } from '../errors/index.js';
import { HttpClient, type HttpClientOptions } from '../http/client.js';
import {
  CompressRequestSchema,
  CompressResponseSchema,
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
  /** Context to compress - single string or array of strings */
  context: string | string[];
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
  /** Use paragraph-level compression (faster, latte_v1 only) */
  coarse?: boolean;
}

/**
 * Options for batch compression
 */
export interface CompressBatchOptions {
  /** List of contexts to compress (1-100 items) */
  contexts: string[];
  /** Single query for all contexts, or one query per context */
  queries: string | string[];
  /** Compression model (default: latte_v1 for batch) */
  compressionModelName?: string;
  /** Target compression ratio */
  targetCompressionRatio?: number;
  /** Use paragraph-level compression (faster) */
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
 * // Agnostic compression
 * const result = await client.compress({
 *   context: 'Your long context...',
 * });
 *
 * // Query-specific compression
 * const result = await client.compress({
 *   context: 'Your long context...',
 *   query: 'What is the main conclusion?',
 *   compressionModelName: 'latte_v1',
 * });
 * ```
 */
export class CompressionClient {
  private readonly http: HttpClient;

  constructor(options: HttpClientOptions) {
    this.http = new HttpClient(options);
  }

  /**
   * Validate that the model is allowed
   */
  private validateModel(modelName: string): void {
    if (!ALLOWED_COMPRESSION_MODELS.has(modelName)) {
      const allowed = Array.from(ALLOWED_COMPRESSION_MODELS).join(', ');
      throw new ValidationError(
        `Model '${modelName}' is not valid for CompressionClient. Allowed: ${allowed}`
      );
    }
  }

  /**
   * Validate query parameter against model requirements
   */
  private validateQueryForModel(modelName: string, query?: string): void {
    if (QUERY_REQUIRED_MODELS.has(modelName) && !query) {
      throw new ValidationError(
        `Model '${modelName}' requires a 'query' parameter.`
      );
    }
    if (!QUERY_REQUIRED_MODELS.has(modelName) && query !== undefined) {
      throw new ValidationError(
        `Model '${modelName}' does not accept a 'query' parameter. ` +
          `Remove the query or use 'latte_v1' for query-specific compression.`
      );
    }
  }

  /**
   * Resolve endpoints based on model name
   */
  private resolveEndpoints(modelName: string): {
    base: string;
    stream: string;
  } {
    if (AGNOSTIC_ENDPOINT_MODELS.has(modelName)) {
      return {
        base: ENDPOINTS.COMPRESS_AGNOSTIC,
        stream: ENDPOINTS.COMPRESS_AGNOSTIC_STREAM,
      };
    }
    return {
      base: ENDPOINTS.COMPRESS_QS,
      stream: ENDPOINTS.COMPRESS_QS_STREAM,
    };
  }

  /**
   * Build request payload
   */
  private buildRequest(options: CompressOptions): Record<string, unknown> {
    const modelName = options.compressionModelName ?? MODELS.ESPRESSO;

    this.validateModel(modelName);
    this.validateQueryForModel(modelName, options.query);

    try {
      const request = CompressRequestSchema.parse({
        context: options.context,
        compression_model_name: modelName,
        query: options.query,
        target_compression_ratio: options.targetCompressionRatio,
        coarse: options.coarse,
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
  // Public Methods
  // ==========================================================================

  /**
   * Compress context (async)
   *
   * @example
   * ```typescript
   * // Agnostic compression (espresso_v1)
   * const result = await client.compress({
   *   context: 'Your long context text...',
   * });
   *
   * // Query-specific compression (latte_v1)
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
    const modelName = options.compressionModelName ?? MODELS.ESPRESSO;
    const request = this.buildRequest(options);
    const { base } = this.resolveEndpoints(modelName);

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
    const modelName = options.compressionModelName ?? MODELS.ESPRESSO;
    const request = this.buildRequest(options);
    const { stream } = this.resolveEndpoints(modelName);

    for await (const content of this.http.stream(stream, request)) {
      yield { content, done: false };
    }
    yield { content: '', done: true };
  }

  /**
   * Batch compress multiple contexts
   *
   * More efficient than calling compress() multiple times.
   * Only supports query-specific models (latte_v1).
   *
   * @example
   * ```typescript
   * // Same query for all contexts
   * const result = await client.compressBatch({
   *   contexts: ['Doc 1...', 'Doc 2...', 'Doc 3...'],
   *   queries: 'What are the key points?',
   * });
   *
   * // Different query per context
   * const result = await client.compressBatch({
   *   contexts: ['ML doc...', 'NLP doc...'],
   *   queries: ['What is ML?', 'What is NLP?'],
   * });
   *
   * console.log(`Saved ${result.data.total_tokens_saved} tokens!`);
   * ```
   */
  async compressBatch(
    options: CompressBatchOptions
  ): Promise<CompressBatchResponse> {
    const modelName = options.compressionModelName ?? MODELS.LATTE;

    // Batch only supports query-specific models
    if (!QUERY_REQUIRED_MODELS.has(modelName)) {
      throw new ValidationError(
        `Batch compression only supports query-specific models: ${Array.from(QUERY_REQUIRED_MODELS).join(', ')}`
      );
    }

    this.validateModel(modelName);

    // Build query list
    const queryList =
      typeof options.queries === 'string'
        ? Array(options.contexts.length).fill(options.queries)
        : options.queries;

    if (queryList.length !== options.contexts.length) {
      throw new ValidationError(
        `Number of queries (${queryList.length}) must match number of contexts (${options.contexts.length})`
      );
    }

    // Build inputs
    const inputs = options.contexts.map((context, i) => ({
      context,
      query: queryList[i],
    }));

    try {
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
