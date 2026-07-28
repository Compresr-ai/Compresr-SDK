/**
 * ``CompressionClient`` — query-aware context compression.
 *
 * Only the question-specific endpoint is exposed; pass any
 * ``compressionModelName`` you have enabled on the backend. ``query`` is
 * optional client-side — the backend validates whether the chosen model
 * requires it.
 */
import { ZodError } from 'zod';

import { ENDPOINTS, MODELS } from '../config/index.js';
import { CompresrError, ValidationError } from '../errors/index.js';
import { HttpClient, type HttpClientOptions } from '../http/client.js';
import {
  CompressBatchRequestSchema,
  CompressBatchResponseSchema,
  CompressRequestSchema,
  CompressResponseSchema,
  type CompressBatchResponse,
  type CompressResponse,
  type StreamChunk,
} from '../schemas/index.js';
import {
  CompresrEngine,
  anthropicMessages,
  openaiChatCompletions,
  nativeRun,
  type CompressionPolicyOptions,
  type NormalizedResult,
} from '../agents/index.js';
import type { AnthropicMessagesFacade } from '../agents/facades/anthropic.js';
import type { OpenAIChatFacade } from '../agents/facades/openai.js';
import type { NativeRunOptions } from '../agents/facades/native.js';
import { ResearchFacade } from '../agents/research/index.js';

export interface CompressOptions {
  context: string;
  query?: string;
  compressionModelName?: string;
  targetCompressionRatio?: number;
  coarse?: boolean;
  heuristicChunking?: boolean;
  disablePlaceholders?: boolean;
  /** latte_v2 only. Adaptive (Kneedle elbow) selection; overrides
   *  `targetCompressionRatio`. */
  dynamic?: boolean;
  /** latte_v2 only. Floor on adaptive compression (server default 1.5x). */
  dynamicMinRatio?: number;
  /** latte_v2 only. Ceiling on adaptive compression (server default 10x). */
  dynamicMaxRatio?: number;
}

export interface BatchInput {
  context: string;
  query?: string;
}

/**
 * Two equivalent input forms:
 *  - Convenience: `contexts` (+ optional `queries` as string or string[]).
 *  - Pair form (matches the wire format): `inputs: [{ context, query }, ...]`.
 * Pass exactly one of `inputs` or `contexts`.
 */
export interface CompressBatchOptions {
  contexts?: string[];
  queries?: string | string[];
  inputs?: BatchInput[];
  compressionModelName?: string;
  targetCompressionRatio?: number;
  coarse?: boolean;
  heuristicChunking?: boolean;
  disablePlaceholders?: boolean;
  /** latte_v2 only. Adaptive selection; overrides `targetCompressionRatio`. */
  dynamic?: boolean;
  /** latte_v2 only. Floor on adaptive compression (server default 1.5x). */
  dynamicMinRatio?: number;
  /** latte_v2 only. Ceiling on adaptive compression (server default 10x). */
  dynamicMaxRatio?: number;
}

export interface CompressionClientOptions extends HttpClientOptions {
  /** Optional ``"provider:model"`` (or ``"provider/model"``) selector to opt
   * into the provider-shape facades (``.messages``, ``.chat``, ``run``). */
  llm?: string;
  /** Optional provider API key forwarded to the underlying LLM. */
  llmApiKey?: string;
  /** Optional compression policy applied to the agent's tool middleware. */
  compression?: CompressionPolicyOptions;
  /** Provider-aware prompt caching. Anthropic: middleware. OpenAI: prompt_cache_key
   *  + retention mapping. Gemini: no-op (implicit at API). Default: true. */
  enablePromptCache?: boolean;
  /** Cache TTL: "5m" or "1h". Anthropic ephemeral TTL; OpenAI maps "1h" -> "24h". */
  promptCacheTtl?: '5m' | '1h';
  /** Anthropic: skip caching below this conversation length. */
  promptCacheMinMessages?: number;
  /** OpenAI routing key — improves cache hit rate. Ignored for non-OpenAI. */
  openaiPromptCacheKey?: string;
}

export class CompressionClient {
  private readonly http: HttpClient;
  private readonly llm?: string;
  private readonly llmApiKey?: string;
  private readonly compressionPolicy?: CompressionPolicyOptions;
  private readonly enablePromptCache: boolean;
  private readonly promptCacheTtl: '5m' | '1h';
  private readonly promptCacheMinMessages: number;
  private readonly openaiPromptCacheKey?: string;
  private engine?: CompresrEngine;
  private anthropicFacade?: AnthropicMessagesFacade;
  private openaiFacade?: OpenAIChatFacade;
  private researchFacade?: ResearchFacade;

  constructor(options: CompressionClientOptions) {
    // Env fallback for missing apiKey — browser-safe (typeof process check).
    // File fallback (`~/.compresr/credentials.json`) lives in the Node-only
    // `@compresr/sdk/auth` subpath: import { createClient } from
    // '@compresr/sdk/auth' for automatic pickup, or call resolveApiKey()
    // directly and pass the result here.
    let resolvedApiKey = options.apiKey;
    if (
      resolvedApiKey === undefined &&
      typeof process !== 'undefined' &&
      process.env?.COMPRESR_API_KEY
    ) {
      resolvedApiKey = process.env.COMPRESR_API_KEY;
    }
    this.http = new HttpClient({ ...options, apiKey: resolvedApiKey });
    if (options.llm !== undefined) this.llm = options.llm;
    if (options.llmApiKey !== undefined) this.llmApiKey = options.llmApiKey;
    if (options.compression !== undefined) {
      this.compressionPolicy = options.compression;
    }
    this.enablePromptCache = options.enablePromptCache ?? true;
    this.promptCacheTtl = options.promptCacheTtl ?? '5m';
    this.promptCacheMinMessages = options.promptCacheMinMessages ?? 2;
    if (options.openaiPromptCacheKey !== undefined) {
      this.openaiPromptCacheKey = options.openaiPromptCacheKey;
    }
  }

  private requireEngine(surface: string): CompresrEngine {
    if (this.engine !== undefined) return this.engine;
    if (this.llm === undefined) {
      throw new CompresrError(
        `CompressionClient.${surface} requires an LLM provider. ` +
          "Construct with new CompressionClient({ apiKey, llm: 'anthropic:claude-...', llmApiKey }).",
        'missing_llm'
      );
    }
    const engineOpts: ConstructorParameters<typeof CompresrEngine>[0] = {
      compresrClient: this,
      llm: this.llm,
      enablePromptCache: this.enablePromptCache,
      promptCacheTtl: this.promptCacheTtl,
      promptCacheMinMessages: this.promptCacheMinMessages,
    };
    if (this.openaiPromptCacheKey !== undefined) {
      engineOpts.openaiPromptCacheKey = this.openaiPromptCacheKey;
    }
    if (this.llmApiKey !== undefined) engineOpts.llmApiKey = this.llmApiKey;
    if (this.compressionPolicy !== undefined) {
      engineOpts.policy = this.compressionPolicy;
    }
    this.engine = new CompresrEngine(engineOpts);
    return this.engine;
  }

  /** Anthropic-shaped facade: ``client.messages.create({...})``. */
  get messages(): AnthropicMessagesFacade {
    const engine = this.requireEngine('messages.create');
    this.anthropicFacade ??= anthropicMessages(engine);
    return this.anthropicFacade;
  }

  /** OpenAI-shaped facade: ``client.chat.completions.create({...})``. */
  get chat(): OpenAIChatFacade {
    const engine = this.requireEngine('chat.completions.create');
    this.openaiFacade ??= openaiChatCompletions(engine);
    return this.openaiFacade;
  }

  /** Native facade — returns the provider-agnostic ``NormalizedResult``. */
  async run(options: NativeRunOptions): Promise<NormalizedResult> {
    const engine = this.requireEngine('run');
    return nativeRun(engine, options);
  }

  /**
   * Research facade: ``await client.research.run("question")``.
   *
   * Multi-step web-research agent with per-snippet `latte_v1` compression and
   * multi-provider prompt caching. Requires `llm=` on the client.
   */
  get research(): ResearchFacade {
    const engine = this.requireEngine('research.run');
    // ResearchFacade takes a structural ResearchEngine; CompresrEngine has the
    // required @internal accessors but its types are wider, so cast at the
    // single integration point.
    this.researchFacade ??= new ResearchFacade(
      engine as unknown as ConstructorParameters<typeof ResearchFacade>[0]
    );
    return this.researchFacade;
  }

  private buildRequest(options: CompressOptions): Record<string, unknown> {
    try {
      return CompressRequestSchema.parse({
        context: options.context,
        query: options.query,
        compression_model_name: options.compressionModelName ?? MODELS.LATTE,
        target_compression_ratio: options.targetCompressionRatio,
        coarse: options.coarse,
        heuristic_chunking: options.heuristicChunking,
        disable_placeholders: options.disablePlaceholders,
        dynamic: options.dynamic,
        dynamic_min_ratio: options.dynamicMinRatio,
        dynamic_max_ratio: options.dynamicMaxRatio,
      });
    } catch (error) {
      throw mapZodError(error);
    }
  }

  async compress(options: CompressOptions): Promise<CompressResponse> {
    const request = this.buildRequest(options);
    const response = await this.http.post<unknown>(ENDPOINTS.COMPRESS, request);
    return CompressResponseSchema.parse(response);
  }

  async *compressStream(
    options: CompressOptions
  ): AsyncGenerator<StreamChunk, void, undefined> {
    const request = this.buildRequest(options);
    for await (const content of this.http.stream(ENDPOINTS.COMPRESS_STREAM, request)) {
      yield { content, done: false };
    }
    yield { content: '', done: true };
  }

  async compressBatch(options: CompressBatchOptions): Promise<CompressBatchResponse> {
    const inputs = buildBatchInputs(options);

    try {
      const request = CompressBatchRequestSchema.parse({
        inputs,
        compression_model_name: options.compressionModelName ?? MODELS.LATTE,
        target_compression_ratio: options.targetCompressionRatio,
        coarse: options.coarse,
        heuristic_chunking: options.heuristicChunking,
        disable_placeholders: options.disablePlaceholders,
        dynamic: options.dynamic,
        dynamic_min_ratio: options.dynamicMinRatio,
        dynamic_max_ratio: options.dynamicMaxRatio,
      });
      const response = await this.http.post<unknown>(ENDPOINTS.COMPRESS_BATCH, request);
      return CompressBatchResponseSchema.parse(response);
    } catch (error) {
      throw mapZodError(error);
    }
  }
}

function buildBatchInputs(options: CompressBatchOptions): BatchInput[] {
  if (options.inputs !== undefined && options.contexts !== undefined) {
    throw new ValidationError('Pass `inputs` OR `contexts`, not both.');
  }
  if (options.inputs !== undefined) {
    return options.inputs.map((it) => ({
      context: it.context,
      ...(it.query !== undefined ? { query: it.query } : {}),
    }));
  }
  if (options.contexts === undefined) {
    throw new ValidationError('Must provide either `inputs` or `contexts`.');
  }
  const queryList = resolveQueryList(options.contexts, options.queries);
  return options.contexts.map((context, i) => ({
    context,
    ...(queryList[i] !== undefined ? { query: queryList[i] } : {}),
  }));
}

function resolveQueryList(
  contexts: string[],
  queries: string | string[] | undefined
): Array<string | undefined> {
  if (queries === undefined) {
    return contexts.map(() => undefined);
  }
  if (typeof queries === 'string') {
    return contexts.map(() => queries);
  }
  if (queries.length !== contexts.length) {
    throw new ValidationError(
      `Number of queries (${queries.length}) must match contexts (${contexts.length})`
    );
  }
  return queries;
}

function mapZodError(error: unknown): unknown {
  if (error instanceof ZodError) {
    const first = error.issues[0];
    return new ValidationError(
      first?.message ?? 'Validation failed',
      first?.path.map(String).join('.')
    );
  }
  return error;
}
