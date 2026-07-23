/**
 * Provider-agnostic agent engine — TypeScript port of Wave 2A.
 *
 * ``CompresrEngine`` wraps LangChain.js's ``initChatModel`` + ``createAgent``
 * with Compresr's ``compresrToolMiddleware`` and normalizes the final
 * ``AIMessage`` into a provider-shape-free :class:`NormalizedResult` that
 * facades will remap back into provider-native shapes.
 *
 * Mirrors Python ``compresr/agents/engine.py``.
 */
import type { CompressionClient } from '../clients/compression.js';
import { CompresrError } from '../errors/index.js';
import { getLogger } from '../logger.js';
import {
  DEFAULT_MIN_TOKENS,
  DEFAULT_MODEL,
  DEFAULT_POLICY,
  DEFAULT_RATIO,
  type ErrorPolicy,
} from '../integrations/_shared/index.js';

import {
  defaultCompresrStats,
  type Citation,
  type NormalizedResult,
  type NormalizedToolUse,
} from './normalized.js';
import type {
  ChatModelLike,
  ResearchEngine,
  SearchToolLike,
} from './research/agent.js';

const KNOWN_PROVIDERS = new Set(['anthropic', 'openai', 'google_genai']);

/**
 * Tool names produced by ``WebSearchTool.tavily(...)`` and ``WebSearchTool.brave(...)``.
 *
 * When the generic-agent path (``client.messages.create`` / ``chat.completions.create``
 * / ``client.run``) is invoked with **exactly one** tool whose ``.name`` matches one
 * of these, the engine silently routes the call through the deep-research loop
 * instead of the standard ``createAgent`` middleware path. The caller still
 * receives their facade's expected response shape; only the loop semantics
 * change (``tool_choice="none"`` on the final step, per-snippet compression
 * against the live query, 10-step cap, manual Anthropic ``cache_control``).
 *
 * Detection is by name because ``WebSearchTool`` returns a LangChain
 * ``StructuredTool`` — not a custom subclass — so ``instanceof`` would always
 * be ``false``.
 */
export const WEB_SEARCH_TOOL_NAMES: ReadonlySet<string> = new Set([
  'tavily_search',
  'brave_search',
]);

/** True when ``tools`` is exactly one Compresr-built web-search tool. */
export function isSoloWebSearch(tools: ReadonlyArray<unknown>): boolean {
  if (tools.length !== 1) return false;
  const name = (tools[0] as { name?: unknown })?.name;
  return typeof name === 'string' && WEB_SEARCH_TOOL_NAMES.has(name);
}

/**
 * Pull the last user-role message text out of a facade-shaped chain.
 *
 * Supports plain dicts (Anthropic / OpenAI style) and any object with a
 * ``content`` field. Falls back to the first message's content when no
 * explicit user message is found, and to ``""`` on empty input.
 */
export function extractLastUserText(messages: ReadonlyArray<unknown>): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i] as { role?: unknown; type?: unknown; content?: unknown };
    const role = msg?.role ?? msg?.type;
    if ((role === 'user' || role === 'human') && msg?.content !== undefined) {
      return contentTextAny(msg.content);
    }
  }
  if (messages.length > 0) {
    const first = messages[0] as { content?: unknown };
    return contentTextAny(first?.content);
  }
  return '';
}

function contentTextAny(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (block && typeof block === 'object') {
        const t = (block as { text?: unknown }).text;
        if (typeof t === 'string') parts.push(t);
      } else if (typeof block === 'string') {
        parts.push(block);
      }
    }
    return parts.join('');
  }
  if (content === undefined || content === null) return '';
  if (typeof content === 'number' || typeof content === 'boolean') return String(content);
  // Avoid the default ``[object Object]`` stringification for arbitrary objects.
  try {
    return JSON.stringify(content);
  } catch {
    return '';
  }
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(',')}]`;
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`).join(',')}}`;
}

/**
 * Options that propagate to the underlying chat model via ``.bind(...)`` per call.
 *
 * Covers Anthropic / OpenAI / Gemini. LangChain.js normalizes most of these
 * across providers; unsupported keys are silently ignored upstream. Binding is
 * applied to a fresh wrapper per call so the cached chat model is never
 * polluted across requests.
 */
const LLM_BIND_OPTIONS = new Set<string>([
  'temperature',
  'topP',
  'topK',
  'maxTokens',
  'maxOutputTokens',
  'stop',
  'stopSequences',
  'presencePenalty',
  'frequencyPenalty',
  'seed',
  'logprobs',
  'topLogprobs',
]);

/**
 * Immutable bundle of compression knobs forwarded to the tool middleware.
 *
 * Mirrors Python ``compresr.integrations._shared.CompressionPolicy``.
 */
export interface CompressionPolicyOptions {
  targetCompressionRatio?: number;
  compressionModelName?: string;
  coarse?: boolean;
  minTokens?: number;
  onError?: ErrorPolicy;
  allowTools?: Iterable<string>;
  ignoreTools?: Iterable<string>;
}

interface ResolvedPolicy {
  readonly targetCompressionRatio: number;
  readonly compressionModelName: string;
  readonly minTokens: number;
  readonly onError: ErrorPolicy;
  readonly coarse?: boolean;
  readonly allowTools?: ReadonlyArray<string>;
  readonly ignoreTools?: ReadonlyArray<string>;
}

function resolvePolicy(input?: CompressionPolicyOptions): ResolvedPolicy {
  const base = input ?? {};
  const policy: ResolvedPolicy = {
    targetCompressionRatio: base.targetCompressionRatio ?? DEFAULT_RATIO,
    compressionModelName: base.compressionModelName ?? DEFAULT_MODEL,
    minTokens: base.minTokens ?? DEFAULT_MIN_TOKENS,
    onError: base.onError ?? DEFAULT_POLICY,
    ...(base.coarse !== undefined ? { coarse: base.coarse } : {}),
    ...(base.allowTools !== undefined
      ? { allowTools: [...base.allowTools] }
      : {}),
    ...(base.ignoreTools !== undefined
      ? { ignoreTools: [...base.ignoreTools] }
      : {}),
  };
  return policy;
}

/**
 * Parse an LLM spec into ``{ provider, modelName }``.
 *
 * Accepted forms:
 *  - Bare provider: ``"anthropic"`` -> ``{ provider, modelName: undefined }``.
 *    The model must then be supplied at the call site.
 *  - LangChain-style colon: ``"anthropic:claude-opus-4-8"``.
 *  - Vercel-style slash: ``"anthropic/claude-opus-4-8"`` — normalized to the
 *    colon form to match LangChain.js's ``initChatModel`` convention.
 */
export function parseLlmSpec(llm: string): {
  provider: string;
  modelName: string | undefined;
} {
  const normalized = llm.includes(':')
    ? llm
    : llm.includes('/')
      ? llm.replace('/', ':')
      : llm;

  if (!normalized.includes(':')) {
    const provider = normalized.trim();
    if (!provider) {
      throw new CompresrError(
        "llm must be a provider (e.g. 'anthropic') or 'provider:model' " +
          "(e.g. 'anthropic:claude-opus-4-8')",
        'invalid_llm_spec'
      );
    }
    return { provider, modelName: undefined };
  }

  const idx = normalized.indexOf(':');
  const provider = normalized.slice(0, idx).trim();
  const modelName = normalized.slice(idx + 1).trim();

  if (!provider || !modelName) {
    throw new CompresrError(
      "llm must be 'provider:model' (e.g. 'anthropic:claude-opus-4-8')",
      'invalid_llm_spec'
    );
  }
  return { provider, modelName };
}

export interface CompresrEngineOptions {
  compresrClient: CompressionClient;
  llm: string;
  llmApiKey?: string;
  policy?: CompressionPolicyOptions;
  /** Provider-aware prompt caching. Anthropic: wires middleware. OpenAI:
   *  attaches prompt_cache_key + retention mapping. Gemini: no-op for now. */
  enablePromptCache?: boolean;
  /** Cache TTL: "5m" or "1h". Anthropic ephemeral TTL; OpenAI maps "1h" -> "24h"
   *  retention. Default: "5m". */
  promptCacheTtl?: '5m' | '1h';
  /** Anthropic: skip caching until conversation has at least this many messages. */
  promptCacheMinMessages?: number;
  /** OpenAI routing key — improves cache hit rate when multiple clients share a
   *  backend. Ignored for non-OpenAI providers. */
  openaiPromptCacheKey?: string;
}

export interface RunOptions {
  messages: ReadonlyArray<unknown>;
  tools?: ReadonlyArray<unknown>;
  system?: string;
  maxTokens?: number;
  config?: Record<string, unknown>;
  /**
   * Per-call model name. If omitted, falls back to the engine's default
   * (the model encoded in ``llm: 'provider:model'`` at construction).
   * When neither is provided, ``run`` throws ``CompresrError("missing_model")``.
   */
  model?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  /** Gemini-native alias of ``maxTokens``. The engine aliases when needed. */
  maxOutputTokens?: number;
  stop?: string[];
  stopSequences?: string[];
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  logprobs?: boolean;
  topLogprobs?: number;
}

interface LangChainBindings {
  initChatModel: (modelStr: string, kwargs?: Record<string, unknown>) => unknown;
  createAgent: (params: Record<string, unknown>) => unknown;
}

let cachedLc: LangChainBindings | undefined;

async function importLangChain(): Promise<LangChainBindings> {
  if (cachedLc) return cachedLc;
  let mod: Record<string, unknown>;
  try {
    mod = (await import('langchain'));
  } catch {
    throw new CompresrError(
      'CompresrEngine requires langchain>=1.0. Install with: ' +
        'npm install langchain @langchain/core',
      'missing_peer_dependency'
    );
  }
  const initChatModel = mod.initChatModel as
    | LangChainBindings['initChatModel']
    | undefined;
  const createAgent = mod.createAgent as
    | LangChainBindings['createAgent']
    | undefined;
  if (typeof initChatModel !== 'function' || typeof createAgent !== 'function') {
    throw new CompresrError(
      'langchain package is missing initChatModel / createAgent — ' +
        'upgrade to langchain>=1.0.',
      'missing_peer_dependency'
    );
  }
  cachedLc = { initChatModel, createAgent };
  return cachedLc;
}

/**
 * Test seam — lets vitest stub LangChain without monkey-patching
 * ``node_modules``.
 *
 * @internal — not a stable public API; reserved for SDK tests only.
 */
export function _setLangChainBindings(b: LangChainBindings | undefined): void {
  cachedLc = b;
}

function contentText(msg: unknown): string {
  const content = (msg as { content?: unknown })?.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (block && typeof block === 'object') {
        const b = block as { type?: string; text?: unknown };
        if (b.type === 'text' && typeof b.text === 'string') {
          parts.push(b.text);
        } else if (typeof b.text === 'string') {
          parts.push(b.text);
        }
      } else if (typeof block === 'string') {
        parts.push(block);
      }
    }
    return parts.join('');
  }
  return '';
}

function normalizeContentBlocks(msg: unknown): unknown[] {
  const m = msg as { content_blocks?: unknown[]; content?: unknown };
  if (Array.isArray(m?.content_blocks) && m.content_blocks.length > 0) {
    return [...m.content_blocks];
  }
  if (Array.isArray(m?.content)) {
    return [...(m.content as unknown[])];
  }
  return [];
}

function normalizeToolUses(msg: unknown): NormalizedToolUse[] {
  const tcs = (msg as { tool_calls?: unknown[] })?.tool_calls;
  if (!Array.isArray(tcs)) return [];
  const out: NormalizedToolUse[] = [];
  for (const raw of tcs) {
    if (!raw || typeof raw !== 'object') continue;
    const tc = raw as { id?: string; name?: string; args?: Record<string, unknown> };
    out.push({
      ...(tc.id !== undefined ? { id: tc.id } : {}),
      ...(tc.name !== undefined ? { name: tc.name } : {}),
      ...(tc.args !== undefined ? { input: tc.args } : {}),
    });
  }
  return out;
}

function extractCitations(msg: unknown): Citation[] {
  const out: Citation[] = [];
  const blocks = normalizeContentBlocks(msg);
  for (const block of blocks) {
    if (!block || typeof block !== 'object') continue;
    const b = block as {
      citations?: unknown[];
      annotations?: unknown[];
    };
    for (const cit of b.citations ?? []) {
      if (!cit || typeof cit !== 'object') continue;
      const c = cit as {
        type?: string;
        url?: string;
        title?: string;
        cited_text?: string;
      };
      if (c.type === undefined || c.type === 'web_search_result_location') {
        if (!c.url) continue;
        out.push({
          url: c.url,
          ...(c.title !== undefined ? { title: c.title } : {}),
          ...(c.cited_text !== undefined ? { citedText: c.cited_text } : {}),
          providerMetadata: { ...c },
        });
      }
    }
    for (const ann of b.annotations ?? []) {
      if (!ann || typeof ann !== 'object') continue;
      const a = ann as {
        type?: string;
        url?: string;
        title?: string;
        text?: string;
        cited_text?: string;
      };
      if (a.type === 'url_citation') {
        if (!a.url) continue;
        out.push({
          url: a.url,
          ...(a.title !== undefined ? { title: a.title } : {}),
          ...(a.text !== undefined
            ? { citedText: a.text }
            : a.cited_text !== undefined
              ? { citedText: a.cited_text }
              : {}),
          providerMetadata: { ...a },
        });
      }
    }
  }
  const responseMetadata =
    (msg as { response_metadata?: Record<string, unknown> })?.response_metadata ?? {};
  const grounding =
    (responseMetadata['grounding_metadata'] as Record<string, unknown> | undefined) ??
    {};
  const chunks = grounding['grounding_chunks'];
  if (Array.isArray(chunks)) {
    for (const chunk of chunks) {
      if (!chunk || typeof chunk !== 'object') continue;
      const web = ((chunk as { web?: unknown }).web ?? {}) as {
        uri?: string;
        title?: string;
      };
      if (!web.uri) continue;
      out.push({
        url: web.uri,
        ...(web.title !== undefined ? { title: web.title } : {}),
        providerMetadata: { ...(chunk as Record<string, unknown>) },
      });
    }
  }
  return out;
}

function normalizeStopReason(msg: unknown): string {
  const meta = (msg as { response_metadata?: Record<string, unknown> })
    ?.response_metadata;
  const stop = meta?.['stop_reason'] ?? meta?.['finish_reason'];
  if (typeof stop === 'string' && stop) return stop;
  return 'end_turn';
}

function normalizeUsage(msg: unknown): Record<string, number> {
  const direct = (msg as { usage_metadata?: Record<string, unknown> })
    ?.usage_metadata;
  if (direct && typeof direct === 'object') {
    return coerceUsage(direct);
  }
  const meta = (msg as { response_metadata?: Record<string, unknown> })
    ?.response_metadata;
  const u = meta?.['usage'];
  if (u && typeof u === 'object') {
    return coerceUsage(u as Record<string, unknown>);
  }
  return {};
}

function coerceUsage(raw: Record<string, unknown>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (typeof v === 'number') out[k] = v;
    else if (typeof v === 'string' && !Number.isNaN(Number(v))) {
      out[k] = Number(v);
    }
  }
  return out;
}

/**
 * Sum usage across every AIMessage in the conversation.
 *
 * Each agent turn produces one AIMessage with its own usage block. Reading
 * only the last turn (as the previous code did) under-counted any multi-turn
 * loop — tool use, ReAct, etc. — by N-1 LLM round-trips. Walks the chain
 * and adds, including cache numbers nested under ``input_token_details``.
 */
const USAGE_NUMERIC_KEYS = [
  'input_tokens',
  'output_tokens',
  'total_tokens',
  'cache_read_input_tokens',
  'cache_creation_input_tokens',
] as const;

const INPUT_DETAILS_CACHE_READ_ALIASES = [
  'cached_tokens',
  'cache_read',
  'priority_cache_read',
  'flex_cache_read',
  'cached_content_token_count',
] as const;

function aggregateUsage(messages: unknown[]): Record<string, number> {
  const total: Record<string, number> = {};
  for (const k of USAGE_NUMERIC_KEYS) total[k] = 0;
  let aiMessageCount = 0;
  for (const msg of messages) {
    const ctor = (msg as { constructor?: { name?: string } })?.constructor?.name;
    if (ctor !== 'AIMessage') continue;
    const per = normalizeUsage(msg);
    if (Object.keys(per).length === 0) continue;
    aiMessageCount++;
    for (const k of USAGE_NUMERIC_KEYS) {
      const v = per[k];
      if (typeof v === 'number') total[k] += v;
    }
    // Top-level cache fields (newer LangChain) win over nested
    // ``input_token_details`` (older) to avoid double-counting.
    const hasTopCache =
      typeof per['cache_read_input_tokens'] === 'number' ||
      typeof per['cache_creation_input_tokens'] === 'number';
    if (!hasTopCache) {
      const rawPer = (msg as { usage_metadata?: Record<string, unknown> })
        ?.usage_metadata;
      const details = rawPer?.['input_token_details'];
      if (details && typeof details === 'object') {
        const d = details as Record<string, unknown>;
        let msgCacheRead = 0;
        for (const alias of INPUT_DETAILS_CACHE_READ_ALIASES) {
          const v = d[alias];
          if (typeof v === 'number') msgCacheRead += v;
        }
        // langchain-anthropic zeroes cache_creation when ttl-specific keys are set.
        const ttl5m = typeof d['ephemeral_5m_input_tokens'] === 'number' ? (d['ephemeral_5m_input_tokens']) : 0;
        const ttl1h = typeof d['ephemeral_1h_input_tokens'] === 'number' ? (d['ephemeral_1h_input_tokens']) : 0;
        const ttlWrites = ttl5m + ttl1h;
        const genericCreate = typeof d['cache_creation'] === 'number' ? (d['cache_creation']) : 0;
        const msgCacheCreate = ttlWrites > 0 ? ttlWrites : genericCreate;
        total['cache_read_input_tokens'] += msgCacheRead;
        total['cache_creation_input_tokens'] += msgCacheCreate;
        // langchain-anthropic 1.x inflates input_tokens to fresh+read+create.
        const thisInput = typeof rawPer?.['input_tokens'] === 'number' ? (rawPer['input_tokens']) : 0;
        if ((msgCacheRead > 0 || msgCacheCreate > 0) && thisInput > 0) {
          const overcount = msgCacheRead + msgCacheCreate;
          total['input_tokens'] -= Math.min(overcount, thisInput);
        }
      }
    }
    for (const [k, v] of Object.entries(per)) {
      if ((USAGE_NUMERIC_KEYS as readonly string[]).includes(k) || k === 'input_token_details') continue;
      if (typeof v === 'number') total[k] = (total[k] ?? 0) + v;
    }
  }
  total['ai_message_count'] = aiMessageCount;
  return total;
}

function toolName(t: unknown): string {
  const n = (t as { name?: unknown })?.name;
  if (typeof n === 'string') return n;
  if (t && typeof t === 'object' && typeof (t as Record<string, unknown>)['name'] === 'string') {
    return (t as Record<string, string>)['name'];
  }
  return '';
}

function sortedToolsForCache(tools: ReadonlyArray<unknown>): unknown[] {
  const indexed = tools.map((t, i) => ({ t, i }));
  indexed.sort((a, b) => {
    const an = toolName(a.t);
    const bn = toolName(b.t);
    if (an !== bn) return an < bn ? -1 : 1;
    return a.i - b.i;
  });
  return indexed.map(({ t }) => t);
}

function lastAiMessage(messages: unknown[]): unknown {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const ctor = (msg as { constructor?: { name?: string } })?.constructor?.name;
    if (ctor === 'AIMessage') return msg;
  }
  return messages.length > 0 ? messages[messages.length - 1] : null;
}

export class CompresrEngine {
  readonly provider: string;
  /**
   * Default model bound to the engine, parsed from the constructor's ``llm``
   * spec. ``undefined`` when the spec is a bare provider — in that case the
   * model must be supplied per-call on :meth:`run`.
   */
  readonly defaultModelName: string | undefined;

  private readonly compresrClient: CompressionClient;
  private readonly llmApiKey?: string;
  private readonly policy: ResolvedPolicy;
  private readonly chatModels: Map<string, unknown> = new Map();
  private readonly enablePromptCache: boolean;
  private readonly promptCacheTtl: '5m' | '1h';
  private readonly promptCacheMinMessages: number;
  private readonly openaiPromptCacheKey?: string;

  constructor(options: CompresrEngineOptions) {
    const { provider, modelName } = parseLlmSpec(options.llm);
    if (!KNOWN_PROVIDERS.has(provider)) {
      // Warn-only — LangChain may know providers we don't.
      getLogger().warn(
        `agents: unknown provider '${provider}'. ` +
          `Supported: ${[...KNOWN_PROVIDERS].join(', ')}.`
      );
    }
    this.provider = provider;
    this.defaultModelName = modelName;
    this.compresrClient = options.compresrClient;
    if (options.llmApiKey !== undefined) {
      this.llmApiKey = options.llmApiKey;
    }
    this.policy = resolvePolicy(options.policy);
    this.enablePromptCache = options.enablePromptCache ?? true;
    this.promptCacheTtl = options.promptCacheTtl ?? '5m';
    this.promptCacheMinMessages = options.promptCacheMinMessages ?? 2;
    if (options.openaiPromptCacheKey !== undefined) {
      this.openaiPromptCacheKey = options.openaiPromptCacheKey;
    }
  }

  /**
   * Back-compat alias for ``defaultModelName`` — returns the constructor
   * default and may be ``undefined`` when the engine was built with a bare
   * provider ``llm`` spec.
   */
  get modelName(): string | undefined {
    return this.defaultModelName;
  }

  /** @internal — used by ResearchAgent. Not part of the stable public API. */
  async _getChatModel(modelName: string): Promise<unknown> {
    return this.getChatModel(modelName);
  }

  /** @internal — used by ResearchAgent. Not part of the stable public API. */
  get _compresrClient(): CompressionClient {
    return this.compresrClient;
  }

  /** @internal — used by ResearchAgent. Not part of the stable public API. */
  _resolveModel(model?: string): string {
    const effective = model ?? this.defaultModelName;
    if (!effective) {
      throw new Error(
        "model is required — set it on the client (llm: 'anthropic:claude-...') or pass it to .research.run({ model: '...' })."
      );
    }
    return effective;
  }

  /** @internal — read by ResearchAgent for manual cache_control stamping. */
  get _promptCacheConfig(): {
    readonly enabled: boolean;
    readonly ttl: '5m' | '1h';
    readonly minMessages: number;
  } {
    return {
      enabled: this.enablePromptCache,
      ttl: this.promptCacheTtl,
      minMessages: this.promptCacheMinMessages,
    };
  }

  private async getChatModel(
    modelName: string,
    extraKwargs?: Record<string, unknown>
  ): Promise<unknown> {
    // LangChain.js's bind_tools strips a prior chat.bind(...)'s kwargs, so
    // we bake the per-call knobs (maxTokens, temperature, …) into the chat-
    // model constructor and cache per (modelName, extraKwargs).
    const cacheKey = `${modelName}::${stableStringify(extraKwargs ?? {})}`;
    const cached = this.chatModels.get(cacheKey);
    if (cached !== undefined) return cached;
    const { initChatModel } = await importLangChain();
    const kwargs: Record<string, unknown> = { ...(extraKwargs ?? {}) };
    if (this.llmApiKey !== undefined) kwargs['apiKey'] = this.llmApiKey;
    if (this.provider === 'openai' && this.enablePromptCache) {
      const openaiExtra = this.openaiCacheModelKwargs();
      if (Object.keys(openaiExtra).length > 0) {
        const existing = (kwargs['modelKwargs'] as Record<string, unknown> | undefined) ?? {};
        kwargs['modelKwargs'] = { ...existing, ...openaiExtra };
      }
    }
    let chat: unknown;
    if (this.provider === 'openai') {
      try {
        chat = initChatModel(`${this.provider}:${modelName}`, {
          ...kwargs,
          outputVersion: 'responses/v1',
        });
        this.chatModels.set(cacheKey, chat);
        return chat;
      } catch {
        // older @langchain/openai rejects unknown kwargs
      }
    }
    chat = initChatModel(`${this.provider}:${modelName}`, kwargs);
    this.chatModels.set(cacheKey, chat);
    return chat;
  }

  private openaiCacheModelKwargs(): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    if (this.openaiPromptCacheKey) out['prompt_cache_key'] = this.openaiPromptCacheKey;
    // Only set the retention override when caller asked for "1h" (24h tier);
    // default "5m" leaves OpenAI's server default ("in_memory") untouched.
    if (this.promptCacheTtl === '1h') out['prompt_cache_retention'] = '24h';
    return out;
  }

  private async buildMiddleware(): Promise<unknown[]> {
    // Lazy-load the existing langchain middleware so importing the engine
    // doesn't drag in @langchain/core at module load.
    const { compresrToolMiddleware } = await import(
      '../integrations/langchain/middleware.js'
    );
    const mwOpts: Record<string, unknown> = {
      client: this.compresrClient,
      targetCompressionRatio: this.policy.targetCompressionRatio,
      compressionModel: this.policy.compressionModelName,
      minTokens: this.policy.minTokens,
      onError: this.policy.onError,
    };
    if (this.policy.coarse !== undefined) mwOpts['coarse'] = this.policy.coarse;
    if (this.policy.allowTools !== undefined) {
      mwOpts['allowTools'] = this.policy.allowTools;
    }
    if (this.policy.ignoreTools !== undefined) {
      mwOpts['ignoreTools'] = this.policy.ignoreTools;
    }
    const mw: unknown[] = [compresrToolMiddleware(mwOpts)];
    const cacheMw = await this.maybeBuildCacheMiddleware();
    if (cacheMw !== null) {
      // AFTER compresrToolMiddleware so cache markers stamp post-compression content.
      mw.push(cacheMw);
    }
    return mw;
  }

  /** Anthropic prompt-cache middleware when enabled. Silent no-op on older langchain. */
  private async maybeBuildCacheMiddleware(): Promise<unknown> {
    if (!this.enablePromptCache || this.provider !== 'anthropic') return null;
    try {
      const mod = (await import('langchain')) as {
        anthropicPromptCachingMiddleware?: (opts: Record<string, unknown>) => unknown;
      };
      const factory = mod.anthropicPromptCachingMiddleware;
      if (typeof factory !== 'function') return null;
      return factory({
        ttl: this.promptCacheTtl,
        minMessagesToCache: this.promptCacheMinMessages,
        unsupportedModelBehavior: 'ignore',
      });
    } catch {
      return null;
    }
  }

  private async buildAgent(
    chat: unknown,
    tools: ReadonlyArray<unknown>,
    system?: string
  ): Promise<unknown> {
    const { createAgent } = await importLangChain();
    const middleware = await this.buildMiddleware();
    const params: Record<string, unknown> = {
      model: chat,
      tools: sortedToolsForCache(tools),
      middleware,
    };
    if (system !== undefined) params['systemPrompt'] = system;
    try {
      return createAgent(params);
    } catch (exc) {
      // Only retry without ``systemPrompt`` when the error clearly signals
      // an unknown-keyword failure from an older LangChain. Otherwise
      // re-throw so we don't mask unrelated bugs (auth, validation, etc.).
      const message = exc instanceof Error ? exc.message : String(exc);
      const isUnknownKwarg = /unexpected keyword|unknown (?:argument|kwarg|option)|not a known argument|systemPrompt/i.test(
        message
      );
      if (!isUnknownKwarg || system === undefined) {
        throw exc;
      }
      delete params['systemPrompt'];
      return createAgent(params);
    }
  }

  /**
   * Pull every option whose key lives in :data:`LLM_BIND_OPTIONS` and return a
   * fresh object — the caller passes it to ``chat.bind(...)``. Non-LLM keys
   * are silently ignored, which keeps the public ``RunOptions`` interface
   * forgiving when callers mix in custom config.
   *
   * Gemini quirk: LangChain.js's ``ChatGoogleGenerativeAI`` uses
   * ``maxOutputTokens`` instead of ``maxTokens``. We alias when the caller
   * provided ``maxTokens`` without an explicit ``maxOutputTokens``.
   */
  private extractBindOptions(
    options: RunOptions
  ): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(options)) {
      if (value === undefined) continue;
      if (LLM_BIND_OPTIONS.has(key)) {
        out[key] = value;
      }
    }
    if (
      this.provider === 'google_genai' &&
      out['maxTokens'] !== undefined &&
      out['maxOutputTokens'] === undefined
    ) {
      const { maxTokens, ...rest } = out;
      return { ...rest, maxOutputTokens: maxTokens };
    }
    return out;
  }

  async run(options: RunOptions): Promise<NormalizedResult> {
    const effectiveModel = options.model ?? this.defaultModelName;
    if (effectiveModel === undefined) {
      throw new CompresrError(
        "model is required — set it on the client " +
          "(llm: 'anthropic:claude-...') or pass it to messages.create({ model: '...' }).",
        'missing_model'
      );
    }
    const tools = options.tools ?? [];
    const bindOptions = this.extractBindOptions(options);
    // Auto-route: solo WebSearchTool -> deep-research loop, with the caller's
    // facade shape preserved by feeding the message chain through ``normalize``.
    if (isSoloWebSearch(tools)) {
      return this.runViaResearchLoop({
        messages: options.messages,
        searchTool: tools[0],
        system: options.system,
        model: effectiveModel,
        bindKwargs: Object.keys(bindOptions).length > 0 ? bindOptions : undefined,
      });
    }
    // Bake LLM knobs into the chat-model constructor (not chat.bind(...))
    // so they survive create_agent's internal bind_tools call. Per-knob
    // combinations are cached separately.
    const chat = await this.getChatModel(
      effectiveModel,
      Object.keys(bindOptions).length > 0 ? bindOptions : undefined
    );
    const agent = (await this.buildAgent(
      chat,
      tools,
      options.system
    )) as {
      invoke: (state: Record<string, unknown>, config?: unknown) => unknown;
    };
    let result: unknown;
    try {
      result = await Promise.resolve(
        agent.invoke(
          { messages: [...options.messages] },
          options.config ?? {}
        )
      );
    } catch (exc) {
      // Re-wrap LangChain failures so callers always see a CompresrError
      // (parity with Python M1). The original error is preserved via
      // ``cause`` for debugging.
      if (exc instanceof CompresrError) throw exc;
      const message = exc instanceof Error ? exc.message : String(exc);
      const wrapped = new CompresrError(
        `Agent execution failed: ${message}`,
        'agent_execution_failed'
      );
      // Native ``Error.cause`` is part of ES2022 — preserve the original
      // for debugging without losing CompresrError typing.
      (wrapped as Error & { cause?: unknown }).cause = exc;
      throw wrapped;
    }
    return this.normalize(result);
  }

  /**
   * Auto-route handler for solo-WebSearchTool calls.
   *
   * Constructs a {@link ResearchAgent}, runs its strict-output loop against
   * the last user message, then feeds the resulting message chain back through
   * {@link normalize} so each facade still gets its expected shape.
   *
   * The caller's ``system`` wins when supplied as a non-empty string; otherwise
   * the research-agent default prompt is used.
   */
  private async runViaResearchLoop(params: {
    messages: ReadonlyArray<unknown>;
    searchTool: unknown;
    system: string | undefined;
    model: string;
    bindKwargs: Record<string, unknown> | undefined;
  }): Promise<NormalizedResult> {
    const { ResearchAgent } = await import('./research/agent.js');
    const question = extractLastUserText(params.messages);
    const systemPrompt =
      typeof params.system === 'string' && params.system.trim().length > 0
        ? params.system
        : undefined;
    // ``ResearchEngine`` is a structural-only contract over the engine — the
    // real ``CompressionClient.data`` is nullable while ``ResearchEngine``'s
    // ``CompresrClientLike`` treats it as optional. The two shapes are
    // functionally compatible (the research agent never observes ``null``
    // separately from ``undefined``); the ``unknown`` bridge is the cleanest
    // way to express that without weakening either contract.
    const engineForLoop: ResearchEngine = params.bindKwargs
      ? ({
          provider: this.provider,
          ...(this.defaultModelName !== undefined
            ? { defaultModelName: this.defaultModelName }
            : {}),
          _resolveModel: (m?: string) => this._resolveModel(m),
          _getChatModel: (modelName: string) =>
            this.getChatModel(modelName, params.bindKwargs) as Promise<ChatModelLike>,
          _compresrClient: this._compresrClient,
          _promptCacheConfig: this._promptCacheConfig,
        } as unknown as ResearchEngine)
      : (this as unknown as ResearchEngine);
    const agent = new ResearchAgent({
      engine: engineForLoop,
      searchTool: params.searchTool as SearchToolLike,
      ...(systemPrompt !== undefined ? { systemPrompt } : {}),
    });
    const state = await agent._runLoop(question, { model: params.model });
    return this.normalize({ messages: state.messages });
  }

  private normalize(agentResult: unknown): NormalizedResult {
    const messages =
      agentResult && typeof agentResult === 'object'
        ? ((agentResult as { messages?: unknown[] }).messages ?? [])
        : [];
    const msg = lastAiMessage(messages);
    if (msg == null) {
      return {
        text: '',
        contentBlocks: [],
        toolUses: [],
        citations: [],
        stopReason: 'end_turn',
        usage: {},
        compresrStats: defaultCompresrStats(),
        raw: null,
        messages: [...messages],
      };
    }
    return {
      text: contentText(msg),
      contentBlocks: normalizeContentBlocks(msg),
      toolUses: normalizeToolUses(msg),
      citations: extractCitations(msg),
      stopReason: normalizeStopReason(msg),
      usage: aggregateUsage(messages),
      compresrStats: defaultCompresrStats(),
      raw: msg,
      messages: [...messages],
    };
  }
}
