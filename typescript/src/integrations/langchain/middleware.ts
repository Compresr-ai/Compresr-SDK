/**
 * Agent middleware for LangChain.js 1.0+ `createAgent`.
 *
 * - `compresrToolMiddleware` — compresses each tool output as it returns,
 *   before it enters agent state.
 * - `compresrSummarizationMiddleware` — when state grows past a token
 *   threshold, compresses the older block into a single summary message
 *   and replaces it in state. Mirrors LangChain's `SummarizationMiddleware`
 *   but uses Compresr instead of an LLM call. KV-cache-friendly:
 *   the summary is generated once at trigger time and persists in state.
 *
 * Unified Compresr query API across both:
 *
 *     query: string                                 // static query
 *     queryExtractor: (...) => string               // custom extractor
 *     queryArg: string                              // name of the tool arg
 */
import {
  HumanMessage,
  RemoveMessage,
  ToolMessage,
  type BaseMessage,
} from '@langchain/core/messages';

import type { CompressionClient } from '../../clients/compression.js';
import {
  buildClient,
  compressSafe,
  DEFAULT_MIN_TOKENS,
  DEFAULT_MODEL,
  DEFAULT_POLICY,
  DEFAULT_RATIO,
  type ErrorPolicy,
  estimateTokens,
  makeFilter,
  resolveQuery,
} from '../_shared/index.js';

const REMOVE_ALL_MESSAGES = '__remove_all__';

type ToolMessageType = ToolMessage;

export interface ToolCall {
  id?: string;
  name?: string;
  args?: Record<string, unknown>;
}

export interface ToolCallRequest {
  toolCall: ToolCall;
  messages?: unknown[];
}

export type ToolHandler = (
  request: ToolCallRequest
) => Promise<ToolMessageType> | ToolMessageType;

export type ToolQueryExtractor = (
  toolCall: ToolCall,
  messages: unknown[]
) => string | undefined;

export type SummaryQueryExtractor = (messages: unknown[]) => string | undefined;

export interface CompresrMiddlewareCommonOptions {
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  targetCompressionRatio?: number;
  coarse?: boolean;
  onError?: ErrorPolicy;
}

export interface CompresrToolMiddlewareOptions
  extends CompresrMiddlewareCommonOptions {
  minTokens?: number;
  allowTools?: Iterable<string>;
  ignoreTools?: Iterable<string>;
  query?: string;
  queryExtractor?: ToolQueryExtractor;
  queryArg?: string;
}

export interface CompresrSummarizationMiddlewareOptions
  extends CompresrMiddlewareCommonOptions {
  /** Compression fires when the message-state token count crosses this. */
  maxTokensBeforeSummary?: number;
  /** Number of recent messages to preserve verbatim. */
  messagesToKeep?: number;
  /** Alias for `maxTokensBeforeSummary` — matches LangChain.js `SummarizationMiddleware`. */
  trigger?: number;
  /** Alias for `messagesToKeep` — matches LangChain.js `SummarizationMiddleware`. */
  keep?: number;
  /** Custom token counter; defaults to the chars/4 estimator. */
  tokenCounter?: (s: string) => number;
  query?: string;
  queryExtractor?: SummaryQueryExtractor;
}

export interface ModelRequest {
  messages: unknown[];
  [k: string]: unknown;
}

export type ModelHandler<T = unknown> = (request: ModelRequest) => Promise<T> | T;

export interface CompresrMiddleware {
  name: string;
  wrapToolCall?: (
    request: ToolCallRequest,
    handler: ToolHandler
  ) => Promise<ToolMessageType>;
  beforeModel?: (state: {
    messages?: unknown[];
  }) => Promise<{ messages?: unknown[]; llm_input_messages?: unknown[] } | undefined>;
  wrapModelCall?: <T = unknown>(
    request: ModelRequest,
    handler: ModelHandler<T>
  ) => Promise<T>;
}

function isToolMessage(value: unknown): value is ToolMessageType {
  return value instanceof ToolMessage;
}

function buildToolMessage(content: string, original: ToolMessageType): ToolMessageType {
  const orig = original as { tool_call_id?: string; name?: string };
  return new ToolMessage({
    content,
    tool_call_id: orig.tool_call_id ?? '',
    ...(orig.name !== undefined ? { name: orig.name } : {}),
  });
}

function isEligible(
  result: unknown,
  toolCall: ToolCall,
  eligible: (name?: string | null) => boolean
): result is ToolMessageType {
  if (!isToolMessage(result)) return false;
  const content = (result as { content?: unknown }).content;
  if (typeof content !== 'string') return false;
  const name = toolCall.name ?? (result as { name?: string }).name;
  return eligible(name);
}

export function compresrToolMiddleware(
  options: CompresrToolMiddlewareOptions
): CompresrMiddleware {
  const {
    apiKey,
    client,
    baseUrl,
    compressionModel = DEFAULT_MODEL,
    targetCompressionRatio = DEFAULT_RATIO,
    minTokens = DEFAULT_MIN_TOKENS,
    coarse,
    allowTools,
    ignoreTools,
    onError = DEFAULT_POLICY,
    query,
    queryExtractor,
    queryArg,
  } = options;

  const eligible = makeFilter({ allow: allowTools, ignore: ignoreTools });
  const compresr =
    client ?? buildClient({ apiKey, baseUrl, caller: 'compresrToolMiddleware' });

  function resolveTheQuery(toolCall: ToolCall, messages: unknown[]): string | undefined {
    return resolveQuery({
      staticQuery: query,
      extractor: queryExtractor
        ? (arg) => queryExtractor(...(arg as [ToolCall, unknown[]]))
        : undefined,
      extractorArg: queryExtractor ? [toolCall, messages] : undefined,
      args: toolCall.args ?? null,
      argsKey: queryArg,
      messages,
      toolCallId: toolCall.id,
    });
  }

  return {
    name: 'compresr-tool',
    async wrapToolCall(request, handler) {
      const result = await handler(request);
      const toolCall = request.toolCall ?? {};
      if (!isEligible(result, toolCall, eligible)) {
        return result;
      }
      const messages = request.messages ?? [];
      const resolvedQuery = resolveTheQuery(toolCall, messages);
      const compressed = await compressSafe(compresr, {
        context: (result as { content: string }).content,
        ...(resolvedQuery !== undefined ? { query: resolvedQuery } : {}),
        compressionModel,
        targetCompressionRatio,
        ...(coarse !== undefined ? { coarse } : {}),
        minTokens,
        onError,
        contextLabel: `tool:${(result as { name?: string }).name ?? '?'}`,
      });
      return compressed !== (result as { content: string }).content
        ? buildToolMessage(compressed, result)
        : result;
    },
  };
}

const SUMMARY_PREFIX = '[Earlier conversation summary]\n\n';

function msgToText(m: unknown): string {
  const ctor = (m as { constructor?: { name?: string } }).constructor?.name ?? 'Msg';
  const role = ctor.replace('Message', '').toLowerCase() || 'msg';
  const name = (m as { name?: string }).name;
  const label = name ? `${role}:${name}` : role;
  const c = (m as { content?: unknown }).content;
  const text = typeof c === 'string' ? c : JSON.stringify(c);
  return `${label}: ${text}`;
}

function isSummaryMessage(m: unknown): boolean {
  if (!(m instanceof HumanMessage)) return false;
  const c = (m as { content?: unknown }).content;
  return typeof c === 'string' && c.startsWith(SUMMARY_PREFIX);
}

export function compresrSummarizationMiddleware(
  options: CompresrSummarizationMiddlewareOptions
): CompresrMiddleware {
  const {
    apiKey,
    client,
    baseUrl,
    compressionModel = DEFAULT_MODEL,
    targetCompressionRatio = DEFAULT_RATIO,
    coarse,
    onError = DEFAULT_POLICY,
    maxTokensBeforeSummary = 4_000,
    messagesToKeep = 20,
    trigger,
    keep: keepAlias,
    tokenCounter,
    query,
    queryExtractor,
  } = options;

  const compresr =
    client ?? buildClient({ apiKey, baseUrl, caller: 'compresrSummarizationMiddleware' });

  const maxTokens = Math.max(1, trigger ?? maxTokensBeforeSummary);
  const keep = Math.max(1, keepAlias ?? messagesToKeep);
  const countTokens = tokenCounter ?? estimateTokens;

  function resolveTheQuery(messages: unknown[]): string | undefined {
    return resolveQuery({
      staticQuery: query,
      extractor: queryExtractor
        ? (arg) => queryExtractor(arg as unknown[])
        : undefined,
      extractorArg: queryExtractor ? messages : undefined,
      messages,
    });
  }

  return {
    name: 'compresr-summarization',
    async beforeModel(state) {
      const messages = [...(state.messages ?? [])];
      if (messages.length <= keep) return undefined;

      let total = 0;
      for (const m of messages) {
        const c = (m as { content?: unknown }).content;
        if (typeof c === 'string') total += countTokens(c);
      }
      if (total < maxTokens) return undefined;

      const toSummarize = messages.slice(0, messages.length - keep);
      const recent = messages.slice(messages.length - keep);
      if (toSummarize.length === 0) return undefined;

      let joined: string;
      if (toSummarize.length > 0 && isSummaryMessage(toSummarize[0])) {
        const prev = ((toSummarize[0] as BaseMessage).content as string).slice(
          SUMMARY_PREFIX.length
        );
        const extra = toSummarize.slice(1).map(msgToText).join('\n');
        joined = prev + '\n\n' + extra;
      } else {
        joined = toSummarize.map(msgToText).join('\n');
      }

      const compressed = await compressSafe(compresr, {
        context: joined,
        ...(resolveTheQuery(messages) !== undefined
          ? { query: resolveTheQuery(messages) }
          : {}),
        compressionModel,
        targetCompressionRatio,
        ...(coarse !== undefined ? { coarse } : {}),
        minTokens: 1,
        onError,
        contextLabel: 'summary',
      });
      if (compressed === joined) return undefined;

      return {
        messages: [
          new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
          new HumanMessage({ content: `${SUMMARY_PREFIX}${compressed}` }),
          ...recent,
        ],
      };
    },
  };
}

export interface CompresrPromptMiddlewareOptions extends CompresrMiddlewareCommonOptions {
  /** Hard ceiling for the outbound prompt (sum of message content tokens). */
  maxTokens: number;
  /** Skip compression on individual messages shorter than this. */
  minTokens?: number;
  tokenCounter?: (s: string) => number;
  query?: string;
  queryExtractor?: SummaryQueryExtractor;
}

function rebuildWithContent(msg: unknown, content: string): unknown {
  if (msg === null || typeof msg !== 'object') return msg;
  const ctor = (msg as { constructor?: new (init: Record<string, unknown>) => unknown }).constructor;
  if (typeof ctor !== 'function') return msg;
  const init: Record<string, unknown> = { content };
  for (const attr of ['tool_call_id', 'name', 'tool_calls', 'id'] as const) {
    const v = (msg as Record<string, unknown>)[attr];
    if (v !== undefined && v !== null) init[attr] = v;
  }
  try {
    return new ctor(init);
  } catch {
    // Some message classes need positional or different shape — fall back.
    return msg;
  }
}

export function compresrPromptMiddleware(
  options: CompresrPromptMiddlewareOptions
): CompresrMiddleware {
  const {
    apiKey,
    client,
    baseUrl,
    compressionModel = DEFAULT_MODEL,
    coarse,
    onError = DEFAULT_POLICY,
    maxTokens,
    minTokens = DEFAULT_MIN_TOKENS,
    tokenCounter,
    query,
    queryExtractor,
  } = options;

  const compresr =
    client ?? buildClient({ apiKey, baseUrl, caller: 'compresrPromptMiddleware' });
  const budget = Math.max(1, maxTokens);
  const floor = Math.max(1, minTokens);
  const countTokens = tokenCounter ?? estimateTokens;

  function totalTokens(messages: unknown[]): number {
    let t = 0;
    for (const m of messages) {
      const c = (m as { content?: unknown }).content;
      if (typeof c === 'string') t += countTokens(c);
    }
    return t;
  }

  function resolveTheQuery(messages: unknown[]): string | undefined {
    return resolveQuery({
      staticQuery: query,
      extractor: queryExtractor ? (arg) => queryExtractor(arg as unknown[]) : undefined,
      extractorArg: queryExtractor ? messages : undefined,
      messages,
    });
  }

  async function shrink(messages: unknown[]): Promise<unknown[]> {
    if (totalTokens(messages) <= budget) return messages;

    const candidates: { idx: number; msg: unknown; tokens: number }[] = [];
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      const c = (m as { content?: unknown }).content;
      if (typeof c === 'string') {
        const t = countTokens(c);
        if (t >= floor) candidates.push({ idx: i, msg: m, tokens: t });
      }
    }
    candidates.sort((a, b) => b.tokens - a.tokens);

    const resolvedQuery = resolveTheQuery(messages);
    const next = [...messages];
    for (const { idx, msg, tokens: current } of candidates) {
      const totalNow = totalTokens(next);
      if (totalNow <= budget) break;
      const overshoot = totalNow - budget;
      const target = Math.max(floor, current - Math.max(1, overshoot));
      const ratio = Math.max(current / Math.max(1, target), 1.0);
      const original = (msg as { content: string }).content;
      const compressed = await compressSafe(compresr, {
        context: original,
        ...(resolvedQuery !== undefined ? { query: resolvedQuery } : {}),
        compressionModel,
        targetCompressionRatio: ratio,
        ...(coarse !== undefined ? { coarse } : {}),
        minTokens: floor,
        onError,
        contextLabel: 'prompt_budget',
      });
      if (compressed !== original) {
        next[idx] = rebuildWithContent(msg, compressed);
      }
    }
    return next;
  }

  return {
    name: 'compresr-prompt',
    async wrapModelCall(request, handler) {
      let next = request;
      try {
        const shrunk = await shrink(request.messages);
        if (shrunk !== request.messages) {
          // Parity with Python PY-AM4 — never mutate the caller's request.
          next = { ...request, messages: shrunk };
        }
      } catch {
        // Fail open — send the original prompt rather than block the call.
      }
      return handler(next);
    },
  };
}
