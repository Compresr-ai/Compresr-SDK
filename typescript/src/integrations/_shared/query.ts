/**
 * Extract a query string for `latte_v1` query-aware compression.
 *
 * Three resolution layers, in priority order:
 *   1. Static (`query: string`).
 *   2. Custom extractor (`queryExtractor: (ctx) => string`).
 *   3. Smart default — pick from common arg keys, then last user message,
 *      then a fallback string.
 *
 * Mirrors Python `_shared/query.py`.
 */

export const DEFAULT_FALLBACK = 'summarize';

/** Common tool-arg keys that typically carry the user/LLM intent. */
export const COMMON_QUERY_KEYS: readonly string[] = [
  'query',
  'question',
  'search_query',
  'q',
  'prompt',
  'input',
  'text',
];


interface MessageLike {
  role?: string;
  content?: unknown;
  tool_calls?: ToolCallLike[];
  toolCalls?: ToolCallLike[];
}

interface ToolCallLike {
  id?: string;
  args?: Record<string, unknown>;
}

function isAiMessage(msg: unknown): msg is MessageLike {
  if (typeof msg !== 'object' || msg === null) return false;
  const m = msg as MessageLike;
  const name = (msg as { constructor?: { name?: string } }).constructor?.name;
  if (name === 'AIMessage' || name === 'AIMessageChunk') return true;
  return m.role === 'assistant' || m.role === 'ai';
}

function isHumanMessage(msg: unknown): msg is MessageLike {
  if (typeof msg !== 'object' || msg === null) return false;
  const m = msg as MessageLike;
  const name = (msg as { constructor?: { name?: string } }).constructor?.name;
  if (name === 'HumanMessage' || name === 'HumanMessageChunk') return true;
  return m.role === 'user' || m.role === 'human';
}

function getToolCalls(msg: MessageLike): ToolCallLike[] {
  return msg.tool_calls ?? msg.toolCalls ?? [];
}

function getStringContent(msg: MessageLike): string | undefined {
  const c = msg.content;
  return typeof c === 'string' && c.trim() ? c : undefined;
}

export interface ExtractFromMessagesOptions {
  toolCallId?: string;
  fallback?: string;
}

export function extractQueryFromMessages(
  messages: readonly unknown[],
  options: ExtractFromMessagesOptions = {}
): string {
  const { toolCallId, fallback = DEFAULT_FALLBACK } = options;

  if (toolCallId !== undefined) {
    for (let i = messages.length - 1; i >= 0; i--) {
      const prev = messages[i];
      if (!isAiMessage(prev)) continue;
      for (const call of getToolCalls(prev)) {
        if (call.id === toolCallId) {
          const q = extractQueryFromArgs(call.args ?? {});
          if (q) return q;
        }
      }
    }
  }

  for (let i = messages.length - 1; i >= 0; i--) {
    const prev = messages[i];
    if (!isHumanMessage(prev)) continue;
    const c = getStringContent(prev);
    if (c) return c;
  }

  return fallback;
}


export interface ExtractFromArgsOptions {
  preferredKey?: string;
  candidates?: readonly string[];
}

export function extractQueryFromArgs(
  args: Record<string, unknown> | null | undefined,
  options: ExtractFromArgsOptions = {}
): string | undefined {
  if (!args || typeof args !== 'object' || Object.keys(args).length === 0) {
    return undefined;
  }
  const { preferredKey, candidates = COMMON_QUERY_KEYS } = options;

  if (preferredKey !== undefined) {
    const v = args[preferredKey];
    if (typeof v === 'string' && v.trim()) return v;
    return undefined; // user named a key; don't silently fall back
  }

  for (const key of candidates) {
    const v = args[key];
    if (typeof v === 'string' && v.trim()) return v;
  }

  return undefined;
}


export interface ResolveQueryOptions {
  staticQuery?: string;
  extractor?: (arg: unknown) => string | undefined | null;
  extractorArg?: unknown;
  args?: Record<string, unknown> | null;
  argsKey?: string;
  messages?: readonly unknown[];
  toolCallId?: string;
  fallback?: string;
}

const MISSING = Symbol('MISSING');

/**
 * Resolve a query string with consistent priority across integrations.
 *
 * Priority (first non-empty wins):
 *   1. `staticQuery`
 *   2. `extractor(extractorArg)` — if `extractorArg` is the MISSING
 *       sentinel (not passed), the extractor is called with no args.
 *   3. `args[argsKey]`
 *   4. `args` smart-picked via COMMON_QUERY_KEYS
 *   5. Most recent human message in `messages`
 *   6. `fallback` (default `"summarize"`)
 */
export function resolveQuery(options: ResolveQueryOptions = {}): string {
  const {
    staticQuery,
    extractor,
    extractorArg = MISSING,
    args,
    argsKey,
    messages,
    toolCallId,
    fallback = DEFAULT_FALLBACK,
  } = options;

  if (typeof staticQuery === 'string' && staticQuery.trim()) {
    return staticQuery;
  }

  if (extractor !== undefined) {
    let value: string | undefined | null;
    try {
      value =
        extractorArg === MISSING
          ? (extractor as () => string | undefined | null)()
          : extractor(extractorArg);
    } catch {
      value = undefined;
    }
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }

  if (args !== undefined && args !== null) {
    const v = extractQueryFromArgs(args, { preferredKey: argsKey });
    if (v) return v;
  }

  if (messages && messages.length > 0) {
    return extractQueryFromMessages(messages, { toolCallId, fallback });
  }

  return fallback;
}
