/**
 * Drop-in compression node for a custom `StateGraph`.
 *
 * Mirrors Python `make_compresr_node`. Reads a string field from state,
 * compresses it, writes back. Same unified query API.
 */
import type { CompressionClient } from '../../clients/compression.js';
import {
  buildClient,
  compressSafe,
  DEFAULT_MIN_TOKENS,
  DEFAULT_MODEL,
  DEFAULT_POLICY,
  DEFAULT_RATIO,
  type ErrorPolicy,
  resolveQuery,
} from '../_shared/index.js';

export interface MakeCompresrNodeOptions<S extends Record<string, unknown>> {
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  /** State key holding the text to compress. Required. */
  contextKey: keyof S & string;
  compressionModel?: string;
  targetCompressionRatio?: number;
  minTokens?: number;
  coarse?: boolean;
  /** Static query (latte_v1 only). */
  query?: string;
  /** State key holding the query string. */
  queryKey?: keyof S & string;
  /** Custom extractor over state. Highest priority after `query`. */
  queryExtractor?: (state: S) => string | undefined;
  onError?: ErrorPolicy;
}

/**
 * Build a state-graph node that compresses `state[contextKey]`.
 *
 * Returns an async function `(state) => Partial<State>` compatible with
 * `graph.addNode("compress", node)` in LangGraph.js.
 */
export function makeCompresrNode<S extends Record<string, unknown>>(
  options: MakeCompresrNodeOptions<S>
): (state: S) => Promise<Partial<S>> {
  const {
    apiKey,
    client,
    baseUrl,
    contextKey,
    compressionModel = DEFAULT_MODEL,
    targetCompressionRatio = DEFAULT_RATIO,
    minTokens = DEFAULT_MIN_TOKENS,
    coarse,
    query,
    queryKey,
    queryExtractor,
    onError = DEFAULT_POLICY,
  } = options;

  const compresr =
    client ?? buildClient({ apiKey, baseUrl, caller: 'makeCompresrNode' });

  function resolve(state: S): string | undefined {
    let argsDict: Record<string, unknown> | null = null;
    if (queryKey !== undefined) {
      const v = state[queryKey];
      if (typeof v === 'string' && v.trim()) {
        argsDict = { [queryKey]: v };
      }
    }
    return resolveQuery({
      staticQuery: query,
      extractor: queryExtractor
        ? (arg) => queryExtractor(arg as S)
        : undefined,
      extractorArg: queryExtractor ? state : undefined,
      args: argsDict,
      argsKey: queryKey,
    });
  }

  return async (state: S) => {
    const ctx = state[contextKey];
    if (typeof ctx !== 'string' || !ctx.trim()) {
      return {};
    }
    const q = resolve(state);
    const compressed = await compressSafe(compresr, {
      context: ctx,
      ...(q !== undefined ? { query: q } : {}),
      compressionModel,
      targetCompressionRatio,
      ...(coarse !== undefined ? { coarse } : {}),
      minTokens,
      onError,
      contextLabel: `node:${String(contextKey)}`,
    });
    if (compressed === ctx) {
      return {};
    }
    return { [contextKey]: compressed } as Partial<S>;
  };
}
