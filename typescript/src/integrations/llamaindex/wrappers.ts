/**
 * Wrap a LlamaIndex.TS `FunctionTool` (or any object with a `call` method)
 * so its return value is compressed.
 *
 * Mirrors Python `wrap_tool_with_compresr`. Same unified query API.
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

export interface WrapToolOptions {
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  targetCompressionRatio?: number;
  minTokens?: number;
  coarse?: boolean;
  query?: string;
  queryExtractor?: (args: Record<string, unknown>) => string | undefined;
  queryArg?: string;
  onError?: ErrorPolicy;
}

/**
 * The narrow callable shape we wrap. `FunctionTool` in llamaindex.ts
 * exposes `metadata` + `call(args)` / `acall(args)` — we wrap that.
 */
export interface CompresrWrappableTool<TArgs extends Record<string, unknown> = Record<string, unknown>> {
  metadata: { name: string; description?: string };
  call: (args: TArgs) => unknown;
}

/**
 * Returns a new tool with the same `metadata` and a wrapped `call` that
 * compresses string return values transparently.
 */
export function wrapToolWithCompresr<
  TArgs extends Record<string, unknown>,
  T extends CompresrWrappableTool<TArgs>,
>(tool: T, options: WrapToolOptions = {}): T {
  const {
    apiKey,
    client,
    baseUrl,
    compressionModel = DEFAULT_MODEL,
    targetCompressionRatio = DEFAULT_RATIO,
    minTokens = DEFAULT_MIN_TOKENS,
    coarse,
    query,
    queryExtractor,
    queryArg,
    onError = DEFAULT_POLICY,
  } = options;

  const compresr =
    client ?? buildClient({ apiKey, baseUrl, caller: 'wrapToolWithCompresr' });

  function resolveTheQuery(args: Record<string, unknown>): string | undefined {
    return resolveQuery({
      staticQuery: query,
      extractor: queryExtractor
        ? (arg) => queryExtractor(arg as Record<string, unknown>)
        : undefined,
      extractorArg: queryExtractor ? args : undefined,
      args,
      argsKey: queryArg,
    });
  }

  const originalCall = tool.call.bind(tool);

  const wrapped = {
    ...(tool as object),
    metadata: tool.metadata,
    async call(args: TArgs): Promise<unknown> {
      const out = await originalCall(args);
      if (typeof out !== 'string') return out;
      const q = resolveTheQuery(args);
      return compressSafe(compresr, {
        context: out,
        ...(q !== undefined ? { query: q } : {}),
        compressionModel,
        targetCompressionRatio,
        ...(coarse !== undefined ? { coarse } : {}),
        minTokens,
        onError,
        contextLabel: `tool:${tool.metadata.name}`,
      });
    },
  } as T;

  return wrapped;
}
