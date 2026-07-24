/**
 * Tool-output wrappers for LangChain.js.
 *
 * Mirrors Python `wrap_tool_with_compression`. Works without `createAgent`
 * middleware — wrap any tool and call it directly in LCEL chains or
 * LangGraph custom graphs.
 *
 * For `latte_v1` the unified Compresr query API applies (see middleware).
 */
import { tool as toolFactory, type StructuredTool } from '@langchain/core/tools';

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

type StructuredToolType = StructuredTool;

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

interface WrappableTool {
  name: string;
  description?: string;
  schema?: unknown;
  func?: (args: Record<string, unknown>) => unknown;
  _call?: (args: Record<string, unknown>) => unknown;
  invoke?: (args: Record<string, unknown>) => Promise<unknown>;
}

function isWrappableTool(t: unknown): t is WrappableTool {
  if (typeof t !== 'object' || t === null) return false;
  const x = t as WrappableTool;
  return (
    typeof x.name === 'string' &&
    (typeof x.func === 'function' ||
      typeof x._call === 'function' ||
      typeof x.invoke === 'function')
  );
}

/**
 * Wrap a structured tool so its string output is compressed transparently.
 *
 * The wrapped tool preserves the original `name`, `description`, and schema.
 * Accepts any tool that exposes `name` + (`func` | `_call` | `invoke`) — i.e.
 * what the `tool()` factory and `StructuredTool` subclasses produce.
 */
export function wrapToolWithCompression(
  tool: StructuredToolType,
  options: WrapToolOptions = {}
): StructuredToolType {
  if (!isWrappableTool(tool)) {
    throw new TypeError(
      'wrapToolWithCompression supports structured tools (with `name` + ' +
        '`func`/`_call`/`invoke`). Use the `tool()` factory from @langchain/core.'
    );
  }

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
    client ?? buildClient({ apiKey, baseUrl, caller: 'wrapToolWithCompression' });

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

  const t = tool as WrappableTool;
  const innerCall = pickCallable(t, tool);

  const wrappedFunc = async (args: Record<string, unknown>): Promise<unknown> => {
    const out = await innerCall(args);
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
      contextLabel: `tool:${t.name}`,
    });
  };

  const cfg: Record<string, unknown> = { name: t.name };
  if (t.description !== undefined) cfg.description = t.description;
  if (t.schema !== undefined) cfg.schema = t.schema;
  return (toolFactory as unknown as (fn: typeof wrappedFunc, cfg: Record<string, unknown>) => StructuredToolType)(
    wrappedFunc,
    cfg
  );
}

function pickCallable(
  meta: WrappableTool,
  tool: unknown
): (args: Record<string, unknown>) => unknown {
  if (typeof meta.func === 'function') return meta.func.bind(tool);
  if (typeof meta._call === 'function') return meta._call.bind(tool);
  if (typeof meta.invoke === 'function') return meta.invoke.bind(tool);
  throw new TypeError(`Tool '${meta.name}' has no callable surface.`);
}

export type ToolDecorator = (tool: StructuredToolType) => StructuredToolType;

/** Decorator-style helper — for users who prefer function composition. */
export function compressToolOutput(options: WrapToolOptions = {}): ToolDecorator {
  return (tool) => wrapToolWithCompression(tool, options);
}
