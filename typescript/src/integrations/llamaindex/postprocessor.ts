/**
 * `BaseNodePostprocessor` that compresses retrieved node content.
 *
 * Mirrors Python `CompresrNodePostprocessor`. Query comes from the
 * `QueryBundle` supplied by the query engine; override via `query`.
 */
import type { CompressionClient } from '../../clients/compression.js';
import { getLogger } from '../../logger.js';
import {
  BATCH_LIMIT,
  buildClient,
  DEFAULT_MIN_TOKENS,
  DEFAULT_MODEL,
  DEFAULT_POLICY,
  DEFAULT_RATIO,
  type ErrorPolicy,
  estimateTokens,
} from '../_shared/index.js';

import type { NodeWithScore } from '@llamaindex/core/schema';

type NodeWithScoreType = NodeWithScore;

/**
 * LlamaIndex.TS query input — either the raw query string or an object
 * carrying the query string (matching LlamaIndex.TS' `MessageContent`
 * conventions plus Python-side `QueryBundle.query_str`).
 */
type QueryInput =
  | string
  | { queryStr?: string; query_str?: string }
  | undefined;

export interface CompresrNodePostprocessorOptions {
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  targetCompressionRatio?: number;
  /**
   * Absolute output token budget per node. When set, overrides
   * `targetCompressionRatio` with `ratio = avg_chunk_tokens / targetToken`.
   * Approximate (chars/4 estimator).
   */
  targetToken?: number;
  minTokens?: number;
  coarse?: boolean;
  onError?: ErrorPolicy;
  /** Override the query supplied by the query engine. */
  query?: string;
}

/**
 * @example
 * ```ts
 * import { CompresrNodePostprocessor } from '@compresr/sdk/integrations/llamaindex';
 * const pp = new CompresrNodePostprocessor({
 *   apiKey: process.env.COMPRESR_API_KEY!,
 *   compressionModel: 'latte_v1',
 * });
 * const queryEngine = index.asQueryEngine({ nodePostprocessors: [pp] });
 * ```
 */
export class CompresrNodePostprocessor {
  private readonly compresr: CompressionClient;
  private readonly compressionModel: string;
  private readonly targetCompressionRatio: number;
  private readonly targetToken: number | undefined;
  private readonly minTokens: number;
  private readonly coarse: boolean | undefined;
  private readonly onError: ErrorPolicy;
  private readonly staticQuery: string | undefined;

  constructor(options: CompresrNodePostprocessorOptions) {
    const {
      apiKey,
      client,
      baseUrl,
      compressionModel = DEFAULT_MODEL,
      targetCompressionRatio = DEFAULT_RATIO,
      targetToken,
      minTokens = DEFAULT_MIN_TOKENS,
      coarse,
      onError = DEFAULT_POLICY,
      query,
    } = options;

    this.compresr =
      client ?? buildClient({ apiKey, baseUrl, caller: 'CompresrNodePostprocessor' });
    this.compressionModel = compressionModel;
    this.targetCompressionRatio = targetCompressionRatio;
    this.targetToken = targetToken;
    this.minTokens = minTokens;
    this.coarse = coarse;
    this.onError = onError;
    this.staticQuery = query;
  }

  private effectiveRatio(chunk: readonly string[]): number {
    if (!this.targetToken || this.targetToken <= 0) return this.targetCompressionRatio;
    const avg = Math.max(
      1,
      chunk.reduce((acc, c) => acc + estimateTokens(c), 0) / Math.max(1, chunk.length)
    );
    return Math.max(avg / this.targetToken, 1.0);
  }

  private resolveQuery(queryBundle?: QueryInput | string): string | undefined {
    if (this.compressionModel !== 'latte_v1') return undefined;
    if (typeof this.staticQuery === 'string' && this.staticQuery.trim()) {
      return this.staticQuery;
    }
    if (typeof queryBundle === 'string' && queryBundle.trim()) {
      return queryBundle;
    }
    if (queryBundle && typeof queryBundle === 'object') {
      const q = (queryBundle).queryStr ??
        (queryBundle as { query_str?: string }).query_str;
      if (typeof q === 'string' && q.trim()) return q;
    }
    return undefined;
  }

  private partition(
    nodes: readonly NodeWithScoreType[]
  ): { indices: number[]; contents: string[] } {
    const indices: number[] = [];
    const contents: string[] = [];
    nodes.forEach((n, i) => {
      const text = getNodeText(n);
      if (typeof text === 'string' && estimateTokens(text) >= this.minTokens) {
        indices.push(i);
        contents.push(text);
      }
    });
    return { indices, contents };
  }

  /** LlamaIndex.TS `BaseNodePostprocessor.postprocessNodes` signature. */
  async postprocessNodes(
    nodes: readonly NodeWithScoreType[],
    queryBundle?: QueryInput | string
  ): Promise<NodeWithScoreType[]> {
    if (nodes.length === 0) return [...nodes];

    const { indices, contents } = this.partition(nodes);
    if (contents.length === 0) return [...nodes];

    const query = this.resolveQuery(queryBundle);
    if (this.compressionModel === 'latte_v1' && !query) {
      getLogger().warn(
        'CompresrNodePostprocessor: latte_v1 needs a query; none found. Passthrough.'
      );
      return [...nodes];
    }

    const out = nodes.map(cloneNodeWithScore);

    for (let start = 0; start < contents.length; start += BATCH_LIMIT) {
      const chunk = contents.slice(start, start + BATCH_LIMIT);
      let results: { compressed_context: string }[];
      try {
        const resp = await this.compresr.compressBatch({
          contexts: chunk,
          ...(query !== undefined ? { queries: query } : {}),
          compressionModelName: this.compressionModel,
          targetCompressionRatio: this.effectiveRatio(chunk),
          ...(this.coarse !== undefined ? { coarse: this.coarse } : {}),
        });
        results = (resp.data?.results ?? []);
      } catch (exc) {
        if (this.onError === 'raise') throw exc;
        getLogger().warn(
          `postprocessor batch failed (${stringifyError(exc)}); passthrough ${chunk.length} nodes.`
        );
        continue;
      }

      results.slice(0, chunk.length).forEach((item, offset) => {
        const globalIdx = indices[start + offset];
        const newText = item?.compressed_context;
        if (typeof newText === 'string' && newText && globalIdx !== undefined) {
          const node = out[globalIdx];
          if (node !== undefined) {
            setNodeText(node, newText);
          }
        }
      });
    }

    return out;
  }
}


function getNodeText(nws: NodeWithScoreType): string | undefined {
  const node = (nws as { node?: unknown }).node;
  if (!node || typeof node !== 'object') return undefined;
  const n = node as { getContent?: () => string; text?: string };
  if (typeof n.getContent === 'function') {
    try {
      const t = n.getContent();
      if (typeof t === 'string') return t;
    } catch {
      /* fall through */
    }
  }
  if (typeof n.text === 'string') return n.text;
  return undefined;
}

function cloneNodeWithScore(nws: NodeWithScoreType): NodeWithScoreType {
  return { ...(nws as object) } as NodeWithScoreType;
}

function setNodeText(nws: NodeWithScoreType, newText: string): void {
  const node = (nws as { node?: unknown }).node;
  if (!node || typeof node !== 'object') return;
  const n = node as { setContent?: (t: string) => void; text?: string };
  if (typeof n.setContent === 'function') {
    try {
      n.setContent(newText);
      return;
    } catch {
      /* fall through */
    }
  }
  try {
    n.text = newText;
    return;
  } catch {
    /* fall through */
  }
  const md = (node as { metadata?: Record<string, unknown> }).metadata;
  if (md && typeof md === 'object') {
    md['compresr_compressed'] = newText;
  }
  getLogger().warn(
    'node has no writable text field; compressed text stored in metadata.compresr_compressed but the synthesizer will see the original text.'
  );
}

function stringifyError(exc: unknown): string {
  return exc instanceof Error ? exc.message : String(exc);
}
