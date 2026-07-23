/**
 * `CompresrStore` — wrap a LangGraph.js `BaseStore` so long string fields
 * are compressed before being persisted to the underlying store.
 *
 * Composition over inheritance: holds an inner store and proxies all
 * methods, intercepting `put` and `batch` to walk the value object and
 * rewrite strings above `minTokens` in place. Lossy — there is no
 * decompression on read.
 */
import {
  BaseStore,
  type Item,
  type Operation,
  type OperationResults,
  type SearchItem,
} from '@langchain/langgraph-checkpoint';

import type { CompressionClient } from '../../clients/compression.js';
import {
  buildClient,
  compressSafe,
  DEFAULT_MODEL,
  DEFAULT_POLICY,
  DEFAULT_RATIO,
  type ErrorPolicy,
  estimateTokens,
} from '../_shared/index.js';

export interface CompresrStoreOptions {
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  /**
   * Query passed to `latte_v1`. If unset, the field name (e.g. `retrievedText`)
   * is used as the query.
   */
  query?: string;
  targetCompressionRatio?: number;
  minTokens?: number;
  coarse?: boolean;
  fields?: ReadonlySet<string>;
  onError?: ErrorPolicy;
}

interface PutLikeOp {
  namespace: string[];
  key: string;
  value: Record<string, unknown> | null;
  index?: false | string[];
}

function isPutOp(op: Operation): op is PutLikeOp & Operation {
  const o = op as { value?: unknown };
  return Object.prototype.hasOwnProperty.call(op, 'value') && typeof o.value !== 'undefined';
}

/**
 * Drop-in `BaseStore` wrapper that compresses string fields on write.
 *
 * @example
 * import { InMemoryStore } from '@langchain/langgraph';
 * import { CompresrStore } from '@compresr/sdk/integrations/langgraph';
 *
 * const store = new CompresrStore(new InMemoryStore(), {
 *   apiKey: process.env.COMPRESR_API_KEY!,
 *   fields: new Set(['retrievedText']),
 *   minTokens: 500,
 * });
 */
export class CompresrStore extends BaseStore {
  private readonly inner: BaseStore;
  private readonly compresr: CompressionClient;
  private readonly compressionModel: string;
  private readonly query: string | undefined;
  private readonly targetCompressionRatio: number;
  private readonly minTokens: number;
  private readonly coarse: boolean | undefined;
  private readonly fields: ReadonlySet<string> | undefined;
  private readonly onError: ErrorPolicy;

  constructor(inner: BaseStore, options: CompresrStoreOptions = {}) {
    super();
    const {
      apiKey,
      client,
      baseUrl,
      compressionModel = DEFAULT_MODEL,
      query,
      targetCompressionRatio = DEFAULT_RATIO,
      minTokens = 500,
      coarse,
      fields,
      onError = DEFAULT_POLICY,
    } = options;

    this.inner = inner;
    this.compresr = client ?? buildClient({ apiKey, baseUrl, caller: 'CompresrStore' });
    this.compressionModel = compressionModel;
    this.query = query;
    this.targetCompressionRatio = targetCompressionRatio;
    this.minTokens = Math.max(1, minTokens);
    this.coarse = coarse;
    this.fields = fields;
    this.onError = onError;
  }

  private shouldCompress(text: string, parentKey: string | undefined): boolean {
    if (this.fields !== undefined && (parentKey === undefined || !this.fields.has(parentKey))) {
      return false;
    }
    return estimateTokens(text) >= this.minTokens;
  }

  private async compressValue(text: string, parentKey: string | undefined): Promise<string> {
    return await compressSafe(this.compresr, {
      context: text,
      query: this.query ?? parentKey ?? 'stored content',
      compressionModel: this.compressionModel,
      targetCompressionRatio: this.targetCompressionRatio,
      ...(this.coarse !== undefined ? { coarse: this.coarse } : {}),
      minTokens: 1,
      onError: this.onError,
      contextLabel: `store:${parentKey ?? '?'}`,
    });
  }

  private async walk(value: unknown, parentKey: string | undefined): Promise<unknown> {
    if (typeof value === 'string') {
      if (!this.shouldCompress(value, parentKey)) return value;
      return await this.compressValue(value, parentKey);
    }
    if (Array.isArray(value)) {
      return Promise.all(value.map((v) => this.walk(v, parentKey)));
    }
    if (value !== null && typeof value === 'object') {
      const entries = await Promise.all(
        Object.entries(value as Record<string, unknown>).map(
          async ([k, v]) => [k, await this.walk(v, k)] as const
        )
      );
      return Object.fromEntries(entries);
    }
    return value;
  }

  async put(
    namespace: string[],
    key: string,
    value: Record<string, unknown>,
    index?: false | string[]
  ): Promise<void> {
    const compressed = (await this.walk(value, undefined)) as Record<string, unknown>;
    return this.inner.put(namespace, key, compressed, index);
  }

  async batch<Op extends Operation[]>(operations: Op): Promise<OperationResults<Op>> {
    const rewritten = await Promise.all(
      operations.map(async (op) => {
        if (isPutOp(op) && op.value !== null) {
          return { ...op, value: (await this.walk(op.value, undefined)) as Record<string, unknown> };
        }
        return op;
      })
    );
    return this.inner.batch(rewritten as Op);
  }

  async get(namespace: string[], key: string): Promise<Item | null> {
    return this.inner.get(namespace, key);
  }

  async search(
    namespacePrefix: string[],
    options?: { filter?: Record<string, unknown>; limit?: number; offset?: number; query?: string }
  ): Promise<SearchItem[]> {
    return this.inner.search(namespacePrefix, options);
  }

  async delete(namespace: string[], key: string): Promise<void> {
    return this.inner.delete(namespace, key);
  }

  async listNamespaces(options?: {
    prefix?: string[];
    suffix?: string[];
    maxDepth?: number;
    limit?: number;
    offset?: number;
  }): Promise<string[][]> {
    return this.inner.listNamespaces(options);
  }

  async start(): Promise<void> {
    await this.inner.start();
  }

  async stop(): Promise<void> {
    await this.inner.stop();
  }
}
