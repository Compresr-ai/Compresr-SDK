/**
 * `CompresrCheckpointSerializer` — compresses long string fields before
 * they hit the checkpoint store.
 *
 * Lossy by design: there is no decompression — compressed strings persist
 * in storage and flow back into state on resume. Use `fields` to restrict
 * compression to known-large keys.
 */
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

const SENTINEL_KEY = '__compresr__';

export interface CompresrCheckpointSerializerOptions {
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  /**
   * Query passed to `latte_v1`. If unset, the field name (e.g. `retrieved_text`)
   * is used as the query — a reasonable hint for what the value represents.
   */
  query?: string;
  targetCompressionRatio?: number;
  minTokens?: number;
  coarse?: boolean;
  fields?: ReadonlySet<string>;
  onError?: ErrorPolicy;
}

/**
 * Drop-in for LangGraph.js checkpoint serializers. Calls
 * `dumpsTyped(obj) → [tag, bytes]`; recursively rewrites long strings
 * into a `{ __compresr__: true, v: <compressed> }` sentinel before
 * JSON-encoding. `loadsTyped` returns the payload as-is (sentinels and all).
 */
export class CompresrCheckpointSerializer {
  private readonly compresr: CompressionClient;
  private readonly compressionModel: string;
  private readonly query: string | undefined;
  private readonly targetCompressionRatio: number;
  private readonly minTokens: number;
  private readonly coarse: boolean | undefined;
  private readonly fields: ReadonlySet<string> | undefined;
  private readonly onError: ErrorPolicy;

  constructor(options: CompresrCheckpointSerializerOptions) {
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

    this.compresr =
      client ?? buildClient({ apiKey, baseUrl, caller: 'CompresrCheckpointSerializer' });
    this.compressionModel = compressionModel;
    this.query = query;
    this.targetCompressionRatio = targetCompressionRatio;
    this.minTokens = Math.max(1, minTokens);
    this.coarse = coarse;
    this.fields = fields;
    this.onError = onError;
  }

  async dumpsTyped(obj: unknown): Promise<[string, Uint8Array]> {
    const transformed = await this.walkCompress(obj, undefined);
    const encoded = new TextEncoder().encode(JSON.stringify(transformed));
    return ['json', encoded];
  }

  loadsTyped([_tag, bytes]: [string, Uint8Array]): unknown {
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  private async walkCompress(value: unknown, parentKey: string | undefined): Promise<unknown> {
    if (typeof value === 'string') {
      if (!this.shouldCompress(value, parentKey)) return value;
      return await this.wrapCompressed(value, parentKey);
    }
    if (Array.isArray(value)) {
      return Promise.all(value.map((v) => this.walkCompress(v, parentKey)));
    }
    if (value !== null && typeof value === 'object') {
      const entries = await Promise.all(
        Object.entries(value as Record<string, unknown>).map(
          async ([k, v]) => [k, await this.walkCompress(v, k)] as const
        )
      );
      return Object.fromEntries(entries);
    }
    return value;
  }

  private shouldCompress(text: string, parentKey: string | undefined): boolean {
    if (this.fields !== undefined && (parentKey === undefined || !this.fields.has(parentKey))) {
      return false;
    }
    return estimateTokens(text) >= this.minTokens;
  }

  private async wrapCompressed(text: string, parentKey: string | undefined): Promise<unknown> {
    const compressed = await compressSafe(this.compresr, {
      context: text,
      query: this.query ?? parentKey ?? 'stored content',
      compressionModel: this.compressionModel,
      targetCompressionRatio: this.targetCompressionRatio,
      ...(this.coarse !== undefined ? { coarse: this.coarse } : {}),
      minTokens: 1,
      onError: this.onError,
      contextLabel: `checkpoint:${parentKey ?? '?'}`,
    });
    if (compressed === text) return text;
    return { [SENTINEL_KEY]: true, v: compressed };
  }
}
