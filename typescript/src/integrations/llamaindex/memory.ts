/**
 * `CompresrMemoryBlock` — long-term memory block that compresses the
 * conversation buffer with Compresr before serving it back to the model.
 *
 * LlamaIndex.TS's `BaseMemoryBlock` exposes `put` (sink) and `get` (source).
 * We accumulate puts into a single text buffer and, on `get`, return one
 * compressed message — or the raw buffer if it's below the compression
 * threshold.
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

interface PartialMemoryMessage {
  id?: string;
  role: string;
  content: unknown;
  createdAt?: Date;
}

interface BaseMemoryBlockShape {
  readonly id: string;
  readonly priority: number;
  readonly isLongTerm: boolean;
  get(messages?: PartialMemoryMessage[]): Promise<PartialMemoryMessage[]>;
  put(messages: PartialMemoryMessage[]): Promise<void>;
}

export interface CompresrMemoryBlockOptions {
  id?: string;
  priority?: number;
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  /**
   * Query passed to `latte_v1`. If unset, the most recent `user:` line in
   * the buffer is used (falling back to a generic placeholder).
   */
  query?: string;
  targetToken?: number;
  targetCompressionRatio?: number;
  minTokens?: number;
  coarse?: boolean;
  onError?: ErrorPolicy;
}

const DEFAULT_PRIORITY = 2;

/**
 * Drop-in `BaseMemoryBlock` for LlamaIndex.TS that compresses on `get`.
 *
 * @example
 * const memory = new Memory({
 *   tokenLimit: 8_000,
 *   memoryBlocks: [
 *     new CompresrMemoryBlock({ apiKey: process.env.COMPRESR_API_KEY!, targetToken: 2_000 }),
 *   ],
 * });
 */
export class CompresrMemoryBlock implements BaseMemoryBlockShape {
  readonly id: string;
  readonly priority: number;
  readonly isLongTerm = true;

  private readonly compresr: CompressionClient;
  private readonly compressionModel: string;
  private readonly explicitQuery: string | undefined;
  private readonly targetToken: number | undefined;
  private readonly targetCompressionRatio: number;
  private readonly minTokens: number;
  private readonly coarse: boolean | undefined;
  private readonly onError: ErrorPolicy;

  private buffer = '';

  constructor(options: CompresrMemoryBlockOptions = {}) {
    const {
      id,
      priority = DEFAULT_PRIORITY,
      apiKey,
      client,
      baseUrl,
      compressionModel = DEFAULT_MODEL,
      query,
      targetToken,
      targetCompressionRatio = DEFAULT_RATIO,
      minTokens = 200,
      coarse,
      onError = DEFAULT_POLICY,
    } = options;

    this.id = id ?? `compresr-${Math.random().toString(36).slice(2, 10)}`;
    this.priority = priority;
    this.compresr = client ?? buildClient({ apiKey, baseUrl, caller: 'CompresrMemoryBlock' });
    this.compressionModel = compressionModel;
    this.explicitQuery = query;
    this.targetToken = targetToken;
    this.targetCompressionRatio = targetCompressionRatio;
    this.minTokens = Math.max(1, minTokens);
    this.coarse = coarse;
    this.onError = onError;
  }

  private resolveQuery(): string {
    if (this.explicitQuery) return this.explicitQuery;
    const lines = this.buffer.split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i];
      if (line?.startsWith('user:')) {
        const q = line.slice('user:'.length).trim();
        if (q) return q;
      }
    }
    return 'conversation history';
  }

  // eslint-disable-next-line @typescript-eslint/require-await
  async put(messages: PartialMemoryMessage[]): Promise<void> {
    for (const m of messages) {
      const content = typeof m.content === 'string' ? m.content : '';
      if (!content) continue;
      const role = String(m.role ?? '');
      this.buffer = `${this.buffer}\n${role}: ${content}`.trim();
    }
  }

  async get(_messages?: PartialMemoryMessage[]): Promise<PartialMemoryMessage[]> {
    if (!this.buffer) return [];
    const tokens = estimateTokens(this.buffer);
    if (tokens < this.minTokens) {
      return [{ role: 'system', content: this.buffer }];
    }
    const ratio = this.computeRatio(tokens);
    const compressed = await compressSafe(this.compresr, {
      context: this.buffer,
      query: this.resolveQuery(),
      compressionModel: this.compressionModel,
      targetCompressionRatio: ratio,
      ...(this.coarse !== undefined ? { coarse: this.coarse } : {}),
      minTokens: this.minTokens,
      onError: this.onError,
      contextLabel: 'memory_block',
    });
    return [{ role: 'system', content: compressed }];
  }

  private computeRatio(currentTokens: number): number {
    if (this.targetToken && this.targetToken > 0) {
      return Math.max(currentTokens / this.targetToken, 1.0);
    }
    return this.targetCompressionRatio;
  }
}
