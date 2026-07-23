/**
 * In-process mock of CompressionClient for integration tests.
 *
 * Mirrors Python `FakeCompressionClient` in tests/integration/conftest.py.
 * Records every call; returns deterministic "compressed" output (first half
 * of the input + `<<C>>` marker).
 */
import type { CompressionClient } from '../../src/clients/compression.js';
import type {
  CompressResponse,
  CompressBatchResponse,
} from '../../src/schemas/index.js';

export interface FakeCall {
  context: string;
  query?: string;
  compression_model_name?: string;
  target_compression_ratio?: number;
  coarse?: boolean;
  compressionModelName?: string;
  targetCompressionRatio?: number;
}

export interface FakeBatchCall {
  contexts: string[];
  queries?: string | string[];
  compressionModelName?: string;
  targetCompressionRatio?: number;
  coarse?: boolean;
}

function half(text: string): string {
  return text.slice(0, Math.max(1, Math.floor(text.length / 2))) + '<<C>>';
}

export class FakeCompressionClient {
  public calls: FakeCall[] = [];
  public batchCalls: FakeBatchCall[] = [];
  private readonly shouldRaise: boolean;

  constructor(options: { raiseOnCall?: boolean } = {}) {
    this.shouldRaise = options.raiseOnCall ?? false;
  }

  async compress(options: {
    context: string;
    query?: string;
    compressionModelName?: string;
    targetCompressionRatio?: number;
    coarse?: boolean;
  }): Promise<CompressResponse> {
    this.calls.push({
      context: options.context,
      ...(options.query !== undefined ? { query: options.query } : {}),
      ...(options.compressionModelName !== undefined
        ? { compressionModelName: options.compressionModelName }
        : {}),
      ...(options.targetCompressionRatio !== undefined
        ? { targetCompressionRatio: options.targetCompressionRatio }
        : {}),
      ...(options.coarse !== undefined ? { coarse: options.coarse } : {}),
    });
    if (this.shouldRaise) {
      throw new Error('forced failure');
    }
    const out = half(options.context);
    const inT = Math.max(1, Math.floor(options.context.length / 4));
    const outT = Math.max(1, Math.floor(out.length / 4));
    return {
      success: true,
      data: {
        original_context: options.context,
        compressed_context: out,
        original_tokens: inT,
        compressed_tokens: outT,
        actual_compression_ratio: 0.5,
        tokens_saved: Math.max(0, inT - outT),
        duration_ms: 1,
      },
    } as CompressResponse;
  }

  async compressBatch(options: {
    contexts: string[];
    queries?: string | string[];
    compressionModelName?: string;
    targetCompressionRatio?: number;
    coarse?: boolean;
  }): Promise<CompressBatchResponse> {
    this.batchCalls.push({
      contexts: [...options.contexts],
      ...(options.queries !== undefined ? { queries: options.queries } : {}),
      ...(options.compressionModelName !== undefined
        ? { compressionModelName: options.compressionModelName }
        : {}),
      ...(options.targetCompressionRatio !== undefined
        ? { targetCompressionRatio: options.targetCompressionRatio }
        : {}),
      ...(options.coarse !== undefined ? { coarse: options.coarse } : {}),
    });
    if (this.shouldRaise) {
      throw new Error('forced failure');
    }

    let totalIn = 0;
    let totalOut = 0;
    const items = options.contexts.map((ctx) => {
      const out = half(ctx);
      const inT = Math.max(1, Math.floor(ctx.length / 4));
      const outT = Math.max(1, Math.floor(out.length / 4));
      totalIn += inT;
      totalOut += outT;
      return {
        original_context: ctx,
        compressed_context: out,
        original_tokens: inT,
        compressed_tokens: outT,
        actual_compression_ratio: 0.5,
        tokens_saved: inT - outT,
        duration_ms: 1,
      };
    });

    return {
      success: true,
      data: {
        results: items,
        total_original_tokens: totalIn,
        total_compressed_tokens: totalOut,
        total_tokens_saved: totalIn - totalOut,
        average_compression_ratio: 0.5,
        count: items.length,
      },
    } as CompressBatchResponse;
  }
}

/** Typed alias for use with the integration code (which expects the real client). */
export type FakeAsClient = FakeCompressionClient & CompressionClient;
