/** Unit tests for SDK schemas. */
import { describe, expect, it } from 'vitest';

import {
  CompressBatchInputSchema,
  CompressBatchRequestSchema,
  CompressBatchResponseSchema,
  CompressBatchResultSchema,
  CompressRequestSchema,
  CompressResponseSchema,
  CompressResultSchema,
  StreamChunkSchema,
} from '../../src/schemas/index.js';

describe('CompressRequestSchema', () => {
  it('accepts a minimal request with query', () => {
    const result = CompressRequestSchema.parse({
      context: 'Test context',
      query: 'What is X?',
      compression_model_name: 'latte_v1',
    });
    expect(result.context).toBe('Test context');
    expect(result.query).toBe('What is X?');
  });

  it('allows omitting query (backend validates per model)', () => {
    const result = CompressRequestSchema.parse({
      context: 'Test context',
      compression_model_name: 'latte_v1',
    });
    expect(result.query).toBeUndefined();
  });

  it('rejects empty context', () => {
    expect(() =>
      CompressRequestSchema.parse({
        context: '',
        compression_model_name: 'latte_v1',
      })
    ).toThrow();
  });

  it('rejects empty query when supplied', () => {
    expect(() =>
      CompressRequestSchema.parse({
        context: 'Test',
        query: '',
        compression_model_name: 'latte_v1',
      })
    ).toThrow();
  });

  it('accepts arbitrary model name (backend validates)', () => {
    const result = CompressRequestSchema.parse({
      context: 'Test',
      compression_model_name: 'future_v3',
    });
    expect(result.compression_model_name).toBe('future_v3');
  });

  it('accepts target_compression_ratio', () => {
    const result = CompressRequestSchema.parse({
      context: 'Test',
      query: 'Q',
      compression_model_name: 'latte_v1',
      target_compression_ratio: 0.5,
    });
    expect(result.target_compression_ratio).toBe(0.5);
  });

  it('rejects negative ratio', () => {
    expect(() =>
      CompressRequestSchema.parse({
        context: 'Test',
        compression_model_name: 'latte_v1',
        target_compression_ratio: -0.1,
      })
    ).toThrow();
  });
});

describe('CompressRequestSchema — latte_v2 dynamic fields', () => {
  it('defaults all three dynamic fields to undefined', () => {
    const result = CompressRequestSchema.parse({
      context: 'Test',
      query: 'Q',
      compression_model_name: 'latte_v1',
    });
    expect(result.dynamic).toBeUndefined();
    expect(result.dynamic_min_ratio).toBeUndefined();
    expect(result.dynamic_max_ratio).toBeUndefined();
  });

  it('accepts dynamic + dynamic_min_ratio + dynamic_max_ratio', () => {
    const result = CompressRequestSchema.parse({
      context: 'Test',
      query: 'Q',
      compression_model_name: 'latte_v2',
      dynamic: true,
      dynamic_min_ratio: 2.0,
      dynamic_max_ratio: 8.0,
    });
    expect(result.dynamic).toBe(true);
    expect(result.dynamic_min_ratio).toBe(2.0);
    expect(result.dynamic_max_ratio).toBe(8.0);
  });

  it('does not expose internal aggregation/include_tokens knobs', () => {
    // Regression: these were briefly added then removed in 2.7.9 / 1.6.8 —
    // they're internal/debug knobs and must stay out of the SDK surface.
    const shape = CompressRequestSchema.shape;
    expect('aggregation' in shape).toBe(false);
    expect('include_tokens' in shape).toBe(false);
  });
});

describe('CompressBatchInputSchema', () => {
  it('accepts context + query', () => {
    const result = CompressBatchInputSchema.parse({ context: 'C', query: 'Q' });
    expect(result.context).toBe('C');
    expect(result.query).toBe('Q');
  });

  it('allows omitting query', () => {
    const result = CompressBatchInputSchema.parse({ context: 'C' });
    expect(result.query).toBeUndefined();
  });

  it('rejects empty context', () => {
    expect(() => CompressBatchInputSchema.parse({ context: '' })).toThrow();
  });
});

describe('CompressBatchRequestSchema', () => {
  it('accepts a valid batch', () => {
    const result = CompressBatchRequestSchema.parse({
      inputs: [
        { context: 'C1', query: 'Q1' },
        { context: 'C2', query: 'Q2' },
      ],
      compression_model_name: 'latte_v1',
    });
    expect(result.inputs).toHaveLength(2);
  });

  it('rejects empty inputs', () => {
    expect(() =>
      CompressBatchRequestSchema.parse({
        inputs: [],
        compression_model_name: 'latte_v1',
      })
    ).toThrow();
  });

  it('rejects more than 100 inputs', () => {
    const inputs = Array.from({ length: 101 }, (_, i) => ({
      context: `C${i}`,
      query: `Q${i}`,
    }));
    expect(() =>
      CompressBatchRequestSchema.parse({
        inputs,
        compression_model_name: 'latte_v1',
      })
    ).toThrow();
  });

  it('accepts latte_v2 dynamic fields on the batch', () => {
    const result = CompressBatchRequestSchema.parse({
      inputs: [{ context: 'C', query: 'Q' }],
      compression_model_name: 'latte_v2',
      dynamic: true,
      dynamic_min_ratio: 1.5,
      dynamic_max_ratio: 10.0,
    });
    expect(result.dynamic).toBe(true);
    expect(result.dynamic_min_ratio).toBe(1.5);
    expect(result.dynamic_max_ratio).toBe(10.0);
  });
});

describe('CompressResultSchema', () => {
  it('accepts a valid result', () => {
    const result = CompressResultSchema.parse({
      compressed_context: 'C',
      original_tokens: 100,
      compressed_tokens: 50,
      actual_compression_ratio: 0.5,
      tokens_saved: 50,
      duration_ms: 100,
    });
    expect(result.compressed_tokens).toBe(50);
  });
});

describe('CompressResponseSchema', () => {
  it('accepts a success response', () => {
    const result = CompressResponseSchema.parse({
      success: true,
      data: {
        compressed_context: 'C',
        original_tokens: 100,
        compressed_tokens: 50,
        actual_compression_ratio: 0.5,
        tokens_saved: 50,
        duration_ms: 100,
      },
    });
    expect(result.success).toBe(true);
  });

  it('accepts a failure response with null data', () => {
    const result = CompressResponseSchema.parse({
      success: false,
      data: null,
      message: 'failed',
    });
    expect(result.success).toBe(false);
    expect(result.data).toBeNull();
  });
});

describe('CompressBatchResultSchema', () => {
  it('accepts a valid batch result', () => {
    const result = CompressBatchResultSchema.parse({
      results: [
        {
          compressed_context: 'C',
          original_tokens: 10,
          compressed_tokens: 5,
          actual_compression_ratio: 0.5,
          tokens_saved: 5,
          duration_ms: 10,
        },
      ],
      total_original_tokens: 10,
      total_compressed_tokens: 5,
      total_tokens_saved: 5,
      average_compression_ratio: 0.5,
      count: 1,
    });
    expect(result.count).toBe(1);
  });
});

describe('CompressBatchResponseSchema', () => {
  it('accepts a success response', () => {
    const result = CompressBatchResponseSchema.parse({
      success: true,
      data: {
        results: [],
        total_original_tokens: 0,
        total_compressed_tokens: 0,
        total_tokens_saved: 0,
        average_compression_ratio: 0,
        count: 0,
      },
    });
    expect(result.success).toBe(true);
  });
});

describe('StreamChunkSchema', () => {
  it('accepts a content chunk', () => {
    const chunk = StreamChunkSchema.parse({ content: 'partial', done: false });
    expect(chunk.content).toBe('partial');
  });

  it('accepts a done chunk', () => {
    const chunk = StreamChunkSchema.parse({ content: '', done: true });
    expect(chunk.done).toBe(true);
  });
});
