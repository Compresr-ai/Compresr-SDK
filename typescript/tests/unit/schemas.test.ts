/**
 * Unit tests for Zod schemas
 */
import { describe, it, expect } from 'vitest';
import {
  CompressRequestSchema,
  CompressResponseSchema,
  CompressBatchRequestSchema,
  CompressBatchResponseSchema,
  StreamChunkSchema,
} from '../../src/schemas/index.js';

describe('CompressRequestSchema', () => {
  it('should validate minimal request', () => {
    const result = CompressRequestSchema.safeParse({
      context: 'Hello world',
      compression_model_name: 'espresso_v1',
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.context).toBe('Hello world');
      expect(result.data.compression_model_name).toBe('espresso_v1');
      expect(result.data.source).toBe('sdk:typescript');
    }
  });

  it('should validate request with all fields', () => {
    const result = CompressRequestSchema.safeParse({
      context: 'Hello world',
      compression_model_name: 'latte_v1',
      query: 'What is this?',
      target_compression_ratio: 0.5,
      coarse: true,
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.query).toBe('What is this?');
      expect(result.data.target_compression_ratio).toBe(0.5);
      expect(result.data.coarse).toBe(true);
    }
  });

  it('should validate request with array context', () => {
    const result = CompressRequestSchema.safeParse({
      context: ['Doc 1', 'Doc 2'],
      compression_model_name: 'espresso_v1',
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.context).toEqual(['Doc 1', 'Doc 2']);
    }
  });

  it('should reject empty context string', () => {
    const result = CompressRequestSchema.safeParse({
      context: '',
      compression_model_name: 'espresso_v1',
    });

    expect(result.success).toBe(false);
  });

  it('should reject empty context array', () => {
    const result = CompressRequestSchema.safeParse({
      context: [],
      compression_model_name: 'espresso_v1',
    });

    expect(result.success).toBe(false);
  });

  it('should reject negative compression ratio', () => {
    const result = CompressRequestSchema.safeParse({
      context: 'Hello',
      compression_model_name: 'espresso_v1',
      target_compression_ratio: -0.5,
    });

    expect(result.success).toBe(false);
  });
});

describe('CompressResponseSchema', () => {
  it('should validate successful response', () => {
    const result = CompressResponseSchema.safeParse({
      success: true,
      data: {
        original_context: 'Hello world',
        compressed_context: 'Hello',
        original_tokens: 10,
        compressed_tokens: 5,
        actual_compression_ratio: 0.5,
        tokens_saved: 5,
        duration_ms: 100,
      },
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.success).toBe(true);
      expect(result.data.data?.tokens_saved).toBe(5);
    }
  });

  it('should validate response with null data', () => {
    const result = CompressResponseSchema.safeParse({
      success: false,
      data: null,
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.data).toBeNull();
    }
  });
});

describe('CompressBatchRequestSchema', () => {
  it('should validate batch request', () => {
    const result = CompressBatchRequestSchema.safeParse({
      inputs: [
        { context: 'Doc 1', query: 'Q1' },
        { context: 'Doc 2', query: 'Q2' },
      ],
      compression_model_name: 'latte_v1',
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.inputs).toHaveLength(2);
    }
  });

  it('should reject empty inputs array', () => {
    const result = CompressBatchRequestSchema.safeParse({
      inputs: [],
      compression_model_name: 'latte_v1',
    });

    expect(result.success).toBe(false);
  });

  it('should reject inputs exceeding max length', () => {
    const inputs = Array.from({ length: 101 }, (_, i) => ({
      context: `Doc ${i}`,
      query: `Q${i}`,
    }));

    const result = CompressBatchRequestSchema.safeParse({
      inputs,
      compression_model_name: 'latte_v1',
    });

    expect(result.success).toBe(false);
  });
});

describe('CompressBatchResponseSchema', () => {
  it('should validate batch response', () => {
    const result = CompressBatchResponseSchema.safeParse({
      success: true,
      data: {
        results: [
          {
            original_context: 'Doc 1',
            compressed_context: 'D1',
            original_tokens: 10,
            compressed_tokens: 5,
            actual_compression_ratio: 0.5,
            tokens_saved: 5,
            duration_ms: 50,
          },
        ],
        total_original_tokens: 10,
        total_compressed_tokens: 5,
        total_tokens_saved: 5,
        average_compression_ratio: 0.5,
        count: 1,
      },
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.data?.count).toBe(1);
    }
  });
});

describe('StreamChunkSchema', () => {
  it('should validate stream chunk', () => {
    const result = StreamChunkSchema.safeParse({
      content: 'Hello',
      done: false,
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.content).toBe('Hello');
      expect(result.data.done).toBe(false);
    }
  });

  it('should default done to false', () => {
    const result = StreamChunkSchema.safeParse({
      content: 'Hello',
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.done).toBe(false);
    }
  });

  it('should include error field', () => {
    const result = StreamChunkSchema.safeParse({
      content: '',
      done: true,
      error: 'Something failed',
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.error).toBe('Something failed');
    }
  });
});
