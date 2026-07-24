/** Unit tests for ``CompressionClient`` — uses mocked global fetch. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CompressionClient } from '../../src/clients/compression.js';
import { AuthenticationError, ValidationError } from '../../src/errors/index.js';

const mockFetch = vi.fn();
global.fetch = mockFetch as never;

const SUCCESS_BODY = {
  success: true,
  data: {
    compressed_context: 'C',
    original_tokens: 100,
    compressed_tokens: 50,
    actual_compression_ratio: 0.5,
    tokens_saved: 50,
    duration_ms: 10,
  },
};

const SUCCESS_BATCH_BODY = {
  success: true,
  data: {
    results: [
      {
        compressed_context: 'C',
        original_tokens: 100,
        compressed_tokens: 50,
        actual_compression_ratio: 0.5,
        tokens_saved: 50,
        duration_ms: 10,
      },
    ],
    total_original_tokens: 100,
    total_compressed_tokens: 50,
    total_tokens_saved: 50,
    average_compression_ratio: 0.5,
    count: 1,
  },
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('CompressionClient', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('creates client with a valid API key', () => {
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      expect(client).toBeInstanceOf(CompressionClient);
    });

    it('rejects an empty API key', () => {
      expect(() => new CompressionClient({ apiKey: '' })).toThrow(
        AuthenticationError
      );
    });

    it('rejects an API key missing the cmp_ prefix', () => {
      expect(() => new CompressionClient({ apiKey: 'invalid' })).toThrow(
        /cmp_/
      );
    });
  });

  describe('compress', () => {
    it('hits the question-specific endpoint', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compress({ context: 'X', query: 'Q' });

      const [url] = mockFetch.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/api/compress/question-specific/');
    });

    it('defaults compression model to latte_v1', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compress({ context: 'X', query: 'Q' });

      const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.compression_model_name).toBe('latte_v1');
    });

    it('passes through arbitrary model names', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compress({
        context: 'X',
        query: 'Q',
        compressionModelName: 'future_v3',
      });

      const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.compression_model_name).toBe('future_v3');
    });

    it('allows omitting query', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compress({ context: 'X' });

      const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.query).toBeUndefined();
    });

    it('rejects empty context', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await expect(client.compress({ context: '' })).rejects.toThrow(
        ValidationError
      );
    });

    it('forwards dynamic fields with snake_case wire format', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compress({
        context: 'X',
        query: 'Q',
        compressionModelName: 'latte_v2',
        dynamic: true,
        dynamicMinRatio: 2.0,
        dynamicMaxRatio: 8.0,
      });

      const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.dynamic).toBe(true);
      expect(body.dynamic_min_ratio).toBe(2.0);
      expect(body.dynamic_max_ratio).toBe(8.0);
      // Sanity: camelCase variants must not leak onto the wire.
      expect(body.dynamicMinRatio).toBeUndefined();
      expect(body.dynamicMaxRatio).toBeUndefined();
    });

    it('omits dynamic fields when not set', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compress({ context: 'X', query: 'Q' });

      const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.dynamic).toBeUndefined();
      expect(body.dynamic_min_ratio).toBeUndefined();
      expect(body.dynamic_max_ratio).toBeUndefined();
    });
  });

  describe('compressBatch', () => {
    it('hits the batch endpoint with a single query applied to all', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BATCH_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compressBatch({
        contexts: ['A', 'B'],
        queries: 'shared',
      });

      const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      expect(url).toContain('/api/compress/question-specific/batch');
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.inputs).toHaveLength(2);
      expect(body.inputs[0].query).toBe('shared');
      expect(body.inputs[1].query).toBe('shared');
    });

    it('accepts per-context queries', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BATCH_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compressBatch({
        contexts: ['A', 'B'],
        queries: ['Q1', 'Q2'],
      });

      const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.inputs[0].query).toBe('Q1');
      expect(body.inputs[1].query).toBe('Q2');
    });

    it('rejects mismatched query count', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await expect(
        client.compressBatch({
          contexts: ['A', 'B'],
          queries: ['only one'],
        })
      ).rejects.toThrow(ValidationError);
    });

    it('allows omitting queries entirely', async () => {
      mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BATCH_BODY));
      const client = new CompressionClient({ apiKey: 'cmp_test' });
      await client.compressBatch({ contexts: ['A', 'B'] });

      const [, init] = mockFetch.mock.calls[0] as [string, RequestInit];
      const body = JSON.parse((init.body as string) ?? '{}');
      expect(body.inputs[0].query).toBeUndefined();
      expect(body.inputs[1].query).toBeUndefined();
    });
  });
});
