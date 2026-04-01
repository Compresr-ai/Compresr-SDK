/**
 * Unit tests for CompressionClient
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { CompressionClient } from '../../src/clients/compression.js';
import { AuthenticationError, ValidationError } from '../../src/errors/index.js';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('CompressionClient', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('constructor', () => {
    it('should create client with valid API key', () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });
      expect(client).toBeInstanceOf(CompressionClient);
    });

    it('should throw AuthenticationError for missing API key', () => {
      expect(() => new CompressionClient({ apiKey: '' })).toThrow(
        AuthenticationError
      );
    });

    it('should throw AuthenticationError for invalid API key prefix', () => {
      expect(() => new CompressionClient({ apiKey: 'invalid_key' })).toThrow(
        AuthenticationError
      );
      expect(() => new CompressionClient({ apiKey: 'invalid_key' })).toThrow(
        /must start with 'cmp_'/
      );
    });
  });

  describe('compress', () => {
    it('should make request with espresso_v1 by default', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
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
        }),
      });

      const result = await client.compress({ context: 'Hello world' });

      expect(mockFetch).toHaveBeenCalledTimes(1);
      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toContain('/api/compress/question-agnostic/');
      expect(options.method).toBe('POST');
      
      const body = JSON.parse(options.body);
      expect(body.compression_model_name).toBe('espresso_v1');
      expect(body.context).toBe('Hello world');

      expect(result.success).toBe(true);
      expect(result.data?.tokens_saved).toBe(5);
    });

    it('should make request with latte_v1 when specified', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
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
        }),
      });

      await client.compress({
        context: 'Hello world',
        query: 'What is this?',
        compressionModelName: 'latte_v1',
      });

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toContain('/api/compress/question-specific/');
      
      const body = JSON.parse(options.body);
      expect(body.compression_model_name).toBe('latte_v1');
      expect(body.query).toBe('What is this?');
    });

    it('should throw ValidationError when latte_v1 used without query', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      await expect(
        client.compress({
          context: 'Hello world',
          compressionModelName: 'latte_v1',
        })
      ).rejects.toThrow(ValidationError);

      await expect(
        client.compress({
          context: 'Hello world',
          compressionModelName: 'latte_v1',
        })
      ).rejects.toThrow(/requires a 'query' parameter/);
    });

    it('should throw ValidationError when espresso_v1 used with query', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      await expect(
        client.compress({
          context: 'Hello world',
          query: 'What is this?',
          compressionModelName: 'espresso_v1',
        })
      ).rejects.toThrow(ValidationError);

      await expect(
        client.compress({
          context: 'Hello world',
          query: 'What is this?',
          compressionModelName: 'espresso_v1',
        })
      ).rejects.toThrow(/does not accept a 'query' parameter/);
    });

    it('should throw ValidationError for invalid model', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      await expect(
        client.compress({
          context: 'Hello world',
          compressionModelName: 'invalid_model',
        })
      ).rejects.toThrow(ValidationError);
    });

    it('should include compression ratio in request', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: {
            original_context: 'Hello',
            compressed_context: 'Hi',
            original_tokens: 5,
            compressed_tokens: 2,
            actual_compression_ratio: 0.4,
            tokens_saved: 3,
            duration_ms: 50,
          },
        }),
      });

      await client.compress({
        context: 'Hello',
        targetCompressionRatio: 0.5,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.target_compression_ratio).toBe(0.5);
    });

    it('should include coarse parameter in request', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: {
            original_context: 'Hello',
            compressed_context: 'Hi',
            original_tokens: 5,
            compressed_tokens: 2,
            actual_compression_ratio: 0.4,
            tokens_saved: 3,
            duration_ms: 50,
          },
        }),
      });

      await client.compress({
        context: 'Hello',
        query: 'What?',
        compressionModelName: 'latte_v1',
        coarse: true,
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.coarse).toBe(true);
    });

    it('should handle array context', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: {
            original_context: ['Doc 1', 'Doc 2'],
            compressed_context: ['D1', 'D2'],
            original_tokens: 20,
            compressed_tokens: 10,
            actual_compression_ratio: 0.5,
            tokens_saved: 10,
            duration_ms: 100,
          },
        }),
      });

      const result = await client.compress({
        context: ['Doc 1', 'Doc 2'],
      });

      expect(result.data?.compressed_context).toEqual(['D1', 'D2']);
    });
  });

  describe('compressBatch', () => {
    it('should make batch request with same query for all', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
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
              {
                original_context: 'Doc 2',
                compressed_context: 'D2',
                original_tokens: 10,
                compressed_tokens: 5,
                actual_compression_ratio: 0.5,
                tokens_saved: 5,
                duration_ms: 50,
              },
            ],
            total_original_tokens: 20,
            total_compressed_tokens: 10,
            total_tokens_saved: 10,
            average_compression_ratio: 0.5,
            count: 2,
          },
        }),
      });

      const result = await client.compressBatch({
        contexts: ['Doc 1', 'Doc 2'],
        queries: 'What is this?',
      });

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toContain('/api/compress/question-specific/batch');
      
      const body = JSON.parse(options.body);
      expect(body.inputs).toHaveLength(2);
      expect(body.inputs[0].query).toBe('What is this?');
      expect(body.inputs[1].query).toBe('What is this?');

      expect(result.data?.count).toBe(2);
      expect(result.data?.total_tokens_saved).toBe(10);
    });

    it('should make batch request with different queries', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: {
            results: [],
            total_original_tokens: 20,
            total_compressed_tokens: 10,
            total_tokens_saved: 10,
            average_compression_ratio: 0.5,
            count: 2,
          },
        }),
      });

      await client.compressBatch({
        contexts: ['Doc 1', 'Doc 2'],
        queries: ['Q1', 'Q2'],
      });

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.inputs[0].query).toBe('Q1');
      expect(body.inputs[1].query).toBe('Q2');
    });

    it('should throw ValidationError when queries length mismatch', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      await expect(
        client.compressBatch({
          contexts: ['Doc 1', 'Doc 2'],
          queries: ['Q1'], // Only 1 query for 2 contexts
        })
      ).rejects.toThrow(ValidationError);

      await expect(
        client.compressBatch({
          contexts: ['Doc 1', 'Doc 2'],
          queries: ['Q1'],
        })
      ).rejects.toThrow(/must match number of contexts/);
    });

    it('should throw ValidationError for non-query model in batch', async () => {
      const client = new CompressionClient({ apiKey: 'cmp_test_key' });

      await expect(
        client.compressBatch({
          contexts: ['Doc 1'],
          queries: 'Q1',
          compressionModelName: 'espresso_v1',
        })
      ).rejects.toThrow(ValidationError);

      await expect(
        client.compressBatch({
          contexts: ['Doc 1'],
          queries: 'Q1',
          compressionModelName: 'espresso_v1',
        })
      ).rejects.toThrow(/only supports query-specific models/);
    });
  });
});
