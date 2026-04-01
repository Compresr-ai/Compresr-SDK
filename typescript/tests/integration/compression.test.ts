/**
 * Integration tests for CompressionClient
 *
 * These tests require a valid COMPRESR_API_KEY environment variable.
 * Run with: npm run test:integration
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { CompressionClient } from '../../src/clients/compression.js';
import type { CompressResponse, CompressBatchResponse } from '../../src/schemas/index.js';
import { TEST_API_KEY, skipIfNoApiKey } from './config.js';

describe('CompressionClient Integration', () => {
  let client: CompressionClient;

  beforeAll(() => {
    if (skipIfNoApiKey()) {
      return;
    }
    client = new CompressionClient({ apiKey: TEST_API_KEY });
  });

  describe('compress with espresso_v1', () => {
    it.skipIf(skipIfNoApiKey())(
      'should compress context without query',
      async () => {
        const result: CompressResponse = await client.compress({
          context:
            'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.',
        });

        expect(result.success).toBe(true);
        expect(result.data).not.toBeNull();
        expect(result.data!.original_tokens).toBeGreaterThan(0);
        expect(result.data!.compressed_tokens).toBeGreaterThan(0);
        expect(result.data!.compressed_tokens).toBeLessThanOrEqual(
          result.data!.original_tokens
        );
        expect(result.data!.tokens_saved).toBeGreaterThanOrEqual(0);
        expect(result.data!.compressed_context).toBeTruthy();
      }
    );

    it.skipIf(skipIfNoApiKey())(
      'should compress array of contexts',
      async () => {
        const result: CompressResponse = await client.compress({
          context: [
            'Machine learning enables computers to learn from data.',
            'Deep learning uses neural networks with many layers.',
          ],
        });

        expect(result.success).toBe(true);
        expect(result.data).not.toBeNull();
        expect(Array.isArray(result.data!.compressed_context)).toBe(true);
        expect((result.data!.compressed_context as string[]).length).toBe(2);
      }
    );

    it.skipIf(skipIfNoApiKey())(
      'should respect compression ratio',
      async () => {
        const result: CompressResponse = await client.compress({
          context:
            'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.',
          targetCompressionRatio: 0.5,
        });

        expect(result.success).toBe(true);
        expect(result.data).not.toBeNull();
        // The actual ratio might not be exactly 0.5 but should be in that range
        expect(result.data!.actual_compression_ratio).toBeGreaterThan(0);
      }
    );
  });

  describe('compress with latte_v1', () => {
    it.skipIf(skipIfNoApiKey())(
      'should compress context with query',
      async () => {
        const result: CompressResponse = await client.compress({
          context:
            'Python is a programming language created by Guido van Rossum in 1991. JavaScript was created by Brendan Eich in 1995. Java was created by James Gosling also in 1995.',
          query: 'Who created Python?',
          compressionModelName: 'latte_v1',
        });

        expect(result.success).toBe(true);
        expect(result.data).not.toBeNull();
        expect(result.data!.compressed_context).toBeTruthy();
        // The compressed context should be relevant to Python
        expect(
          (result.data!.compressed_context as string).toLowerCase()
        ).toContain('python');
      }
    );

    it.skipIf(skipIfNoApiKey())(
      'should use coarse mode',
      async () => {
        const result: CompressResponse = await client.compress({
          context:
            'Python is a programming language.\n\nJavaScript is used for web development.\n\nJava is used for enterprise applications.',
          query: 'Tell me about Python',
          compressionModelName: 'latte_v1',
          coarse: true,
        });

        expect(result.success).toBe(true);
        expect(result.data).not.toBeNull();
      }
    );
  });

  describe('compressBatch', () => {
    it.skipIf(skipIfNoApiKey())(
      'should batch compress with same query',
      async () => {
        const result: CompressBatchResponse = await client.compressBatch({
          contexts: [
            'Document about machine learning applications.',
            'Document about deep learning architectures.',
            'Document about natural language processing.',
          ],
          queries: 'What are the key concepts?',
        });

        expect(result.success).toBe(true);
        expect(result.data).not.toBeNull();
        expect(result.data!.results.length).toBe(3);
        expect(result.data!.count).toBe(3);
        expect(result.data!.total_tokens_saved).toBeGreaterThanOrEqual(0);
      }
    );

    it.skipIf(skipIfNoApiKey())(
      'should batch compress with different queries',
      async () => {
        const result: CompressBatchResponse = await client.compressBatch({
          contexts: [
            'Machine learning is a subset of AI.',
            'Neural networks have multiple layers.',
          ],
          queries: ['What is ML?', 'What are neural networks?'],
        });

        expect(result.success).toBe(true);
        expect(result.data).not.toBeNull();
        expect(result.data!.results.length).toBe(2);
      }
    );
  });

  describe('compressStream', () => {
    it.skipIf(skipIfNoApiKey())(
      'should stream compression chunks',
      async () => {
        const chunks: string[] = [];

        for await (const chunk of client.compressStream({
          context:
            'Machine learning is a method of data analysis that automates analytical model building.',
        })) {
          if (chunk.content) {
            chunks.push(chunk.content);
          }
          if (chunk.done) {
            break;
          }
        }

        // Should have received at least some content
        expect(chunks.length).toBeGreaterThan(0);
        const fullContent = chunks.join('');
        expect(fullContent.length).toBeGreaterThan(0);
      }
    );
  });
});
