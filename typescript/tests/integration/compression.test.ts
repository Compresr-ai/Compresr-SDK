/** End-to-end ``CompressionClient`` tests against the real backend. */
import { beforeAll, describe, expect, it } from 'vitest';

import { getLiveContext, LIVE_LONG_TEXT, LIVE_QUERY } from './_live-config.js';

describe('CompressionClient (live)', () => {
  let ctx: Awaited<ReturnType<typeof getLiveContext>>;

  beforeAll(async () => {
    ctx = await getLiveContext();
  });

  it.skipIf(!process.env.COMPRESR_API_KEY)('compress shrinks long text', async () => {
    if (!ctx) return;
    const response = await ctx.client.compress({
      context: LIVE_LONG_TEXT,
      query: LIVE_QUERY,
      targetCompressionRatio: 0.5,
    });
    expect(response.success).toBe(true);
    expect(response.data).not.toBeNull();
    expect(response.data!.compressed_tokens).toBeLessThanOrEqual(
      response.data!.original_tokens
    );
  });

  it.skipIf(!process.env.COMPRESR_API_KEY)('compressBatch processes 3 docs', async () => {
    if (!ctx) return;
    const response = await ctx.client.compressBatch({
      contexts: [LIVE_LONG_TEXT, LIVE_LONG_TEXT, LIVE_LONG_TEXT],
      queries: LIVE_QUERY,
      targetCompressionRatio: 0.5,
    });
    expect(response.success).toBe(true);
    expect(response.data?.count).toBe(3);
    expect(response.data?.results).toHaveLength(3);
  });

  it.skipIf(!process.env.COMPRESR_API_KEY)(
    'compressBatch accepts per-context queries',
    async () => {
      if (!ctx) return;
      const response = await ctx.client.compressBatch({
        contexts: [LIVE_LONG_TEXT, LIVE_LONG_TEXT],
        queries: ['What is ML?', 'What is GPT?'],
        targetCompressionRatio: 0.5,
      });
      expect(response.success).toBe(true);
      expect(response.data?.count).toBe(2);
    }
  );
});
