/**
 * Live end-to-end tests for the LangChain integration (TypeScript).
 *
 * Hits the real Compresr backend. Requires `COMPRESR_API_KEY` in the
 * environment. Skips cleanly otherwise.
 *
 * Run via:
 *   npm run test:integration
 *   COMPRESR_BASE_URL=https://api.compresr.ai npm run test:integration
 */
import { Document } from '@langchain/core/documents';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { describe, it, expect, beforeAll } from 'vitest';
import { z } from 'zod';

import {
  CompresrExtractor,
  compresrToolMiddleware,
  wrapToolWithCompression,
} from '../../src/integrations/langchain/index.js';
import { getLiveContext, LIVE_LONG_TEXT, LIVE_QUERY } from './_live-config.js';

describe('langchain integrations (live)', () => {
  let ctx: Awaited<ReturnType<typeof getLiveContext>>;

  beforeAll(async () => {
    ctx = await getLiveContext();
  });

  it.skipIf(!process.env.COMPRESR_API_KEY)(
    'compresrToolMiddleware compresses real tool output',
    async () => {
      if (!ctx) return; // backend unreachable — already logged in probe.
      const mw = compresrToolMiddleware({
        client: ctx.client,
        compressionModel: 'latte_v1',
        queryArg: 'query',
        targetCompressionRatio: 0.5,
        minTokens: 100,
      });
      const handler = async () =>
        new ToolMessage({
          content: LIVE_LONG_TEXT,
          tool_call_id: 't1',
          name: 'web_search',
        });
      const out = await mw.wrapToolCall!(
        {
          toolCall: { id: 't1', name: 'web_search', args: { query: LIVE_QUERY } },
          messages: [new HumanMessage(LIVE_QUERY)],
        },
        handler
      );
      const compressed = (out as ToolMessage).content as string;
      expect(compressed.length).toBeGreaterThan(0);
      expect(compressed.length).toBeLessThan(LIVE_LONG_TEXT.length);
    }
  );

  it.skipIf(!process.env.COMPRESR_API_KEY)(
    'wrapToolWithCompression shrinks output end-to-end',
    async () => {
      if (!ctx) return;
      const searchTool = tool(async () => LIVE_LONG_TEXT, {
        name: 'web_search',
        description: 'noisy mock search',
        schema: z.object({ query: z.string() }),
      });
      const wrapped = wrapToolWithCompression(searchTool, {
        client: ctx.client,
        compressionModel: 'latte_v1',
        queryArg: 'query',
        targetCompressionRatio: 0.6,
        minTokens: 100,
      });
      const out = (await wrapped.invoke({ query: LIVE_QUERY })) as string;
      expect(out.length).toBeGreaterThan(0);
      expect(out.length).toBeLessThan(LIVE_LONG_TEXT.length);
    }
  );

  it.skipIf(!process.env.COMPRESR_API_KEY)(
    'CompresrExtractor shrinks retrieved docs',
    async () => {
      if (!ctx) return;
      const comp = new CompresrExtractor({
        client: ctx.client,
        compressionModel: 'latte_v1',
        targetCompressionRatio: 0.6,
        minTokens: 100,
      });
      const docs = Array.from(
        { length: 3 },
        (_v, i) =>
          new Document({ pageContent: LIVE_LONG_TEXT, metadata: { i } })
      );
      const out = await comp.compressDocuments(docs, LIVE_QUERY);
      expect(out).toHaveLength(3);
      out.forEach((d) => {
        expect(d.pageContent.length).toBeGreaterThan(0);
        expect(d.pageContent.length).toBeLessThan(LIVE_LONG_TEXT.length);
        expect(d.metadata.compresr).toBe(true);
      });
    }
  );
});
