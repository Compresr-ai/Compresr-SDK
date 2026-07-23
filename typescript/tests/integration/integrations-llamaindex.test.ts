/**
 * Live end-to-end tests for the LlamaIndex integration (TypeScript).
 */
import { describe, it, expect, beforeAll } from 'vitest';

import {
  CompresrNodePostprocessor,
  wrapToolWithCompresr,
} from '../../src/integrations/llamaindex/index.js';
import { getLiveContext, LIVE_LONG_TEXT, LIVE_QUERY } from './_live-config.js';

interface MutableNode {
  text: string;
  metadata: Record<string, unknown>;
  getContent(): string;
  setContent(t: string): void;
}

function nws(text: string, score = 1): { node: MutableNode; score: number } {
  const node: MutableNode = {
    text,
    metadata: {},
    getContent() {
      return this.text;
    },
    setContent(t: string) {
      this.text = t;
    },
  };
  return { node, score };
}

describe('llamaindex integration (live)', () => {
  let ctx: Awaited<ReturnType<typeof getLiveContext>>;

  beforeAll(async () => {
    ctx = await getLiveContext();
  });

  it.skipIf(!process.env.COMPRESR_API_KEY)(
    'postprocessor shrinks retrieved nodes with latte_v1',
    async () => {
      if (!ctx) return;
      const pp = new CompresrNodePostprocessor({
        client: ctx.client,
        compressionModel: 'latte_v1',
        targetCompressionRatio: 0.5,
        minTokens: 100,
      });
      const nodes = [nws(LIVE_LONG_TEXT), nws(LIVE_LONG_TEXT), nws(LIVE_LONG_TEXT)] as never;
      const out = await pp.postprocessNodes(nodes, { queryStr: LIVE_QUERY });
      expect(out).toHaveLength(3);
      out.forEach((n: { node: MutableNode }) => {
        const text = n.node.getContent();
        expect(text.length).toBeGreaterThan(0);
        expect(text.length).toBeLessThan(LIVE_LONG_TEXT.length);
      });
    }
  );

  it.skipIf(!process.env.COMPRESR_API_KEY)(
    'wrapToolWithCompresr shrinks return value end-to-end',
    async () => {
      if (!ctx) return;
      const tool = {
        metadata: { name: 'search', description: 'search' },
        async call(_args: { query: string }) {
          return LIVE_LONG_TEXT;
        },
      };
      const wrapped = wrapToolWithCompresr(tool, {
        client: ctx.client,
        compressionModel: 'latte_v1',
        queryArg: 'query',
        targetCompressionRatio: 0.6,
        minTokens: 100,
      });
      const out = (await wrapped.call({ query: LIVE_QUERY })) as string;
      expect(out.length).toBeGreaterThan(0);
      expect(out.length).toBeLessThan(LIVE_LONG_TEXT.length);
    }
  );
});
