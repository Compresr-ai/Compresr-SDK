/**
 * Live end-to-end tests for the LangGraph integration (TypeScript).
 *
 * Exercises `makeCompresrNode` end-to-end against the real backend.
 */
import { describe, it, expect, beforeAll } from 'vitest';

import { makeCompresrNode } from '../../src/integrations/langgraph/index.js';
import { getLiveContext, LIVE_LONG_TEXT, LIVE_QUERY } from './_live-config.js';

interface State extends Record<string, unknown> {
  user_question?: string;
  retrieved_text?: string;
}

describe('langgraph integration (live)', () => {
  let ctx: Awaited<ReturnType<typeof getLiveContext>>;

  beforeAll(async () => {
    ctx = await getLiveContext();
  });

  it.skipIf(!process.env.COMPRESR_API_KEY)(
    'compression node shrinks retrieved text in a real workflow',
    async () => {
      if (!ctx) return;
      const compress = makeCompresrNode<State>({
        client: ctx.client,
        contextKey: 'retrieved_text',
        queryKey: 'user_question',
        compressionModel: 'latte_v1',
        targetCompressionRatio: 0.5,
        minTokens: 100,
      });

      const initialState: State = {
        user_question: LIVE_QUERY,
        retrieved_text: LIVE_LONG_TEXT,
      };
      const patch = await compress(initialState);
      expect(typeof patch.retrieved_text).toBe('string');
      const out = patch.retrieved_text as string;
      expect(out.length).toBeGreaterThan(0);
      expect(out.length).toBeLessThan(LIVE_LONG_TEXT.length);
    }
  );
});
