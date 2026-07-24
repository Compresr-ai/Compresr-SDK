/**
 * Shared live-test setup. Skips the whole suite when:
 *   - COMPRESR_API_KEY is missing, OR
 *   - the configured backend doesn't answer a small ping compress.
 *
 * This prevents silent passthroughs (compression "fails open" by default
 * in integrations) from masquerading as failing assertions.
 */
import { CompressionClient } from '../../src/clients/compression.js';

export const LIVE_LONG_TEXT = (
  "The Transformer architecture introduced by Vaswani et al. in 2017 in " +
  "'Attention Is All You Need' replaced recurrent layers with multi-head " +
  "self-attention. This enabled massive parallelization on GPUs and " +
  "removed the sequential bottleneck of LSTMs and GRUs. The encoder-decoder " +
  "design with stacked attention blocks became the foundation for BERT, " +
  'GPT, T5, and most modern large language models. '
).repeat(8);

export const LIVE_QUERY = 'What replaced recurrence in the Transformer?';

export interface LiveContext {
  client: CompressionClient;
}

let cached: LiveContext | null | undefined = undefined; // undefined = not probed yet

export async function getLiveContext(): Promise<LiveContext | null> {
  if (cached !== undefined) return cached;

  const apiKey = process.env.COMPRESR_API_KEY;
  if (!apiKey) {
    cached = null;
    return null;
  }
  const baseUrl = process.env.COMPRESR_BASE_URL;
  const client = new CompressionClient({
    apiKey,
    ...(baseUrl ? { baseUrl } : {}),
  });

  // Probe: a small compress against the real backend.
  try {
    await client.compress({
      context: 'ping '.repeat(60),
      query: 'ping',
      targetCompressionRatio: 0.5,
    });
  } catch {
    cached = null;
    return null;
  }

  cached = { client };
  return cached;
}
