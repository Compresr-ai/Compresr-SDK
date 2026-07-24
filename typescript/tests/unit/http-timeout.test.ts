/**
 * Regression guard: the 5-minute default must reach both ``post()`` and
 * ``stream()``, and an explicit ``timeout`` option must replace it. Before
 * this consolidation, ``stream()`` ignored the per-client setting and used
 * a separate ``STREAM_TIMEOUT`` constant.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CompressionClient } from '../../src/clients/compression.js';
import { DEFAULT_TIMEOUT } from '../../src/config/constants.js';

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

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function streamResponse(): Response {
  const sse = 'data: {"content":"hi"}\n\ndata: [DONE]\n\n';
  return new Response(sse, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

describe('HttpClient timeout wiring', () => {
  let setTimeoutSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    mockFetch.mockReset();
    setTimeoutSpy = vi.spyOn(global, 'setTimeout');
  });

  afterEach(() => {
    setTimeoutSpy.mockRestore();
    vi.clearAllMocks();
  });

  it('default timeout constant is five minutes', () => {
    expect(DEFAULT_TIMEOUT).toBe(300_000);
  });

  it('post() schedules abort using DEFAULT_TIMEOUT when no override', async () => {
    mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
    const client = new CompressionClient({ apiKey: 'cmp_test' });
    await client.compress({ context: 'x', query: 'y' });

    const delays = setTimeoutSpy.mock.calls.map((c) => c[1]);
    expect(delays).toContain(DEFAULT_TIMEOUT);
  });

  it('post() honours an explicit per-client timeout', async () => {
    mockFetch.mockResolvedValue(jsonResponse(SUCCESS_BODY));
    const client = new CompressionClient({ apiKey: 'cmp_test', timeout: 42 });
    await client.compress({ context: 'x', query: 'y' });

    const delays = setTimeoutSpy.mock.calls.map((c) => c[1]);
    expect(delays).toContain(42);
    expect(delays).not.toContain(DEFAULT_TIMEOUT);
  });

  it('stream() uses the same per-client timeout as post()', async () => {
    mockFetch.mockResolvedValue(streamResponse());
    const client = new CompressionClient({ apiKey: 'cmp_test', timeout: 77 });

    const chunks: string[] = [];
    for await (const chunk of client.compressStream({
      context: 'x',
      query: 'y',
    })) {
      if (!chunk.done) chunks.push(chunk.content);
    }

    const delays = setTimeoutSpy.mock.calls.map((c) => c[1]);
    expect(delays).toContain(77);
  });
});
