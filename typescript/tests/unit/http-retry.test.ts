/**
 * Unit tests for the transport-layer retry policy.
 *
 * Defaults retry 429/503 with exponential backoff; the test suite locks
 * down the defaults, the success-after-retry path, the no-retry-on-4xx
 * guard, exhaustion, opt-out (`maxRetries: 0`), and the `Retry-After`
 * server-hint behaviour. Mirrors `tests/unit/test_retry.py` in the Python
 * SDK so both clients evolve together.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CompressionClient } from '../../src/clients/compression.js';
import {
  DEFAULT_RETRY_CONFIG,
  computeBackoffMs,
  resolveRetryConfig,
} from '../../src/http/retry.js';
import {
  RateLimitError,
  ServiceUnavailableError,
  ValidationError,
} from '../../src/errors/index.js';

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

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

describe('resolveRetryConfig', () => {
  it('exposes the documented defaults', () => {
    expect(DEFAULT_RETRY_CONFIG.maxRetries).toBe(3);
    expect(DEFAULT_RETRY_CONFIG.initialBackoffMs).toBe(500);
    expect(DEFAULT_RETRY_CONFIG.maxBackoffMs).toBe(30_000);
    expect(DEFAULT_RETRY_CONFIG.multiplier).toBe(2.0);
    expect(DEFAULT_RETRY_CONFIG.jitter).toBe(0.25);
    expect(DEFAULT_RETRY_CONFIG.respectRetryAfter).toBe(true);
    expect([...DEFAULT_RETRY_CONFIG.retryOnStatus].sort()).toEqual([429, 503]);
  });

  it('merges a partial user config onto the defaults', () => {
    const cfg = resolveRetryConfig({ maxRetries: 5, initialBackoffMs: 10 });
    expect(cfg.maxRetries).toBe(5);
    expect(cfg.initialBackoffMs).toBe(10);
    expect(cfg.maxBackoffMs).toBe(DEFAULT_RETRY_CONFIG.maxBackoffMs);
    expect(cfg.jitter).toBe(DEFAULT_RETRY_CONFIG.jitter);
  });

  it('rejects invalid input', () => {
    expect(() => resolveRetryConfig({ maxRetries: -1 })).toThrow(RangeError);
    expect(() => resolveRetryConfig({ initialBackoffMs: -1 })).toThrow(RangeError);
    expect(() => resolveRetryConfig({ multiplier: 0.5 })).toThrow(RangeError);
    expect(() => resolveRetryConfig({ jitter: 2 })).toThrow(RangeError);
  });
});

describe('computeBackoffMs', () => {
  const base = resolveRetryConfig({
    initialBackoffMs: 100,
    multiplier: 2.0,
    jitter: 0,
    maxBackoffMs: 10_000,
  });

  it('grows exponentially without jitter', () => {
    expect(computeBackoffMs(0, base)).toBe(100);
    expect(computeBackoffMs(1, base)).toBe(200);
    expect(computeBackoffMs(2, base)).toBe(400);
    expect(computeBackoffMs(3, base)).toBe(800);
  });

  it('caps at maxBackoffMs', () => {
    const cfg = resolveRetryConfig({ initialBackoffMs: 100, multiplier: 100, jitter: 0, maxBackoffMs: 500 });
    expect(computeBackoffMs(5, cfg)).toBe(500);
  });

  it('keeps jittered delays within bounds', () => {
    const cfg = resolveRetryConfig({ initialBackoffMs: 1000, multiplier: 1, jitter: 0.5, maxBackoffMs: 10_000 });
    for (let i = 0; i < 50; i++) {
      const v = computeBackoffMs(0, cfg);
      expect(v).toBeGreaterThanOrEqual(500);
      expect(v).toBeLessThanOrEqual(1500);
    }
  });

  it('honors retryAfter (seconds) when respectRetryAfter is true', () => {
    expect(computeBackoffMs(0, base, 7.5)).toBe(7500);
  });

  it('still caps retryAfter at maxBackoffMs', () => {
    expect(computeBackoffMs(0, base, 120)).toBe(10_000);
  });

  it('ignores retryAfter when respectRetryAfter is false', () => {
    const cfg = resolveRetryConfig({
      initialBackoffMs: 250,
      jitter: 0,
      maxBackoffMs: 10_000,
      respectRetryAfter: false,
    });
    expect(computeBackoffMs(0, cfg, 60)).toBe(250);
  });
});

describe('HttpClient retry behaviour via CompressionClient.compress', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('retries on 503 then succeeds', async () => {
    const errBody = { error: 'busy', code: 'service_unavailable' };
    mockFetch
      .mockResolvedValueOnce(jsonResponse(errBody, 503))
      .mockResolvedValueOnce(jsonResponse(errBody, 503))
      .mockResolvedValueOnce(jsonResponse(SUCCESS_BODY));

    const client = new CompressionClient({
      apiKey: 'cmp_test',
      retry: { maxRetries: 3, initialBackoffMs: 0, jitter: 0 },
    });
    const result = await client.compress({ context: 'x', query: 'y' });
    expect(result.data?.compressed_context).toBe('C');
    expect(mockFetch).toHaveBeenCalledTimes(3);
  });

  it('retries on 429 then succeeds', async () => {
    mockFetch
      .mockResolvedValueOnce(jsonResponse({ error: 'rate' }, 429))
      .mockResolvedValueOnce(jsonResponse(SUCCESS_BODY));

    const client = new CompressionClient({
      apiKey: 'cmp_test',
      retry: { maxRetries: 2, initialBackoffMs: 0, jitter: 0 },
    });
    const result = await client.compress({ context: 'x', query: 'y' });
    expect(result.data?.compressed_context).toBe('C');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('does not retry on 4xx (validation)', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ error: 'bad', code: 'validation_error' }, 422));

    const client = new CompressionClient({
      apiKey: 'cmp_test',
      retry: { maxRetries: 3, initialBackoffMs: 0, jitter: 0 },
    });
    await expect(client.compress({ context: 'x', query: 'y' })).rejects.toBeInstanceOf(ValidationError);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('exhausts retries and raises ServiceUnavailableError', async () => {
    const errBody = { error: 'busy', code: 'service_unavailable' };
    mockFetch.mockImplementation(async () => jsonResponse(errBody, 503));

    const client = new CompressionClient({
      apiKey: 'cmp_test',
      retry: { maxRetries: 2, initialBackoffMs: 0, jitter: 0 },
    });
    await expect(client.compress({ context: 'x', query: 'y' })).rejects.toBeInstanceOf(
      ServiceUnavailableError
    );
    expect(mockFetch).toHaveBeenCalledTimes(3);
  });

  it('honors maxRetries=0 (no retry)', async () => {
    const errBody = { error: 'busy', code: 'service_unavailable' };
    mockFetch.mockImplementation(async () => jsonResponse(errBody, 503));

    const client = new CompressionClient({
      apiKey: 'cmp_test',
      retry: { maxRetries: 0 },
    });
    await expect(client.compress({ context: 'x', query: 'y' })).rejects.toBeInstanceOf(
      ServiceUnavailableError
    );
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('honors a Retry-After response header', async () => {
    const errBody = { error: 'busy', code: 'service_unavailable' };
    mockFetch
      .mockResolvedValueOnce(jsonResponse(errBody, 503, { 'Retry-After': '0.05' }))
      .mockResolvedValueOnce(jsonResponse(SUCCESS_BODY));

    const setTimeoutSpy = vi.spyOn(global, 'setTimeout');
    const client = new CompressionClient({
      apiKey: 'cmp_test',
      retry: { maxRetries: 1, initialBackoffMs: 99_000, jitter: 0 },
    });
    await client.compress({ context: 'x', query: 'y' });

    // Server hint (50ms) wins over the 99-second default backoff.
    const retryDelays = setTimeoutSpy.mock.calls
      .map((c) => c[1])
      .filter((d): d is number => typeof d === 'number' && d > 0 && d < 1_000);
    expect(retryDelays).toContain(50);
    setTimeoutSpy.mockRestore();
  });
});
