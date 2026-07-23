/**
 * Pure-TS unit tests for the shared integration helpers.
 *
 * No peer-dependency imports — runs in any environment. Mirrors Python
 * `tests/unit/test_integrations_shared.py`.
 */
import { describe, expect, it } from 'vitest';

import {
  applyErrorPolicy,
  applyErrorPolicyAsync,
  COMMON_QUERY_KEYS,
  estimateTokens,
  extractQueryFromArgs,
  extractQueryFromMessages,
  makeFilter,
  resolveQuery,
} from '../../src/integrations/_shared/index.js';

// ---------------------------------------------------------------------------
// tokens
// ---------------------------------------------------------------------------

describe('estimateTokens', () => {
  it('returns 0 for empty string', () => {
    expect(estimateTokens('')).toBe(0);
  });
  it('returns at least 1 for any non-empty string', () => {
    expect(estimateTokens('hi')).toBeGreaterThanOrEqual(1);
  });
  it('grows with input length', () => {
    expect(estimateTokens('hello world '.repeat(100))).toBeGreaterThan(
      estimateTokens('hello world')
    );
  });
});

// ---------------------------------------------------------------------------
// filters
// ---------------------------------------------------------------------------

describe('makeFilter', () => {
  it('default allows everything', () => {
    const f = makeFilter();
    expect(f('anything')).toBe(true);
    expect(f(null)).toBe(true);
  });
  it('allow-list', () => {
    const f = makeFilter({ allow: ['a', 'b'] });
    expect(f('a')).toBe(true);
    expect(f('c')).toBe(false);
    expect(f(null)).toBe(false);
  });
  it('ignore-list', () => {
    const f = makeFilter({ ignore: ['x'] });
    expect(f('a')).toBe(true);
    expect(f('x')).toBe(false);
  });
  it('throws if both allow and ignore', () => {
    expect(() => makeFilter({ allow: ['a'], ignore: ['b'] })).toThrow();
  });
});

// ---------------------------------------------------------------------------
// extractQueryFromArgs
// ---------------------------------------------------------------------------

describe('extractQueryFromArgs', () => {
  it('preferred key wins', () => {
    const out = extractQueryFromArgs(
      { query: 'q', url: 'https://x' },
      { preferredKey: 'url' }
    );
    expect(out).toBe('https://x');
  });
  it('preferred missing returns undefined', () => {
    expect(
      extractQueryFromArgs({ a: 'b' }, { preferredKey: 'missing' })
    ).toBeUndefined();
  });
  it('common keys priority — query beats q', () => {
    expect(extractQueryFromArgs({ q: 'short', query: 'long' })).toBe('long');
  });
  it('no match returns undefined', () => {
    expect(extractQueryFromArgs({ unrelated: 'value' })).toBeUndefined();
  });
  it('empty args', () => {
    expect(extractQueryFromArgs({})).toBeUndefined();
    expect(extractQueryFromArgs(null)).toBeUndefined();
  });
  it('COMMON_QUERY_KEYS includes query + question', () => {
    expect(COMMON_QUERY_KEYS).toContain('query');
    expect(COMMON_QUERY_KEYS).toContain('question');
  });
});

// ---------------------------------------------------------------------------
// extractQueryFromMessages
// ---------------------------------------------------------------------------

describe('extractQueryFromMessages', () => {
  it('tool call args win', () => {
    const msgs = [
      { role: 'user', content: 'old question' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'abc', args: { query: 'actual intent' }, name: 'search' }],
      },
    ];
    expect(
      extractQueryFromMessages(msgs, { toolCallId: 'abc' })
    ).toBe('actual intent');
  });
  it('falls back to last user message', () => {
    const msgs = [
      { role: 'user', content: "what's up" },
      { role: 'assistant', content: 'hi' },
    ];
    expect(extractQueryFromMessages(msgs)).toBe("what's up");
  });
  it('fallback when empty', () => {
    expect(extractQueryFromMessages([], { fallback: 'X' })).toBe('X');
  });
});

// ---------------------------------------------------------------------------
// resolveQuery — the union point
// ---------------------------------------------------------------------------

describe('resolveQuery', () => {
  it('static wins over everything', () => {
    const out = resolveQuery({
      staticQuery: 'STATIC',
      args: { query: 'from_args' },
      messages: [{ role: 'user', content: 'from_msg' }],
    });
    expect(out).toBe('STATIC');
  });
  it('extractor wins over args', () => {
    const out = resolveQuery({
      extractor: () => 'EXTRACTED',
      extractorArg: {},
      args: { query: 'ARGS' },
    });
    expect(out).toBe('EXTRACTED');
  });
  it('args with key', () => {
    expect(
      resolveQuery({ args: { q: 'yes' }, argsKey: 'q' })
    ).toBe('yes');
  });
  it('args smart pick', () => {
    expect(resolveQuery({ args: { question: 'smart' } })).toBe('smart');
  });
  it('messages fallback', () => {
    expect(
      resolveQuery({ messages: [{ role: 'user', content: 'from history' }] })
    ).toBe('from history');
  });
  it('final fallback', () => {
    expect(resolveQuery({ fallback: 'FB' })).toBe('FB');
  });
  it('extractor exception swallowed', () => {
    const out = resolveQuery({
      extractor: () => {
        throw new Error('nope');
      },
      extractorArg: {},
      args: { query: 'fallback_to_args' },
    });
    expect(out).toBe('fallback_to_args');
  });
});

// ---------------------------------------------------------------------------
// error policy
// ---------------------------------------------------------------------------

describe('applyErrorPolicy', () => {
  it('passthrough on failure', () => {
    expect(
      applyErrorPolicy(
        () => {
          throw new Error('boom');
        },
        { fallback: 'ok' }
      )
    ).toBe('ok');
  });
  it('raise on failure', () => {
    expect(() =>
      applyErrorPolicy(
        () => {
          throw new Error('boom');
        },
        { fallback: 'ok', policy: 'raise' }
      )
    ).toThrow('boom');
  });
  it('success returns value', () => {
    expect(applyErrorPolicy(() => 42, { fallback: 0 })).toBe(42);
  });
});

describe('applyErrorPolicyAsync', () => {
  it('passthrough on rejection', async () => {
    const v = await applyErrorPolicyAsync(
      async () => {
        throw new Error('boom');
      },
      { fallback: 'ok' }
    );
    expect(v).toBe('ok');
  });
  it('raise on rejection', async () => {
    await expect(
      applyErrorPolicyAsync(
        async () => {
          throw new Error('boom');
        },
        { fallback: 'ok', policy: 'raise' }
      )
    ).rejects.toThrow('boom');
  });
});
