/** Tests for CompresrCheckpointSerializer (LangGraph). */
import { describe, expect, it } from 'vitest';

import { CompresrCheckpointSerializer } from '../../src/integrations/langgraph/index.js';
import { FakeCompressionClient, type FakeAsClient } from './_fake-client.js';

const LONG = 'x '.repeat(2000);

function decode(bytes: Uint8Array): unknown {
  return JSON.parse(new TextDecoder().decode(bytes));
}

describe('CompresrCheckpointSerializer', () => {
  it('compresses long string fields only', async () => {
    const fake = new FakeCompressionClient();
    const ser = new CompresrCheckpointSerializer({
      client: fake as FakeAsClient,
      minTokens: 10,
    });
    const [, bytes] = await ser.dumpsTyped({ long: LONG, short: 'hi' });
    const decoded = decode(bytes) as Record<string, unknown>;
    expect(decoded.short).toBe('hi');
    expect(decoded.long).toMatchObject({ __compresr__: true });
    expect((decoded.long as { v: string }).v).toContain('<<C>>');
    expect(fake.calls).toHaveLength(1);
  });

  it('respects the field allowlist', async () => {
    const fake = new FakeCompressionClient();
    const ser = new CompresrCheckpointSerializer({
      client: fake as FakeAsClient,
      minTokens: 10,
      fields: new Set(['history']),
    });
    await ser.dumpsTyped({ history: LONG, other_big: LONG });
    expect(fake.calls).toHaveLength(1);
    expect(fake.calls[0]?.context).toBe(LONG);
  });

  it('leaves short fields unchanged', async () => {
    const fake = new FakeCompressionClient();
    const ser = new CompresrCheckpointSerializer({
      client: fake as FakeAsClient,
      minTokens: 10_000,
    });
    await ser.dumpsTyped({ foo: LONG });
    expect(fake.calls).toHaveLength(0);
  });

  it('walks nested dicts + lists', async () => {
    const fake = new FakeCompressionClient();
    const ser = new CompresrCheckpointSerializer({
      client: fake as FakeAsClient,
      minTokens: 10,
    });
    await ser.dumpsTyped({ outer: { inner: LONG, tiny: 'x' }, list: [LONG, 'tiny'] });
    expect(fake.calls).toHaveLength(2);
  });

  it('passthrough on error', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const ser = new CompresrCheckpointSerializer({
      client: fake as FakeAsClient,
      minTokens: 10,
      onError: 'passthrough',
    });
    const [, bytes] = await ser.dumpsTyped({ long: LONG });
    const decoded = decode(bytes) as Record<string, unknown>;
    expect(decoded.long).toBe(LONG);
  });

  it('leaves non-string values untouched', async () => {
    const fake = new FakeCompressionClient();
    const ser = new CompresrCheckpointSerializer({
      client: fake as FakeAsClient,
      minTokens: 10,
    });
    const obj = { count: 42, ratio: 0.5, flag: true, missing: null };
    const [tag, bytes] = await ser.dumpsTyped(obj);
    expect(tag).toBe('json');
    expect(ser.loadsTyped([tag, bytes])).toEqual(obj);
    expect(fake.calls).toHaveLength(0);
  });
});
