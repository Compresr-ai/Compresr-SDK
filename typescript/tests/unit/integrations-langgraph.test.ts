/**
 * Tests for makeCompresrNode (LangGraph integration).
 */
import { Command, InMemoryStore } from '@langchain/langgraph';
import { describe, expect, it } from 'vitest';

import {
  compresrHandoffTool,
  compresrNode,
  CompresrStore,
  makeCompresrNode,
} from '../../src/integrations/langgraph/index.js';

import { FakeCompressionClient, type FakeAsClient } from './_fake-client.js';

describe('compresrNode alias', () => {
  it('is the same factory as makeCompresrNode', () => {
    expect(compresrNode).toBe(makeCompresrNode);
  });
});

const LONG = 'x '.repeat(2000);

interface State extends Record<string, unknown> {
  ctx?: string;
  q?: string;
  topic?: string;
}

describe('makeCompresrNode', () => {
  it('compresses state field with latte query from state', async () => {
    const fake = new FakeCompressionClient();
    const node = makeCompresrNode<State>({
      client: fake as FakeAsClient,
      contextKey: 'ctx',
      compressionModel: 'latte_v1',
      queryKey: 'q',
    });
    const out = await node({ ctx: LONG, q: 'find X' });
    expect(out.ctx).toContain('<<C>>');
    expect(fake.calls[0]?.query).toBe('find X');
  });

  it('short input passes through (returns empty patch)', async () => {
    const fake = new FakeCompressionClient();
    const node = makeCompresrNode<State>({
      client: fake as FakeAsClient,
      contextKey: 'ctx',
    });
    const out = await node({ ctx: 'tiny' });
    expect(out).toEqual({});
    expect(fake.calls).toHaveLength(0);
  });

  it('static query overrides state', async () => {
    const fake = new FakeCompressionClient();
    const node = makeCompresrNode<State>({
      client: fake as FakeAsClient,
      contextKey: 'ctx',
      compressionModel: 'latte_v1',
      query: 'FIXED',
      queryKey: 'q',
    });
    await node({ ctx: LONG, q: 'ignored' });
    expect(fake.calls[0]?.query).toBe('FIXED');
  });

  it('queryExtractor callable wins over key', async () => {
    const fake = new FakeCompressionClient();
    const node = makeCompresrNode<State>({
      client: fake as FakeAsClient,
      contextKey: 'ctx',
      compressionModel: 'latte_v1',
      queryExtractor: (state) => `derived:${state.topic ?? 'general'}`,
    });
    await node({ ctx: LONG, topic: 'ml' });
    expect(fake.calls[0]?.query).toBe('derived:ml');
  });

  it('latte without explicit query uses "summarize" fallback', async () => {
    const fake = new FakeCompressionClient();
    const node = makeCompresrNode<State>({
      client: fake as FakeAsClient,
      contextKey: 'ctx',
      compressionModel: 'latte_v1',
      // no query / queryKey / queryExtractor — resolver returns the
      // "summarize" fallback (matches Python behavior).
    });
    const out = await node({ ctx: LONG });
    expect(out.ctx).toContain('<<C>>');
    expect(fake.calls[0]?.query).toBe('summarize');
  });

  it('passthrough on error returns no patch', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const node = makeCompresrNode<State>({
      client: fake as FakeAsClient,
      contextKey: 'ctx',
      onError: 'passthrough',
    });
    const out = await node({ ctx: LONG });
    // compressSafe returned original; node returns empty patch.
    expect(out).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// compresrHandoffTool
// ---------------------------------------------------------------------------

describe('compresrHandoffTool', () => {
  function invokeAsToolCall(t: ReturnType<typeof compresrHandoffTool>, args: Record<string, unknown>, id: string) {
    return t.invoke({ name: t.name, args, type: 'tool_call', id });
  }

  it('has the expected name and description', () => {
    const fake = new FakeCompressionClient();
    const t = compresrHandoffTool('researcher', { client: fake as FakeAsClient });
    expect(t.name).toBe('transfer_to_researcher');
    expect(t.description).toContain('researcher');
  });

  it('compresses long task_description and context, emits Command.PARENT', async () => {
    const fake = new FakeCompressionClient();
    const t = compresrHandoffTool('researcher', {
      client: fake as FakeAsClient,
      minTokens: 10,
    });
    const cmd = (await invokeAsToolCall(
      t,
      { task_description: LONG, context: LONG },
      'tc1'
    )) as Command;
    expect(cmd).toBeInstanceOf(Command);
    // TS Command normalizes goto to an array; accept either form.
    const goto = Array.isArray(cmd.goto) ? cmd.goto[0] : cmd.goto;
    expect(goto).toBe('researcher');
    expect(cmd.graph).toBe(Command.PARENT);
    const update = cmd.update as Record<string, unknown>;
    expect(String(update.task_description)).toContain('<<C>>');
    expect(String(update.context)).toContain('<<C>>');
    expect(fake.calls).toHaveLength(2);
  });

  it('skips compression for short task_description and empty context', async () => {
    const fake = new FakeCompressionClient();
    const t = compresrHandoffTool('writer', {
      client: fake as FakeAsClient,
      minTokens: 10_000,
    });
    const cmd = (await invokeAsToolCall(t, { task_description: 'tiny' }, 'tc2')) as Command;
    expect(cmd).toBeInstanceOf(Command);
    expect(fake.calls).toHaveLength(0);
    expect((cmd.update as Record<string, unknown>).context).toBe('');
  });

  it('passthrough on error keeps original task_description', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const t = compresrHandoffTool('writer', {
      client: fake as FakeAsClient,
      minTokens: 10,
      onError: 'passthrough',
    });
    const cmd = (await invokeAsToolCall(
      t,
      { task_description: LONG },
      'tc3'
    )) as Command;
    expect((cmd.update as Record<string, unknown>).task_description).toBe(LONG);
  });

  it('only task_description (no context) — single compress call', async () => {
    const fake = new FakeCompressionClient();
    const t = compresrHandoffTool('researcher', {
      client: fake as FakeAsClient,
      minTokens: 10,
    });
    const cmd = (await invokeAsToolCall(t, { task_description: LONG }, 'tc4')) as Command;
    expect(fake.calls).toHaveLength(1);
    expect((cmd.update as Record<string, unknown>).context).toBe('');
  });

  it('honours custom description override', () => {
    const fake = new FakeCompressionClient();
    const t = compresrHandoffTool('researcher', {
      client: fake as FakeAsClient,
      description: 'Custom handoff description',
    });
    expect(t.description).toBe('Custom handoff description');
  });
});

// ---------------------------------------------------------------------------
// CompresrStore
// ---------------------------------------------------------------------------

describe('CompresrStore', () => {
  it('put compresses long string fields, leaves short ones alone', async () => {
    const fake = new FakeCompressionClient();
    const inner = new InMemoryStore();
    const store = new CompresrStore(inner, { client: fake as FakeAsClient, minTokens: 10 });
    await store.put(['u1'], 'doc1', { long: LONG, short: 'hi' });
    const item = await inner.get(['u1'], 'doc1');
    expect(item).not.toBeNull();
    const value = item!.value as Record<string, unknown>;
    expect(String(value.long)).toContain('<<C>>');
    expect(value.short).toBe('hi');
    expect(fake.calls).toHaveLength(1);
  });

  it('respects field allowlist', async () => {
    const fake = new FakeCompressionClient();
    const inner = new InMemoryStore();
    const store = new CompresrStore(inner, {
      client: fake as FakeAsClient,
      minTokens: 10,
      fields: new Set(['history']),
    });
    await store.put(['u1'], 'doc1', { history: LONG, other_big: LONG });
    expect(fake.calls).toHaveLength(1);
    const item = await inner.get(['u1'], 'doc1');
    const value = item!.value as Record<string, unknown>;
    expect(String(value.history)).toContain('<<C>>');
    expect(value.other_big).toBe(LONG);
  });

  it('short values pass through without calling backend', async () => {
    const fake = new FakeCompressionClient();
    const inner = new InMemoryStore();
    const store = new CompresrStore(inner, { client: fake as FakeAsClient, minTokens: 10_000 });
    await store.put(['u1'], 'doc1', { val: LONG });
    expect(fake.calls).toHaveLength(0);
    const item = await inner.get(['u1'], 'doc1');
    expect((item!.value as Record<string, unknown>).val).toBe(LONG);
  });

  it('get is passthrough — never decompresses', async () => {
    const fake = new FakeCompressionClient();
    const inner = new InMemoryStore();
    await inner.put(['u1'], 'doc1', { long: LONG });
    const store = new CompresrStore(inner, { client: fake as FakeAsClient, minTokens: 10 });
    const item = await store.get(['u1'], 'doc1');
    expect((item!.value as Record<string, unknown>).long).toBe(LONG);
    expect(fake.calls).toHaveLength(0);
  });

  it('passthrough on error', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const inner = new InMemoryStore();
    const store = new CompresrStore(inner, {
      client: fake as FakeAsClient,
      minTokens: 10,
      onError: 'passthrough',
    });
    await store.put(['u1'], 'doc1', { long: LONG });
    const item = await inner.get(['u1'], 'doc1');
    expect((item!.value as Record<string, unknown>).long).toBe(LONG);
  });

  it('walks nested objects and arrays', async () => {
    const fake = new FakeCompressionClient();
    const inner = new InMemoryStore();
    const store = new CompresrStore(inner, { client: fake as FakeAsClient, minTokens: 10 });
    await store.put(['u1'], 'doc1', { outer: { inner: LONG }, lst: [LONG, 'tiny'] });
    const item = await inner.get(['u1'], 'doc1');
    const value = item!.value as Record<string, unknown>;
    expect(String((value.outer as Record<string, unknown>).inner)).toContain('<<C>>');
    expect(String((value.lst as unknown[])[0])).toContain('<<C>>');
    expect((value.lst as unknown[])[1]).toBe('tiny');
    expect(fake.calls).toHaveLength(2);
  });

  it('batch() rewrites PutOperation entries', async () => {
    const fake = new FakeCompressionClient();
    const inner = new InMemoryStore();
    const store = new CompresrStore(inner, { client: fake as FakeAsClient, minTokens: 10 });
    await store.batch([
      { namespace: ['u1'], key: 'doc1', value: { long: LONG } },
    ]);
    const item = await inner.get(['u1'], 'doc1');
    expect(String((item!.value as Record<string, unknown>).long)).toContain('<<C>>');
    expect(fake.calls).toHaveLength(1);
  });
});
