/**
 * Tests for the LangChain integration (middleware, wrapper, retriever).
 *
 * Mirrors Python `tests/integration/test_langchain.py`. Uses the in-process
 * `FakeCompressionClient` — no network calls.
 */
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { Document } from '@langchain/core/documents';
import { tool } from '@langchain/core/tools';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';

import {
  CompresrExtractor,
  compresrPromptMiddleware,
  compresrSummarizationMiddleware,
  compresrToolMiddleware,
  wrapToolWithCompression,
  compressToolOutput,
} from '../../src/integrations/langchain/index.js';

import { FakeCompressionClient, type FakeAsClient } from './_fake-client.js';

const LONG = 'x '.repeat(2000);

function makeRequest(toolCall: Record<string, unknown>, messages: unknown[] = []) {
  return { toolCall, messages };
}

// ---------------------------------------------------------------------------
// compresrToolMiddleware
// ---------------------------------------------------------------------------

describe('compresrToolMiddleware', () => {
  it('compresses long tool output', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrToolMiddleware({ client: fake as FakeAsClient });
    const req = makeRequest(
      { id: 't1', name: 'search', args: { query: 'find me X' } },
      [new HumanMessage('please find X')]
    );
    const handler = async () =>
      new ToolMessage({ content: LONG, tool_call_id: 't1', name: 'search' });
    const out = await mw.wrapToolCall!(req, handler);
    expect((out as ToolMessage).content as string).toContain('<<C>>');
    expect(fake.calls).toHaveLength(1);
    expect(fake.calls[0]?.query).toBe('find me X');
    expect(fake.calls[0]?.compressionModelName).toBe('latte_v1');
  });

  it('short output passes through', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrToolMiddleware({ client: fake as FakeAsClient });
    const handler = async () =>
      new ToolMessage({ content: 'tiny', tool_call_id: 't1', name: 'search' });
    const out = await mw.wrapToolCall!(makeRequest({ id: 't1', name: 'search' }), handler);
    expect((out as ToolMessage).content).toBe('tiny');
    expect(fake.calls).toHaveLength(0);
  });

  it('allow list excludes other tools', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrToolMiddleware({
      client: fake as FakeAsClient,
      allowTools: ['search'],
    });
    const handler = async () =>
      new ToolMessage({ content: LONG, tool_call_id: 't1', name: 'fetch' });
    const out = await mw.wrapToolCall!(makeRequest({ id: 't1', name: 'fetch' }), handler);
    expect((out as ToolMessage).content as string).toBe(LONG);
    expect(fake.calls).toHaveLength(0);
  });

  it('static query overrides args', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrToolMiddleware({
      client: fake as FakeAsClient,
      query: 'FIXED',
    });
    const handler = async () =>
      new ToolMessage({ content: LONG, tool_call_id: 't1', name: 'search' });
    await mw.wrapToolCall!(
      makeRequest({ id: 't1', name: 'search', args: { query: 'ignored' } }),
      handler
    );
    expect(fake.calls[0]?.query).toBe('FIXED');
  });

  it('queryArg picks named key', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrToolMiddleware({
      client: fake as FakeAsClient,
      queryArg: 'url',
    });
    const handler = async () =>
      new ToolMessage({ content: LONG, tool_call_id: 't1', name: 'fetch' });
    await mw.wrapToolCall!(
      makeRequest({ id: 't1', name: 'fetch', args: { url: 'https://example.com' } }),
      handler
    );
    expect(fake.calls[0]?.query).toBe('https://example.com');
  });

  it('arbitrary model name passes through', async () => {
    // SDK is permissive — backend validates model names.
    const fake = new FakeCompressionClient();
    const mw = compresrToolMiddleware({
      client: fake as FakeAsClient,
      compressionModel: 'future_v3',
    });
    const handler = async () =>
      new ToolMessage({ content: LONG, tool_call_id: 't1', name: 't' });
    await mw.wrapToolCall!(
      makeRequest({ id: 't1', name: 't', args: { query: 'anything' } }),
      handler
    );
    expect(fake.calls[0]?.compressionModelName).toBe('future_v3');
  });

  it('passthrough on error returns original', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const mw = compresrToolMiddleware({
      client: fake as FakeAsClient,
      onError: 'passthrough',
    });
    const handler = async () =>
      new ToolMessage({ content: LONG, tool_call_id: 't1', name: 't' });
    const out = await mw.wrapToolCall!(makeRequest({ id: 't1', name: 't' }), handler);
    expect((out as ToolMessage).content as string).toBe(LONG);
  });

  it('raise policy propagates', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const mw = compresrToolMiddleware({
      client: fake as FakeAsClient,
      onError: 'raise',
    });
    const handler = async () =>
      new ToolMessage({ content: LONG, tool_call_id: 't1', name: 't' });
    await expect(
      mw.wrapToolCall!(makeRequest({ id: 't1', name: 't' }), handler)
    ).rejects.toThrow('forced failure');
  });
});

// ---------------------------------------------------------------------------
// compresrSummarizationMiddleware
// ---------------------------------------------------------------------------

describe('compresrSummarizationMiddleware', () => {
  function buildHistory(pairs: number) {
    const out: unknown[] = [];
    for (let i = 0; i < pairs; i++) {
      out.push(new HumanMessage(`q${i}`));
      out.push(new ToolMessage({ content: LONG, tool_call_id: `t${i}`, name: 'search' }));
    }
    return out;
  }

  it('under threshold is a no-op', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrSummarizationMiddleware({
      client: fake as FakeAsClient,
      maxTokensBeforeSummary: 1_000_000,
      messagesToKeep: 2,
    });
    const out = await mw.beforeModel!({ messages: buildHistory(3) });
    expect(out).toBeUndefined();
    expect(fake.calls).toHaveLength(0);
  });

  it('above threshold summarizes and keeps recent', async () => {
    const fake = new FakeCompressionClient();
    const msgs = buildHistory(10);
    const mw = compresrSummarizationMiddleware({
      client: fake as FakeAsClient,
      maxTokensBeforeSummary: 100,
      messagesToKeep: 4,
    });
    const out = await mw.beforeModel!({ messages: msgs });
    expect(out?.messages).toBeDefined();
    const newMsgs = out!.messages!;
    // First is RemoveMessage marker, second is the summary HumanMessage.
    expect((newMsgs[1] as HumanMessage).content as string).toContain(
      '[Earlier conversation summary]'
    );
    // Recent 4 preserved verbatim.
    expect(newMsgs.slice(-4)).toEqual(msgs.slice(-4));
    // Exactly one compress call.
    expect(fake.calls).toHaveLength(1);
  });

  it('does not nest summary-of-summary on re-run', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrSummarizationMiddleware({
      client: fake as FakeAsClient,
      maxTokensBeforeSummary: 100,
      messagesToKeep: 4,
    });
    const first = await mw.beforeModel!({ messages: buildHistory(10) });
    // Skip RemoveMessage marker, take the rest, then add 2 more turns.
    const rolled = (first!.messages! as unknown[]).slice(1).concat(buildHistory(2));
    const second = await mw.beforeModel!({ messages: rolled });
    const newMsgs = second!.messages!;
    const summary = (newMsgs[1] as HumanMessage).content as string;
    expect(summary).toContain('[Earlier conversation summary]');
    // No nested prefix.
    expect((summary.match(/\[Earlier conversation summary\]/g) ?? []).length).toBe(1);
  });

  it('trigger and keep aliases override long names', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrSummarizationMiddleware({
      client: fake as FakeAsClient,
      trigger: 100,
      keep: 4,
    });
    const out = await mw.beforeModel!({ messages: buildHistory(10) });
    expect(out?.messages).toBeDefined();
    expect(fake.calls).toHaveLength(1);
  });

  it('tokenCounter is used in trigger calculation', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrSummarizationMiddleware({
      client: fake as FakeAsClient,
      trigger: 100,
      keep: 4,
      tokenCounter: () => 1, // never crosses the trigger
    });
    const out = await mw.beforeModel!({ messages: buildHistory(10) });
    expect(out).toBeUndefined();
    expect(fake.calls).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// wrapToolWithCompression
// ---------------------------------------------------------------------------

describe('wrapToolWithCompression', () => {
  function makeSearchTool() {
    return tool(
      async (input: { query: string }) => {
        return LONG;
      },
      {
        name: 'web_search',
        description: 'Search the web.',
        schema: z.object({ query: z.string() }),
      }
    );
  }

  it('preserves name + description + compresses output', async () => {
    const fake = new FakeCompressionClient();
    const wrapped = wrapToolWithCompression(makeSearchTool(), {
      client: fake as FakeAsClient,
      compressionModel: 'latte_v1',
      queryArg: 'query',
    });
    expect(wrapped.name).toBe('web_search');
    expect(wrapped.description).toContain('Search');
    const out = await wrapped.invoke({ query: 'find X' });
    expect(out as string).toContain('<<C>>');
    expect(fake.calls[0]?.query).toBe('find X');
  });

  it('short output passes through', async () => {
    const tinyTool = tool(async () => 'tiny', {
      name: 'tiny',
      description: 'tiny',
      schema: z.object({ query: z.string() }),
    });
    const fake = new FakeCompressionClient();
    const wrapped = wrapToolWithCompression(tinyTool, {
      client: fake as FakeAsClient,
    });
    const out = await wrapped.invoke({ query: 'anything' });
    expect(out).toBe('tiny');
    expect(fake.calls).toHaveLength(0);
  });

  it('decorator form works', async () => {
    const fake = new FakeCompressionClient();
    const wrap = compressToolOutput({ client: fake as FakeAsClient });
    const wrapped = wrap(makeSearchTool());
    const out = await wrapped.invoke({ query: 'x' });
    expect(out as string).toContain('<<C>>');
  });
});

// ---------------------------------------------------------------------------
// CompresrExtractor
// ---------------------------------------------------------------------------

describe('CompresrExtractor', () => {
  it('compresses long docs via batch call', async () => {
    const fake = new FakeCompressionClient();
    const comp = new CompresrExtractor({ client: fake as FakeAsClient });
    const docs = [
      new Document({ pageContent: LONG, metadata: { src: 'a' } }),
      new Document({ pageContent: LONG, metadata: { src: 'b' } }),
    ];
    const out = await comp.compressDocuments(docs, 'my query');
    expect(out).toHaveLength(2);
    out.forEach((d) => {
      expect(d.pageContent).toContain('<<C>>');
      expect(d.metadata.compresr).toBe(true);
    });
    expect(fake.batchCalls).toHaveLength(1);
    expect(fake.batchCalls[0]?.queries).toBe('my query');
    expect(fake.batchCalls[0]?.compressionModelName).toBe('latte_v1');
  });

  it('short docs pass through', async () => {
    const fake = new FakeCompressionClient();
    const comp = new CompresrExtractor({ client: fake as FakeAsClient });
    const docs = [new Document({ pageContent: 'short', metadata: {} })];
    const out = await comp.compressDocuments(docs, 'q');
    expect(out[0]?.pageContent).toBe('short');
    expect(fake.batchCalls).toHaveLength(0);
  });

  it('passthrough on batch failure', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const comp = new CompresrExtractor({
      client: fake as FakeAsClient,
      onError: 'passthrough',
    });
    const docs = [new Document({ pageContent: LONG, metadata: {} })];
    const out = await comp.compressDocuments(docs, 'q');
    expect(out[0]?.pageContent).toBe(LONG);
  });

  it('raise policy propagates', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const comp = new CompresrExtractor({
      client: fake as FakeAsClient,
      onError: 'raise',
    });
    const docs = [new Document({ pageContent: LONG, metadata: {} })];
    await expect(comp.compressDocuments(docs, 'q')).rejects.toThrow('forced failure');
  });

  it('handles 101-doc batch boundary', async () => {
    const fake = new FakeCompressionClient();
    const comp = new CompresrExtractor({ client: fake as FakeAsClient });
    const docs = Array.from(
      { length: 101 },
      (_v, i) => new Document({ pageContent: LONG, metadata: { i } })
    );
    const out = await comp.compressDocuments(docs, 'q');
    expect(out).toHaveLength(101);
    expect(fake.batchCalls).toHaveLength(2);
    expect(fake.batchCalls[0]?.contexts).toHaveLength(100);
    expect(fake.batchCalls[1]?.contexts).toHaveLength(1);
    out.forEach((d) => expect(d.pageContent).toContain('<<C>>'));
  });
});

describe('compresrPromptMiddleware', () => {
  it('passes through when under budget', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrPromptMiddleware({
      client: fake as FakeAsClient,
      maxTokens: 1_000_000,
    });
    const messages = [new HumanMessage({ content: LONG })];
    const request = { messages };
    const out = await mw.wrapModelCall!(request, async (r) => r.messages.length);
    expect(out).toBe(1);
    expect(fake.calls).toHaveLength(0);
  });

  it('compresses largest message first when over budget', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrPromptMiddleware({
      client: fake as FakeAsClient,
      maxTokens: 500,
      minTokens: 50,
      tokenCounter: (s) => s.length,
    });
    const messages = [
      new HumanMessage({ content: 'z'.repeat(1000) }),
      new HumanMessage({ content: 'z'.repeat(1000) }),
      new HumanMessage({ content: 'z'.repeat(1000) }),
    ];
    let captured: unknown[] = [];
    await mw.wrapModelCall!({ messages }, async (r) => {
      captured = r.messages;
      return null;
    });
    expect(fake.calls.length).toBeGreaterThanOrEqual(1);
    const newTotal = captured.reduce(
      (acc, m) => acc + String((m as { content?: unknown }).content ?? '').length,
      0
    );
    expect(newTotal).toBeLessThan(3000);
  });

  it('skips short messages', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrPromptMiddleware({
      client: fake as FakeAsClient,
      maxTokens: 10,
      minTokens: 1000,
      tokenCounter: (s) => s.length,
    });
    const messages = [new HumanMessage({ content: 'hi' })];
    await mw.wrapModelCall!({ messages }, async () => null);
    expect(fake.calls).toHaveLength(0);
  });

  it('passthrough on error keeps original content', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const mw = compresrPromptMiddleware({
      client: fake as FakeAsClient,
      maxTokens: 10,
      minTokens: 10,
      tokenCounter: (s) => s.length,
      onError: 'passthrough',
    });
    const messages = [new HumanMessage({ content: 'z'.repeat(1000) })];
    let captured: unknown[] = [];
    await mw.wrapModelCall!({ messages }, async (r) => {
      captured = r.messages;
      return null;
    });
    const content = String((captured[0] as { content?: unknown }).content ?? '');
    expect(content).toBe('z'.repeat(1000));
  });

  it('handles AI and Tool messages without crashing', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrPromptMiddleware({
      client: fake as FakeAsClient,
      maxTokens: 100,
      minTokens: 50,
      tokenCounter: (s) => s.length,
    });
    const messages = [
      new AIMessage({ content: 'z'.repeat(800) }),
      new ToolMessage({ content: 'z'.repeat(800), tool_call_id: 't1', name: 'x' }),
    ];
    let captured: unknown[] = [];
    await mw.wrapModelCall!({ messages }, async (r) => {
      captured = r.messages;
      return null;
    });
    // Both messages still present, types preserved.
    expect(captured).toHaveLength(2);
    expect(captured[0]).toBeInstanceOf(AIMessage);
    expect(captured[1]).toBeInstanceOf(ToolMessage);
  });

  it('preserves tool_call_id and name on ToolMessage when shrinking', async () => {
    const fake = new FakeCompressionClient();
    const mw = compresrPromptMiddleware({
      client: fake as FakeAsClient,
      maxTokens: 100,
      minTokens: 50,
      tokenCounter: (s) => s.length,
    });
    const tool = new ToolMessage({
      content: 'z'.repeat(800),
      tool_call_id: 'tc-keep',
      name: 'search',
    });
    let captured: unknown[] = [];
    await mw.wrapModelCall!({ messages: [tool] }, async (r) => {
      captured = r.messages;
      return null;
    });
    const out = captured[0] as ToolMessage & { name?: string };
    expect(out.tool_call_id).toBe('tc-keep');
    expect(out.name).toBe('search');
    expect(out).toBeInstanceOf(ToolMessage);
  });
});
