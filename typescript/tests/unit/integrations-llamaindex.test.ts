/**
 * Tests for the LlamaIndex.TS integration.
 *
 * LlamaIndex.TS schemas use `TextNode` + `NodeWithScore`. We construct
 * lightweight stand-ins to avoid spinning up the full vector-store machinery.
 */
import { describe, expect, it } from 'vitest';

import {
  CompresrMemoryBlock,
  CompresrNodePostprocessor,
  wrapToolWithCompresr,
} from '../../src/integrations/llamaindex/index.js';
import { FakeCompressionClient, type FakeAsClient } from './_fake-client.js';

const LONG = 'x '.repeat(2000);

interface MutableNode {
  text: string;
  metadata: Record<string, unknown>;
  getContent(): string;
  setContent(t: string): void;
}

function makeNodeWithScore(text: string, score = 1): { node: MutableNode; score: number } {
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

// ---------------------------------------------------------------------------
// CompresrNodePostprocessor
// ---------------------------------------------------------------------------

describe('CompresrNodePostprocessor', () => {
  it('compresses long nodes with QueryBundle-style input', async () => {
    const fake = new FakeCompressionClient();
    const pp = new CompresrNodePostprocessor({ client: fake as FakeAsClient });
    const nodes = [makeNodeWithScore(LONG), makeNodeWithScore(LONG)] as never;
    const out = await pp.postprocessNodes(nodes, { queryStr: 'find X' });
    expect(out).toHaveLength(2);
    out.forEach((n: { node: MutableNode }) => {
      expect(n.node.getContent()).toContain('<<C>>');
    });
    expect(fake.batchCalls).toHaveLength(1);
    expect(fake.batchCalls[0]?.queries).toBe('find X');
    expect(fake.batchCalls[0]?.compressionModelName).toBe('latte_v1');
  });

  it('accepts a raw query string', async () => {
    const fake = new FakeCompressionClient();
    const pp = new CompresrNodePostprocessor({ client: fake as FakeAsClient });
    const nodes = [makeNodeWithScore(LONG)] as never;
    await pp.postprocessNodes(nodes, 'find X');
    expect(fake.batchCalls[0]?.queries).toBe('find X');
  });

  it('targetToken overrides targetCompressionRatio', async () => {
    const fake = new FakeCompressionClient();
    const pp = new CompresrNodePostprocessor({
      client: fake as FakeAsClient,
      targetToken: 100,
      targetCompressionRatio: 0.5,
    });
    const nodes = [makeNodeWithScore(LONG), makeNodeWithScore(LONG)] as never;
    await pp.postprocessNodes(nodes, 'q');
    // LONG = "x " * 2000 → 4000 chars → estimateTokens fallback = 1000.
    // ratio = 1000 / 100 = 10.0
    const ratio = fake.batchCalls[0]?.targetCompressionRatio;
    expect(ratio).toBeGreaterThan(1.0);
    expect(ratio).toBe(1000 / 100);
  });

  it('short nodes pass through', async () => {
    const fake = new FakeCompressionClient();
    const pp = new CompresrNodePostprocessor({ client: fake as FakeAsClient });
    const nodes = [makeNodeWithScore('tiny')] as never;
    const out = await pp.postprocessNodes(nodes, 'q');
    expect(out[0]?.node.getContent()).toBe('tiny');
    expect(fake.batchCalls).toHaveLength(0);
  });

  it('latte with no query skips compression', async () => {
    const fake = new FakeCompressionClient();
    const pp = new CompresrNodePostprocessor({ client: fake as FakeAsClient });
    const nodes = [makeNodeWithScore(LONG)] as never;
    const out = await pp.postprocessNodes(nodes);
    expect(out[0]?.node.getContent()).toBe(LONG);
    expect(fake.batchCalls).toHaveLength(0);
  });


  it('static query override', async () => {
    const fake = new FakeCompressionClient();
    const pp = new CompresrNodePostprocessor({
      client: fake as FakeAsClient,
      query: 'STATIC',
    });
    const nodes = [makeNodeWithScore(LONG)] as never;
    await pp.postprocessNodes(nodes, 'ignored');
    expect(fake.batchCalls[0]?.queries).toBe('STATIC');
  });

  it('passthrough on batch failure', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const pp = new CompresrNodePostprocessor({
      client: fake as FakeAsClient,
      onError: 'passthrough',
    });
    const nodes = [makeNodeWithScore(LONG)] as never;
    const out = await pp.postprocessNodes(nodes, 'q');
    expect(out[0]?.node.getContent()).toBe(LONG);
  });

  it('handles 101-node batch boundary', async () => {
    const fake = new FakeCompressionClient();
    const pp = new CompresrNodePostprocessor({ client: fake as FakeAsClient });
    const nodes = Array.from({ length: 101 }, () => makeNodeWithScore(LONG)) as never;
    const out = await pp.postprocessNodes(nodes, 'q');
    expect(out).toHaveLength(101);
    expect(fake.batchCalls).toHaveLength(2);
    expect(fake.batchCalls[0]?.contexts).toHaveLength(100);
    expect(fake.batchCalls[1]?.contexts).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// wrapToolWithCompresr
// ---------------------------------------------------------------------------

describe('wrapToolWithCompresr', () => {
  it('preserves metadata + compresses output', async () => {
    const fake = new FakeCompressionClient();
    const tool = {
      metadata: { name: 'search', description: 'search' },
      async call(args: { query: string }) {
        void args;
        return LONG;
      },
    };
    const wrapped = wrapToolWithCompresr(tool, {
      client: fake as FakeAsClient,
      compressionModel: 'latte_v1',
      queryArg: 'query',
    });
    expect(wrapped.metadata.name).toBe('search');
    const out = await wrapped.call({ query: 'find X' });
    expect(out as string).toContain('<<C>>');
    expect(fake.calls[0]?.query).toBe('find X');
  });

  it('short output passes through', async () => {
    const fake = new FakeCompressionClient();
    const tool = {
      metadata: { name: 's', description: 's' },
      async call(_args: { query: string }) {
        return 'tiny';
      },
    };
    const wrapped = wrapToolWithCompresr(tool, { client: fake as FakeAsClient });
    const out = await wrapped.call({ query: 'x' });
    expect(out).toBe('tiny');
    expect(fake.calls).toHaveLength(0);
  });
});

describe('CompresrMemoryBlock', () => {
  it('put accumulates buffer with role: content lines', async () => {
    const fake = new FakeCompressionClient();
    const block = new CompresrMemoryBlock({ client: fake as FakeAsClient, minTokens: 10 });
    await block.put([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi there' },
    ]);
    const msgs = await block.get();
    // short buffer → no compression
    expect(msgs).toHaveLength(1);
    expect(String(msgs[0]?.content)).toContain('user: hello');
    expect(String(msgs[0]?.content)).toContain('assistant: hi there');
    expect(fake.calls).toHaveLength(0);
  });

  it('get compresses when buffer exceeds minTokens', async () => {
    const fake = new FakeCompressionClient();
    const block = new CompresrMemoryBlock({
      client: fake as FakeAsClient,
      minTokens: 10,
      targetToken: 100,
    });
    await block.put([{ role: 'user', content: LONG }]);
    const msgs = await block.get();
    expect(String(msgs[0]?.content)).toContain('<<C>>');
    expect(fake.calls).toHaveLength(1);
    expect(fake.calls[0]!.targetCompressionRatio).toBeGreaterThanOrEqual(10);
  });

  it('get skips compression for short buffer', async () => {
    const fake = new FakeCompressionClient();
    const block = new CompresrMemoryBlock({ client: fake as FakeAsClient, minTokens: 10_000 });
    await block.put([{ role: 'user', content: LONG }]);
    const msgs = await block.get();
    expect(String(msgs[0]?.content)).toContain('user:');
    expect(fake.calls).toHaveLength(0);
  });

  it('passthrough on error', async () => {
    const fake = new FakeCompressionClient({ raiseOnCall: true });
    const block = new CompresrMemoryBlock({
      client: fake as FakeAsClient,
      minTokens: 10,
      targetToken: 100,
      onError: 'passthrough',
    });
    await block.put([{ role: 'user', content: LONG }]);
    const msgs = await block.get();
    // Buffer is `user: ${LONG}` trimmed — assert against the trimmed form.
    expect(String(msgs[0]?.content)).toContain(LONG.trim());
  });

  it('get returns [] when buffer is empty', async () => {
    const fake = new FakeCompressionClient();
    const block = new CompresrMemoryBlock({ client: fake as FakeAsClient, minTokens: 10 });
    const msgs = await block.get();
    expect(msgs).toEqual([]);
    expect(fake.calls).toHaveLength(0);
  });

  it('put skips messages with empty or non-string content', async () => {
    const fake = new FakeCompressionClient();
    const block = new CompresrMemoryBlock({ client: fake as FakeAsClient, minTokens: 10 });
    await block.put([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: '' },
      { role: 'system', content: { unsupported: true } },
    ]);
    const msgs = await block.get();
    const content = String(msgs[0]?.content ?? '');
    expect(content).toContain('user: hello');
    expect(content).not.toContain('assistant:');
    expect(content).not.toContain('system:');
  });
});
