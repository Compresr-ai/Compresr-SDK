/**
 * Unit tests for the research agent (TS mirror of Python's test_agents_research.py).
 */

import { describe, expect, it } from 'vitest';

import { ResearchAgent, parseResearchOutput } from '../../src/agents/research/index.js';
import { ResearchFacade } from '../../src/agents/research/facade.js';
import { FakeCompressionClient } from './_fake-client.js';

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

describe('parseResearchOutput', () => {
  it('parses all four fields', () => {
    const out = parseResearchOutput(
      [
        'Explanation: I searched a lot.',
        'Exact Answer: SAS 9.1',
        'Confidence: 62%',
        'Citations: https://example.com/a, https://example.com/b',
      ].join('\n')
    );
    expect(out.answer).toBe('SAS 9.1');
    expect(out.explanation).toBe('I searched a lot.');
    expect(out.confidence).toBeCloseTo(0.62);
    expect(out.citationUrls).toEqual([
      'https://example.com/a',
      'https://example.com/b',
    ]);
  });

  it('parses bare-number confidence', () => {
    expect(parseResearchOutput('Confidence: 80').confidence).toBeCloseTo(0.8);
  });

  it('parses fractional confidence', () => {
    expect(parseResearchOutput('Confidence: 0.42').confidence).toBeCloseTo(0.42);
  });

  it('returns empties when no fields found', () => {
    const out = parseResearchOutput('just plain text');
    expect(out.answer).toBe('');
    expect(out.explanation).toBe('');
    expect(out.confidence).toBeNull();
    expect(out.citationUrls).toEqual([]);
  });

  it('dedupes citation URLs', () => {
    const out = parseResearchOutput(
      'Confidence: 50\nCitations: https://a.com, https://a.com, https://b.com'
    );
    expect(out.citationUrls).toEqual(['https://a.com', 'https://b.com']);
  });

  it('handles empty input', () => {
    const out = parseResearchOutput('');
    expect(out.answer).toBe('');
    expect(out.confidence).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// ResearchAgent loop
// ---------------------------------------------------------------------------

class FakeBound {
  constructor(
    private readonly parent: FakeChat,
    public readonly toolChoice: string
  ) {
    this.parent.bindHistory.push({ tool_choice: toolChoice });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async invoke(messages: unknown[]): Promise<any> {
    this.parent.invokeHistory.push([...messages]);
    const next = this.parent.scripted.shift();
    if (next === undefined) {
      throw new Error('FakeChat ran out of scripted responses');
    }
    return next;
  }
}

class FakeChat {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public readonly scripted: any[];
  public readonly bindHistory: Array<{ tool_choice: string }> = [];
  public readonly invokeHistory: unknown[][] = [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(scripted: any[]) {
    this.scripted = [...scripted];
  }

  bindTools(_tools: unknown[], options: { tool_choice?: string } = {}): FakeBound {
    return new FakeBound(this, options.tool_choice ?? 'auto');
  }
}

function ai(content: string, toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }> = []): unknown {
  // Mimic a LangChain AIMessage well enough for ResearchAgent's introspection
  return {
    constructor: { name: 'AIMessage' },
    content,
    tool_calls: toolCalls,
    usage_metadata: { input_tokens: 5, output_tokens: 2, total_tokens: 7 },
  };
}

function makeFakeEngine(
  chat: FakeChat,
  client: FakeCompressionClient,
  overrides: Record<string, unknown> = {}
): unknown {
  return {
    _resolveModel: (_model?: string): string => 'claude-sonnet-4-6',
    _getChatModel: async (_modelName: string) => chat,
    _compresrClient: client,
    defaultModelName: 'claude-sonnet-4-6',
    provider: 'anthropic',
    _promptCacheConfig: {
      enabled: true,
      ttl: '5m' as const,
      minMessages: 2,
    },
    ...overrides,
  };
}

const fakeSearchTool = {
  name: 'fake_search',
  invoke: async (args: Record<string, unknown>): Promise<string> => {
    const q = String(args['query'] ?? '');
    return `raw-result for ${q} https://example.com/${q}`;
  },
};

describe('ResearchAgent', () => {
  it('returns final answer when model emits no tool_calls', async () => {
    const chat = new FakeChat([
      ai('Explanation: trivial\nExact Answer: 42\nConfidence: 99\nCitations: https://a.com'),
    ]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client),
      searchTool: fakeSearchTool,
      maxSteps: 4,
      compressSnippets: false,
    });
    const result = await agent.run('what?');
    expect(result.answer).toBe('42');
    expect(result.confidence).toBeCloseTo(0.99);
    expect(result.citations.map((c) => c.url)).toContain('https://a.com');
    expect(chat.bindHistory[0]?.tool_choice).toBe('auto');
  });

  it('forces tool_choice="none" on the last step', async () => {
    const tc = { id: 't0', name: 'fake_search', args: { query: 'q' } };
    const chat = new FakeChat([
      ai('', [tc]),
      ai('', [tc]),
      ai('', [tc]),
      ai('Explanation: done\nExact Answer: x\nConfidence: 1'),
    ]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client),
      searchTool: fakeSearchTool,
      maxSteps: 4,
      compressSnippets: false,
    });
    await agent.run('q?');
    expect(chat.bindHistory.map((h) => h.tool_choice)).toEqual([
      'auto',
      'auto',
      'auto',
      'none',
    ]);
  });

  it('compresses tool results via the engine client', async () => {
    const tc = { id: 't0', name: 'fake_search', args: { query: 'longq' } };
    const chat = new FakeChat([
      ai('', [tc]),
      ai('Explanation: e\nExact Answer: a\nConfidence: 10'),
    ]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client),
      searchTool: fakeSearchTool,
      maxSteps: 2,
      compressSnippets: true,
      minCompressTokens: 1,
    });
    await agent.run('hi');
    expect(client.calls.length).toBe(1);
    expect(client.calls[0]?.query).toBe('longq');
  });

  it('skips compression when disabled', async () => {
    const tc = { id: 't0', name: 'fake_search', args: { query: 'q' } };
    const chat = new FakeChat([
      ai('', [tc]),
      ai('Explanation: e\nExact Answer: a\nConfidence: 1'),
    ]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client),
      searchTool: fakeSearchTool,
      maxSteps: 2,
      compressSnippets: false,
    });
    await agent.run('hi');
    expect(client.calls.length).toBe(0);
  });

  it('subtracts cache tokens from input_tokens in aggregated usage', async () => {
    const aiWithCache = {
      constructor: { name: 'AIMessage' },
      content: 'Exact Answer: y\nConfidence: 1',
      tool_calls: [],
      usage_metadata: {
        input_tokens: 1000,
        output_tokens: 200,
        total_tokens: 1200,
        input_token_details: {
          cache_read: 600,
          ephemeral_5m_input_tokens: 200,
        },
      },
    };
    const chat = new FakeChat([aiWithCache]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client),
      searchTool: fakeSearchTool,
      maxSteps: 2,
      compressSnippets: false,
    });
    const result = await agent.run('hi');
    expect(result.usage.cache_read_tokens).toBe(600);
    expect(result.usage.cache_creation_tokens).toBe(200);
    // 1000 - 600 - 200 = 200 fresh
    expect(result.usage.input_tokens).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// Facade
// ---------------------------------------------------------------------------

describe('ResearchAgent.applyCacheControl', () => {
  it('stamps ephemeral cache_control on the last message for Anthropic', async () => {
    const chat = new FakeChat([
      ai('Exact Answer: x\nConfidence: 1'),
    ]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client),
      searchTool: fakeSearchTool,
      maxSteps: 1,
      compressSnippets: false,
    });
    await agent.run('q?');
    const lastInvokeMessages = chat.invokeHistory[chat.invokeHistory.length - 1]!;
    const last = lastInvokeMessages[lastInvokeMessages.length - 1] as {
      content?: unknown;
    };
    expect(Array.isArray(last.content)).toBe(true);
    const tail = (last.content as Array<Record<string, unknown>>).at(-1);
    expect(tail?.['cache_control']).toEqual({ type: 'ephemeral', ttl: '5m' });
  });

  it('skips cache_control for non-Anthropic providers', async () => {
    const chat = new FakeChat([
      ai('Exact Answer: x\nConfidence: 1'),
    ]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client, { provider: 'openai' }),
      searchTool: fakeSearchTool,
      maxSteps: 1,
      compressSnippets: false,
    });
    await agent.run('q?');
    const lastInvokeMessages = chat.invokeHistory[chat.invokeHistory.length - 1]!;
    const last = lastInvokeMessages[lastInvokeMessages.length - 1] as {
      content?: unknown;
    };
    expect(typeof last.content).toBe('string');
  });

  it('skips when enablePromptCache=false', async () => {
    const chat = new FakeChat([
      ai('Exact Answer: x\nConfidence: 1'),
    ]);
    const client = new FakeCompressionClient();
    const agent = new ResearchAgent({
      engine: makeFakeEngine(chat, client, {
        _promptCacheConfig: { enabled: false, ttl: '5m', minMessages: 2 },
      }),
      searchTool: fakeSearchTool,
      maxSteps: 1,
      compressSnippets: false,
    });
    await agent.run('q?');
    const lastInvokeMessages = chat.invokeHistory[chat.invokeHistory.length - 1]!;
    const last = lastInvokeMessages[lastInvokeMessages.length - 1] as {
      content?: unknown;
    };
    expect(typeof last.content).toBe('string');
  });
});

describe('ResearchFacade', () => {
  it('rejects unknown search provider', async () => {
    const facade = new ResearchFacade({});
    await expect(facade.run('q?', { search: 'duckduckgo-foo' })).rejects.toThrow(
      /unsupported search provider/
    );
  });
});
