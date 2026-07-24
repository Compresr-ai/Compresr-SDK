/**
 * Unit tests for the agents engine — exercises the parsing/normalization
 * surface that doesn't require a live LangChain ``createAgent`` run.
 */
import { describe, expect, it, vi } from 'vitest';

import { CompresrError } from '../../src/errors/index.js';
import {
  CompresrEngine,
  _setLangChainBindings,
  parseLlmSpec,
} from '../../src/agents/engine.js';
import { FakeCompressionClient, type FakeAsClient } from './_fake-client.js';

describe('parseLlmSpec', () => {
  it('parses colon-form provider:model', () => {
    expect(parseLlmSpec('anthropic:claude-opus-4-8')).toEqual({
      provider: 'anthropic',
      modelName: 'claude-opus-4-8',
    });
  });

  it('parses slash-form provider/model (Vercel AI SDK convention)', () => {
    expect(parseLlmSpec('anthropic/claude-opus-4-8')).toEqual({
      provider: 'anthropic',
      modelName: 'claude-opus-4-8',
    });
  });

  it('handles openai colon form', () => {
    expect(parseLlmSpec('openai:gpt-5')).toEqual({
      provider: 'openai',
      modelName: 'gpt-5',
    });
  });

  it('handles google_genai colon form', () => {
    expect(parseLlmSpec('google_genai:gemini-2.5-pro')).toEqual({
      provider: 'google_genai',
      modelName: 'gemini-2.5-pro',
    });
  });

  it('accepts bare provider with no separator (model deferred to call site)', () => {
    expect(parseLlmSpec('anthropic')).toEqual({
      provider: 'anthropic',
      modelName: undefined,
    });
  });

  it('accepts bare openai provider', () => {
    expect(parseLlmSpec('openai')).toEqual({
      provider: 'openai',
      modelName: undefined,
    });
  });

  it('rejects empty provider', () => {
    expect(() => parseLlmSpec(':claude')).toThrow(CompresrError);
  });

  it('rejects empty model', () => {
    expect(() => parseLlmSpec('anthropic:')).toThrow(CompresrError);
  });

  it('rejects whitespace-only spec', () => {
    expect(() => parseLlmSpec('   ')).toThrow(CompresrError);
  });
});

describe('CompresrEngine', () => {
  it('exposes parsed provider/modelName', () => {
    const fake = new FakeCompressionClient();
    const engine = new CompresrEngine({
      compresrClient: fake as FakeAsClient,
      llm: 'anthropic:claude-haiku-4-5',
    });
    expect(engine.provider).toBe('anthropic');
    expect(engine.modelName).toBe('claude-haiku-4-5');
  });

  it('accepts slash form and normalizes', () => {
    const fake = new FakeCompressionClient();
    const engine = new CompresrEngine({
      compresrClient: fake as FakeAsClient,
      llm: 'openai/gpt-5',
    });
    expect(engine.provider).toBe('openai');
    expect(engine.modelName).toBe('gpt-5');
  });

  it('warns on unknown provider but does not throw', () => {
    const fake = new FakeCompressionClient();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    new CompresrEngine({
      compresrClient: fake as FakeAsClient,
      llm: 'cohere:command',
    });
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it('wires middleware via run() with stubbed langchain', async () => {
    const fake = new FakeCompressionClient();
    const seenMiddleware: unknown[] = [];
    const fakeAgent = {
      invoke: async () => ({
        messages: [
          // synthetic AIMessage-like duck type
          Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
            content: 'hello world',
            tool_calls: [],
            response_metadata: { stop_reason: 'end_turn' },
            usage_metadata: { input_tokens: 4, output_tokens: 2 },
          }),
        ],
      }),
    };
    _setLangChainBindings({
      initChatModel: () => ({ provider: 'fake-chat' }),
      createAgent: (params: Record<string, unknown>) => {
        seenMiddleware.push((params as { middleware?: unknown[] }).middleware);
        return fakeAgent;
      },
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
        llmApiKey: 'sk-ant-test',
        policy: { targetCompressionRatio: 0.7, minTokens: 50 },
      });
      const result = await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(result.text).toBe('hello world');
      expect(result.stopReason).toBe('end_turn');
      expect(result.usage['input_tokens']).toBe(4);
      expect(seenMiddleware).toHaveLength(1);
      const stack = seenMiddleware[0] as Array<{ name: string }>;
      expect(stack[0]?.name).toBe('compresr-tool');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('returns empty normalized result when no AI message present', async () => {
    const fake = new FakeCompressionClient();
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      const result = await engine.run({ messages: [] });
      expect(result.text).toBe('');
      expect(result.stopReason).toBe('end_turn');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('uses call-site model when constructor provided none', async () => {
    const fake = new FakeCompressionClient();
    const seenSpecs: string[] = [];
    _setLangChainBindings({
      initChatModel: (spec: string) => {
        seenSpecs.push(spec);
        return {};
      },
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'ok',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic',
      });
      expect(engine.defaultModelName).toBeUndefined();
      const result = await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'claude-haiku-4-5',
      });
      expect(result.text).toBe('ok');
      expect(seenSpecs).toEqual(['anthropic:claude-haiku-4-5']);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('uses constructor default when call site omits model', async () => {
    const fake = new FakeCompressionClient();
    const seenSpecs: string[] = [];
    _setLangChainBindings({
      initChatModel: (spec: string) => {
        seenSpecs.push(spec);
        return {};
      },
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'ok',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-opus-4-5',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      expect(seenSpecs).toEqual(['anthropic:claude-opus-4-5']);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('call-site model overrides constructor default', async () => {
    const fake = new FakeCompressionClient();
    const seenSpecs: string[] = [];
    _setLangChainBindings({
      initChatModel: (spec: string) => {
        seenSpecs.push(spec);
        return {};
      },
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'ok',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'claude-opus-4-5',
      });
      expect(seenSpecs).toEqual(['anthropic:claude-opus-4-5']);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('throws when neither constructor nor call site supplies a model', async () => {
    const fake = new FakeCompressionClient();
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic',
      });
      await expect(
        engine.run({ messages: [{ role: 'user', content: 'hi' }] })
      ).rejects.toBeInstanceOf(CompresrError);
      await expect(
        engine.run({ messages: [{ role: 'user', content: 'hi' }] })
      ).rejects.toMatchObject({ code: 'missing_model' });
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('caches chat model per name — same model = one initChatModel call', async () => {
    const fake = new FakeCompressionClient();
    let initCalls = 0;
    const seenSpecs: string[] = [];
    _setLangChainBindings({
      initChatModel: (spec: string) => {
        initCalls += 1;
        seenSpecs.push(spec);
        return { spec };
      },
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'ok',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      expect(initCalls).toBe(1);
      // Different model = a fresh init call.
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        model: 'claude-opus-4-5',
      });
      expect(initCalls).toBe(2);
      expect(seenSpecs).toEqual([
        'anthropic:claude-haiku-4-5',
        'anthropic:claude-opus-4-5',
      ]);
      // And the original model is still cached.
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      expect(initCalls).toBe(2);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('CompresrEngine — LLM bind options', () => {
  // Helper: stub ``initChatModel`` with a vi.fn() so tests can assert the
  // kwargs baked into the chat-model constructor. LangChain.js's bind_tools
  // (called internally by create_agent) strips a prior ``chat.bind(...)``'s
  // kwargs, so the engine bakes per-call knobs into ``initChatModel`` instead.
  interface InitChatModelMock {
    initChatModel: ReturnType<typeof vi.fn>;
    chatModelsPassedToAgent: unknown[];
    install: () => void;
  }

  function makeStubbedLangChain(): InitChatModelMock {
    const chatModelsPassedToAgent: unknown[] = [];
    const initChatModel = vi.fn(
      (_spec: string, kwargs?: Record<string, unknown>) => ({
        kind: 'chat',
        kwargs: kwargs ?? {},
      })
    );
    return {
      initChatModel,
      chatModelsPassedToAgent,
      install() {
        _setLangChainBindings({
          initChatModel: initChatModel as unknown as (
            spec: string,
            kwargs?: Record<string, unknown>
          ) => unknown,
          createAgent: (params: Record<string, unknown>) => {
            chatModelsPassedToAgent.push(params['model']);
            return {
              invoke: async () => ({
                messages: [
                  Object.assign(
                    Object.create({ constructor: { name: 'AIMessage' } }),
                    {
                      content: 'ok',
                      tool_calls: [],
                      response_metadata: { stop_reason: 'end_turn' },
                      usage_metadata: {},
                    }
                  ),
                ],
              }),
            };
          },
        });
      },
    };
  }

  it('forwards temperature and topP to initChatModel constructor', async () => {
    const stub = makeStubbedLangChain();
    stub.install();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.5,
        topP: 0.9,
      });
      expect(stub.initChatModel).toHaveBeenCalledTimes(1);
      expect(stub.initChatModel).toHaveBeenCalledWith(
        'anthropic:claude-haiku-4-5',
        expect.objectContaining({ temperature: 0.5, topP: 0.9 })
      );
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('forwards maxTokens to initChatModel constructor', async () => {
    const stub = makeStubbedLangChain();
    stub.install();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        maxTokens: 42,
      });
      expect(stub.initChatModel).toHaveBeenCalledTimes(1);
      expect(stub.initChatModel).toHaveBeenCalledWith(
        'anthropic:claude-haiku-4-5',
        expect.objectContaining({ maxTokens: 42 })
      );
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('aliases maxTokens to maxOutputTokens for google_genai at initChatModel', async () => {
    const stub = makeStubbedLangChain();
    stub.install();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'google_genai:gemini-2.5-flash',
      });
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        maxTokens: 42,
      });
      expect(stub.initChatModel).toHaveBeenCalledTimes(1);
      const [, kwargs] = stub.initChatModel.mock.calls[0]!;
      expect(kwargs).toMatchObject({ maxOutputTokens: 42 });
      // The Gemini-native key MUST replace ``maxTokens`` outright.
      expect(kwargs).not.toHaveProperty('maxTokens');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('drops unknown options before reaching initChatModel', async () => {
    const stub = makeStubbedLangChain();
    stub.install();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.1,
        // Pretend a customer passed something exotic — must not leak.
        widget: true,
      } as unknown as Parameters<typeof engine.run>[0]);
      expect(stub.initChatModel).toHaveBeenCalledTimes(1);
      const [, kwargs] = stub.initChatModel.mock.calls[0]!;
      expect(kwargs).toMatchObject({ temperature: 0.1 });
      expect(kwargs).not.toHaveProperty('widget');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('same options hit cache; different options miss', async () => {
    const stub = makeStubbedLangChain();
    stub.install();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      // Two runs with the same kwargs — only one initChatModel call.
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.5,
      });
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.5,
      });
      expect(stub.initChatModel).toHaveBeenCalledTimes(1);
      // Two more runs with different kwargs — each creates a fresh chat.
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.7,
      });
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.9,
      });
      expect(stub.initChatModel).toHaveBeenCalledTimes(3);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('skips kwargs entirely when no LLM-level options are provided', async () => {
    const stub = makeStubbedLangChain();
    stub.install();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      expect(stub.initChatModel).toHaveBeenCalledTimes(1);
      const [, kwargs] = stub.initChatModel.mock.calls[0]!;
      // When no LLM knobs are passed, the engine should not inject any of
      // them into the chat-model constructor.
      expect(kwargs).not.toHaveProperty('temperature');
      expect(kwargs).not.toHaveProperty('topP');
      expect(kwargs).not.toHaveProperty('maxTokens');
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

// ---------------------------------------------------------------------------
// Usage aggregation — every AIMessage in the conversation contributes
// ---------------------------------------------------------------------------

describe('CompresrEngine — usage aggregation', () => {
  // Duck-type helper: synthesizes an AIMessage-like object the engine
  // recognizes by ``constructor.name``.
  function aiMessage(
    content: string,
    usage: Record<string, unknown> = {}
  ): unknown {
    return Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
      content,
      tool_calls: [],
      response_metadata: { stop_reason: 'end_turn' },
      usage_metadata: usage,
    });
  }

  function humanMessage(content: string): unknown {
    return Object.assign(
      Object.create({ constructor: { name: 'HumanMessage' } }),
      { content }
    );
  }

  function toolMessage(content: string): unknown {
    return Object.assign(
      Object.create({ constructor: { name: 'ToolMessage' } }),
      { content, tool_call_id: 't1' }
    );
  }

  async function runWithMessages(messages: unknown[]) {
    const fake = new FakeCompressionClient();
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages }) }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      return await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
    } finally {
      _setLangChainBindings(undefined);
    }
  }

  it('sums input and output tokens across three AIMessages', async () => {
    const result = await runWithMessages([
      aiMessage('step 1', { input_tokens: 100, output_tokens: 10 }),
      aiMessage('step 2', { input_tokens: 200, output_tokens: 20 }),
      aiMessage('final', { input_tokens: 300, output_tokens: 30 }),
    ]);
    expect(result.usage['input_tokens']).toBe(600);
    expect(result.usage['output_tokens']).toBe(60);
    expect(result.usage['ai_message_count']).toBe(3);
  });

  it('aggregates total_tokens when present', async () => {
    const result = await runWithMessages([
      aiMessage('a', { input_tokens: 10, output_tokens: 2, total_tokens: 12 }),
      aiMessage('b', { input_tokens: 30, output_tokens: 4, total_tokens: 34 }),
    ]);
    expect(result.usage['total_tokens']).toBe(46);
  });

  it('flattens input_token_details cache fields onto top-level keys', async () => {
    const result = await runWithMessages([
      aiMessage('a', {
        input_tokens: 100,
        output_tokens: 10,
        input_token_details: { cache_read: 50, cache_creation: 20 },
      }),
      aiMessage('b', {
        input_tokens: 200,
        output_tokens: 20,
        input_token_details: { cache_read: 70, cache_creation: 30 },
      }),
    ]);
    expect(result.usage['cache_read_input_tokens']).toBe(120);
    expect(result.usage['cache_creation_input_tokens']).toBe(50);
  });

  it('preserves top-level cache_* fields (newer LangChain shape)', async () => {
    const result = await runWithMessages([
      aiMessage('a', {
        input_tokens: 10,
        output_tokens: 2,
        cache_read_input_tokens: 5,
        cache_creation_input_tokens: 3,
      }),
    ]);
    expect(result.usage['cache_read_input_tokens']).toBe(5);
    expect(result.usage['cache_creation_input_tokens']).toBe(3);
  });

  it('top-level cache_* wins over nested input_token_details', async () => {
    // If both shapes are present on the same AIMessage (adapter emitting old
    // + new shape simultaneously), the top-level numbers are authoritative —
    // adding both would double-count what is conceptually the same cache hit.
    const result = await runWithMessages([
      aiMessage('a', {
        input_tokens: 100,
        output_tokens: 5,
        cache_read_input_tokens: 10,
        cache_creation_input_tokens: 4,
        input_token_details: { cache_read: 15, cache_creation: 6 },
      }),
    ]);
    expect(result.usage['cache_read_input_tokens']).toBe(10);
    expect(result.usage['cache_creation_input_tokens']).toBe(4);
  });

  it('skips non-AIMessages (human / tool) when summing', async () => {
    const result = await runWithMessages([
      humanMessage('ask'),
      aiMessage('step', { input_tokens: 100, output_tokens: 5 }),
      toolMessage('search-result'),
      aiMessage('final', { input_tokens: 200, output_tokens: 7 }),
    ]);
    expect(result.usage['input_tokens']).toBe(300);
    expect(result.usage['output_tokens']).toBe(12);
    expect(result.usage['ai_message_count']).toBe(2);
  });

  it('does not count an AIMessage with no usage_metadata', async () => {
    const result = await runWithMessages([
      aiMessage('usage-less', {}),
      aiMessage('real', { input_tokens: 50, output_tokens: 1 }),
    ]);
    expect(result.usage['input_tokens']).toBe(50);
    expect(result.usage['output_tokens']).toBe(1);
    expect(result.usage['ai_message_count']).toBe(1);
  });

  it('returns zeros when no usage_metadata anywhere', async () => {
    const result = await runWithMessages([aiMessage('hi', {})]);
    expect(result.usage['input_tokens']).toBe(0);
    expect(result.usage['output_tokens']).toBe(0);
    expect(result.usage['ai_message_count']).toBe(0);
  });

  it('carries provider-specific numeric extras (e.g. reasoning_tokens)', async () => {
    const result = await runWithMessages([
      aiMessage('a', { input_tokens: 10, output_tokens: 1, reasoning_tokens: 5 }),
      aiMessage('b', { input_tokens: 20, output_tokens: 2, reasoning_tokens: 7 }),
    ]);
    expect(result.usage['reasoning_tokens']).toBe(12);
  });

  it('exposes the full conversation chain on result.messages', async () => {
    const m1 = aiMessage('step 1', {});
    const m2 = aiMessage('step 2', {});
    const m3 = aiMessage('final', {});
    const result = await runWithMessages([m1, m2, m3]);
    expect(result.messages).toHaveLength(3);
    // raw still points at the LAST AIMessage for back-compat.
    expect(result.raw).toBe(m3);
  });

  it('result.messages is the empty array when conversation is empty', async () => {
    const result = await runWithMessages([]);
    expect(result.messages).toEqual([]);
    expect(result.raw).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Solo-WebSearchTool auto-route
// ---------------------------------------------------------------------------

import {
  isSoloWebSearch,
  extractLastUserText,
  WEB_SEARCH_TOOL_NAMES,
} from '../../src/agents/engine.js';

describe('isSoloWebSearch', () => {
  it('detects tavily_search', () => {
    expect(isSoloWebSearch([{ name: 'tavily_search' }])).toBe(true);
  });

  it('detects brave_search', () => {
    expect(isSoloWebSearch([{ name: 'brave_search' }])).toBe(true);
  });

  it('rejects empty', () => {
    expect(isSoloWebSearch([])).toBe(false);
  });

  it('rejects multiple tools', () => {
    expect(
      isSoloWebSearch([{ name: 'tavily_search' }, { name: 'calc' }])
    ).toBe(false);
  });

  it('rejects unknown tool name', () => {
    expect(isSoloWebSearch([{ name: 'calc' }])).toBe(false);
  });

  it('rejects tool with no name', () => {
    expect(isSoloWebSearch([{}])).toBe(false);
  });

  it('exposes a stable name set', () => {
    expect(WEB_SEARCH_TOOL_NAMES.has('tavily_search')).toBe(true);
    expect(WEB_SEARCH_TOOL_NAMES.has('brave_search')).toBe(true);
  });
});

describe('extractLastUserText', () => {
  it('returns the last user message content', () => {
    expect(
      extractLastUserText([
        { role: 'system', content: 'be brief' },
        { role: 'user', content: 'first' },
        { role: 'assistant', content: 'ok' },
        { role: 'user', content: 'second' },
      ])
    ).toBe('second');
  });

  it('joins list-form content blocks', () => {
    expect(
      extractLastUserText([
        {
          role: 'user',
          content: [
            { type: 'text', text: 'hello ' },
            { type: 'text', text: 'world' },
          ],
        },
      ])
    ).toBe('hello world');
  });

  it('returns empty string for empty input', () => {
    expect(extractLastUserText([])).toBe('');
  });
});

// ----- Routing tests ---------------------------------------------------------
//
// Patterns mirror ``tests/unit/agents-research.test.ts``: a ``FakeChat`` whose
// ``bindTools`` returns a stub that pops scripted AIMessage-like responses lets
// us drive the loop without any live LangChain or provider call.

interface RouteFakeMessage {
  constructor: { name: string };
  content: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tool_calls: any[];
  usage_metadata: Record<string, number>;
}

function routeAi(
  content: string,
  toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }> = []
): RouteFakeMessage {
  return {
    constructor: { name: 'AIMessage' },
    content,
    tool_calls: toolCalls,
    usage_metadata: { input_tokens: 5, output_tokens: 2, total_tokens: 7 },
  };
}

class RouteFakeBound {
  constructor(
    private readonly parent: RouteFakeChat,
    public readonly toolChoice: string
  ) {
    this.parent.bindHistory.push({ tool_choice: toolChoice });
  }

  async invoke(messages: unknown[]): Promise<RouteFakeMessage> {
    this.parent.invokeHistory.push([...messages]);
    const next = this.parent.scripted.shift();
    if (next === undefined) {
      throw new Error('FakeChat ran out of scripted responses');
    }
    return next;
  }
}

class RouteFakeChat {
  public readonly scripted: RouteFakeMessage[];
  public readonly bindHistory: Array<{ tool_choice: string }> = [];
  public readonly invokeHistory: unknown[][] = [];

  constructor(scripted: RouteFakeMessage[]) {
    this.scripted = [...scripted];
  }

  bindTools(_tools: unknown[], options: { tool_choice?: string } = {}): RouteFakeBound {
    return new RouteFakeBound(this, options.tool_choice ?? 'auto');
  }
}

function routeTool(): {
  name: string;
  invoke: (args: Record<string, unknown>) => Promise<string>;
} {
  // Long snippet so it clears the default min_compress_tokens=100 threshold.
  const snippet =
    'Compresr is a YC W26-batch startup. Source: https://example.com/compresr\n\n' +
    'paragraph filler content. '.repeat(30);
  return {
    name: 'tavily_search',
    invoke: async () => snippet,
  };
}

describe('CompresrEngine — solo WebSearchTool auto-route', () => {
  function makeFakeAgentInvoke(): {
    invoke: ReturnType<typeof vi.fn>;
  } {
    return { invoke: vi.fn() };
  }

  function setupLangChainStubs(chat: RouteFakeChat, agent: { invoke: ReturnType<typeof vi.fn> }) {
    _setLangChainBindings({
      initChatModel: () => chat,
      createAgent: () => agent,
    });
  }

  it('routes solo tavily_search through the research loop (createAgent never called)', async () => {
    const chat = new RouteFakeChat([
      routeAi('searching', [{ id: 't0', name: 'tavily_search', args: { query: 'q' } }]),
      routeAi(
        'Explanation: e.\nExact Answer: yes\nConfidence: 90\nCitations: https://x.com'
      ),
    ]);
    const agent = makeFakeAgentInvoke();
    setupLangChainStubs(chat, agent);

    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
        llmApiKey: 'sk-ant-test',
      });
      const result = await engine.run({
        messages: [{ role: 'user', content: 'is compresr in YC?' }],
        tools: [routeTool()],
      });

      expect(agent.invoke).not.toHaveBeenCalled();
      expect(result.text).toContain('yes');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('forces tool_choice="none" on the final step at max_steps', async () => {
    const tc = { id: 't0', name: 'tavily_search', args: { query: 'q' } };
    const scripted: RouteFakeMessage[] = [];
    for (let i = 0; i < 10; i++) scripted.push(routeAi('searching', [tc]));
    const chat = new RouteFakeChat(scripted);
    setupLangChainStubs(chat, makeFakeAgentInvoke());

    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
        llmApiKey: 'sk-ant-test',
      });
      await engine.run({
        messages: [{ role: 'user', content: 'q?' }],
        tools: [routeTool()],
      });
      const choices = chat.bindHistory.map((b) => b.tool_choice);
      expect(choices).toHaveLength(10);
      expect(choices[choices.length - 1]).toBe('none');
      expect(choices.slice(0, -1).every((c) => c === 'auto')).toBe(true);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('compresses each snippet with the live tool-call query', async () => {
    const chat = new RouteFakeChat([
      routeAi('searching', [
        { id: 't0', name: 'tavily_search', args: { query: 'yc 2026' } },
      ]),
      routeAi('Explanation: e.\nExact Answer: a\nConfidence: 50'),
    ]);
    setupLangChainStubs(chat, makeFakeAgentInvoke());

    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
        llmApiKey: 'sk-ant-test',
      });
      await engine.run({
        messages: [{ role: 'user', content: 'is compresr in YC?' }],
        tools: [routeTool()],
      });
      expect(fake.calls).toHaveLength(1);
      expect(fake.calls[0]?.query).toBe('yc 2026');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('falls back to createAgent path when more than one tool is supplied', async () => {
    const chat = new RouteFakeChat([]);
    const agent = {
      invoke: vi.fn(async () => ({
        messages: [
          Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
            content: 'standard path',
            tool_calls: [],
            response_metadata: { stop_reason: 'end_turn' },
            usage_metadata: {},
          }),
        ],
      })),
    };
    setupLangChainStubs(chat, agent);

    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
        llmApiKey: 'sk-ant-test',
      });
      const result = await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        tools: [{ name: 'tavily_search' }, { name: 'calculator' }],
      });
      expect(agent.invoke).toHaveBeenCalledOnce();
      expect(result.text).toBe('standard path');
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});
