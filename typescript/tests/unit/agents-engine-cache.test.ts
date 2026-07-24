/**
 * Unit tests for CompresrEngine prompt-caching wiring.
 *
 * Mocks `langchain`'s `anthropicPromptCachingMiddleware` so we can assert
 * shape and ordering without touching the network.
 */

import { describe, expect, it, vi } from 'vitest';

import { FakeCompressionClient } from './_fake-client.js';

vi.mock('langchain', async () => {
  return {
    anthropicPromptCachingMiddleware: (opts: Record<string, unknown>) => ({
      __kind: 'fake-anthropic-cache',
      name: 'anthropic-prompt-caching',
      opts,
    }),
  };
});

const {
  CompresrEngine,
  _setLangChainBindings,
} = await import('../../src/agents/engine.js');

type FakeAsClient = unknown;

function makeFakeAgent(): { invoke: () => Promise<unknown> } {
  return {
    invoke: async () => ({
      messages: [
        Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
          content: 'ok',
          tool_calls: [],
          response_metadata: { stop_reason: 'end_turn' },
          usage_metadata: { input_tokens: 1, output_tokens: 1 },
        }),
      ],
    }),
  };
}

describe('CompresrEngine — Anthropic prompt cache wiring', () => {
  it('appends cache middleware after compresr middleware (default)', async () => {
    const fake = new FakeCompressionClient();
    let seen: Array<{ name: string; __kind?: string; opts?: unknown }> = [];
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: (params: Record<string, unknown>) => {
        seen = params['middleware'] as typeof seen;
        return makeFakeAgent();
      },
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-sonnet-4-6',
        llmApiKey: 'sk-ant-test',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      expect(seen).toHaveLength(2);
      expect(seen[0]?.name).toBe('compresr-tool');
      expect(seen[1]?.__kind).toBe('fake-anthropic-cache');
      expect(seen[1]?.opts).toMatchObject({
        ttl: '5m',
        minMessagesToCache: 2,
        unsupportedModelBehavior: 'ignore',
      });
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('can be disabled', async () => {
    const fake = new FakeCompressionClient();
    let seen: Array<{ name: string; __kind?: string }> = [];
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: (params: Record<string, unknown>) => {
        seen = params['middleware'] as typeof seen;
        return makeFakeAgent();
      },
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-sonnet-4-6',
        llmApiKey: 'sk-ant-test',
        enablePromptCache: false,
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      expect(seen).toHaveLength(1);
      expect(seen[0]?.name).toBe('compresr-tool');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('forwards ttl and minMessages to cache middleware', async () => {
    const fake = new FakeCompressionClient();
    let seen: Array<{ name: string; opts?: Record<string, unknown> }> = [];
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: (params: Record<string, unknown>) => {
        seen = params['middleware'] as typeof seen;
        return makeFakeAgent();
      },
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-sonnet-4-6',
        llmApiKey: 'sk-ant-test',
        promptCacheTtl: '1h',
        promptCacheMinMessages: 5,
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      const cacheMw = seen.find((m) => m.opts !== undefined);
      expect(cacheMw?.opts?.ttl).toBe('1h');
      expect(cacheMw?.opts?.minMessagesToCache).toBe(5);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('does not attach Anthropic middleware for openai provider', async () => {
    const fake = new FakeCompressionClient();
    let seen: Array<{ name: string; __kind?: string }> = [];
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: (params: Record<string, unknown>) => {
        seen = params['middleware'] as typeof seen;
        return makeFakeAgent();
      },
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-4.1',
        llmApiKey: 'sk-test',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
      expect(seen).toHaveLength(1);
      expect(seen[0]?.__kind).toBeUndefined();
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('sorts tools by name for cache stability', async () => {
    const fake = new FakeCompressionClient();
    let toolsSeen: unknown[] = [];
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: (params: Record<string, unknown>) => {
        toolsSeen = params['tools'] as unknown[];
        return makeFakeAgent();
      },
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-sonnet-4-6',
        llmApiKey: 'sk-ant-test',
        enablePromptCache: false,
      });
      const tools = [
        { name: 'zebra' },
        { name: 'alpha' },
        { name: 'middle' },
      ];
      await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
        tools,
      });
      expect(toolsSeen.map((t) => (t as { name?: string }).name)).toEqual([
        'alpha',
        'middle',
        'zebra',
      ]);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('CompresrEngine — langchain-anthropic inflated input', () => {
  it('subtracts cache_read + ephemeral writes from input_tokens', async () => {
    const fake = new FakeCompressionClient();
    const agentMsg = Object.assign(
      Object.create({ constructor: { name: 'AIMessage' } }),
      {
        content: 'ok',
        tool_calls: [],
        usage_metadata: {
          input_tokens: 197_694,
          output_tokens: 2_128,
          total_tokens: 199_822,
          input_token_details: {
            cache_read: 159_233,
            cache_creation: 0,
            ephemeral_5m_input_tokens: 30_000,
            ephemeral_1h_input_tokens: 0,
          },
        },
      }
    );
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({
        invoke: async () => ({ messages: [agentMsg] }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-sonnet-4-6',
        llmApiKey: 'sk-ant-test',
      });
      const result = await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(result.usage['input_tokens']).toBe(8_461);
      expect(result.usage['cache_read_input_tokens']).toBe(159_233);
      expect(result.usage['cache_creation_input_tokens']).toBe(30_000);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('CompresrEngine — OpenAI cache wiring', () => {
  it('passes prompt_cache_key into model_kwargs (modelKwargs)', async () => {
    const fake = new FakeCompressionClient();
    let seenInit: Record<string, unknown> | undefined;
    _setLangChainBindings({
      initChatModel: (_spec: string, kw?: Record<string, unknown>) => {
        seenInit = kw;
        return {};
      },
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'ok',
              tool_calls: [],
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-4.1',
        llmApiKey: 'sk-x',
        openaiPromptCacheKey: 'tenant-42',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
    } finally {
      _setLangChainBindings(undefined);
    }
    const mk = (seenInit?.['modelKwargs'] ?? {}) as Record<string, unknown>;
    expect(mk['prompt_cache_key']).toBe('tenant-42');
    expect(mk['prompt_cache_retention']).toBeUndefined();
  });

  it('maps prompt_cache_ttl="1h" to prompt_cache_retention="24h"', async () => {
    const fake = new FakeCompressionClient();
    let seenInit: Record<string, unknown> | undefined;
    _setLangChainBindings({
      initChatModel: (_spec: string, kw?: Record<string, unknown>) => {
        seenInit = kw;
        return {};
      },
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'ok',
              tool_calls: [],
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-4.1',
        llmApiKey: 'sk-x',
        promptCacheTtl: '1h',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
    } finally {
      _setLangChainBindings(undefined);
    }
    const mk = (seenInit?.['modelKwargs'] ?? {}) as Record<string, unknown>;
    expect(mk['prompt_cache_retention']).toBe('24h');
  });

  it('skips OpenAI cache plumbing when enablePromptCache=false', async () => {
    const fake = new FakeCompressionClient();
    let seenInit: Record<string, unknown> | undefined;
    _setLangChainBindings({
      initChatModel: (_spec: string, kw?: Record<string, unknown>) => {
        seenInit = kw;
        return {};
      },
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'ok',
              tool_calls: [],
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-4.1',
        llmApiKey: 'sk-x',
        enablePromptCache: false,
        openaiPromptCacheKey: 'tenant-42',
        promptCacheTtl: '1h',
      });
      await engine.run({ messages: [{ role: 'user', content: 'hi' }] });
    } finally {
      _setLangChainBindings(undefined);
    }
    const mk = (seenInit?.['modelKwargs'] ?? {}) as Record<string, unknown>;
    expect(mk['prompt_cache_key']).toBeUndefined();
    expect(mk['prompt_cache_retention']).toBeUndefined();
  });
});

describe('CompresrEngine — Gemini cached_content_token_count', () => {
  it('aggregates Gemini cached_content_token_count into cache_read', async () => {
    const fake = new FakeCompressionClient();
    const msg = Object.assign(
      Object.create({ constructor: { name: 'AIMessage' } }),
      {
        content: 'ok',
        tool_calls: [],
        usage_metadata: {
          input_tokens: 1500,
          output_tokens: 80,
          total_tokens: 1580,
          input_token_details: { cached_content_token_count: 1200 },
        },
      }
    );
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [msg] }) }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'google_genai:gemini-2.5-pro',
        llmApiKey: 'AIza-test',
      });
      const result = await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(result.usage['cache_read_input_tokens']).toBe(1200);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('CompresrEngine — usage aggregation extras', () => {
  it('aggregates OpenAI cached_tokens / priority_cache_read / flex_cache_read', async () => {
    const fake = new FakeCompressionClient();
    const agentMsg = (details: Record<string, number>): unknown =>
      Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
        content: 'ok',
        tool_calls: [],
        usage_metadata: {
          input_tokens: 10,
          output_tokens: 2,
          total_tokens: 12,
          input_token_details: details,
        },
      });
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            agentMsg({ cached_tokens: 100 }),
            agentMsg({ priority_cache_read: 50, flex_cache_read: 25 }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-4.1',
        llmApiKey: 'sk-test',
      });
      const result = await engine.run({
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(result.usage['cache_read_input_tokens']).toBe(175);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});
