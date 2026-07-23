/**
 * Tests for ``CompressionClient``'s opt-in agent surface:
 *   - constructing without ``llm`` keeps ``.compress`` working and throws a
 *     clear ``CompresrError`` when ``.messages`` / ``.chat`` / ``.run`` are
 *     touched;
 *   - constructing with ``llm`` returns the facades and routes ``run`` through
 *     the engine.
 */
import { describe, expect, it, vi } from 'vitest';

import { CompressionClient } from '../../src/clients/compression.js';
import { CompresrError } from '../../src/errors/index.js';
import {
  CompresrEngine,
  _setLangChainBindings,
} from '../../src/agents/engine.js';

describe('CompressionClient — no llm', () => {
  it('throws CompresrError when .messages is accessed', () => {
    const client = new CompressionClient({ apiKey: 'cmp_test' });
    expect(() => client.messages).toThrow(CompresrError);
  });

  it('throws CompresrError when .chat is accessed', () => {
    const client = new CompressionClient({ apiKey: 'cmp_test' });
    expect(() => client.chat).toThrow(CompresrError);
  });

  it('throws CompresrError when run() is called', async () => {
    const client = new CompressionClient({ apiKey: 'cmp_test' });
    await expect(client.run({ prompt: 'hi' })).rejects.toBeInstanceOf(
      CompresrError
    );
  });

  it('error message includes migration hint', () => {
    const client = new CompressionClient({ apiKey: 'cmp_test' });
    try {
      void client.messages;
    } catch (exc) {
      expect((exc as Error).message).toContain('requires an LLM provider');
      expect((exc as Error).message).toContain('llm:');
    }
  });
});

describe('CompressionClient — with llm', () => {
  it('lazily exposes the Anthropic facade', () => {
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'anthropic:claude-haiku-4-5',
        llmApiKey: 'sk-ant-x',
      });
      const messages = client.messages;
      expect(typeof messages.create).toBe('function');
      expect(typeof messages.stream).toBe('function');
      // Singleton — repeated access returns the same object.
      expect(client.messages).toBe(messages);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('lazily exposes the OpenAI chat facade', () => {
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'openai:gpt-5',
        llmApiKey: 'sk-openai-x',
      });
      const chat = client.chat;
      expect(typeof chat.completions.create).toBe('function');
      expect(typeof chat.completions.stream).toBe('function');
      expect(client.chat).toBe(chat);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('run() returns a NormalizedResult via the engine', async () => {
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'done',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: { input_tokens: 3, output_tokens: 1 },
            }),
          ],
        }),
      }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'anthropic:claude-haiku-4-5',
      });
      const result = await client.run({ prompt: 'hi' });
      expect(result.text).toBe('done');
      expect(result.usage['input_tokens']).toBe(3);
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('compression policy is forwarded to the engine', () => {
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'anthropic:claude-haiku-4-5',
        compression: { targetCompressionRatio: 0.8, minTokens: 1000 },
      });
      // Access .messages to trigger engine construction — should not throw.
      expect(() => client.messages).not.toThrow();
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('CompressionClient — .compress remains independent', () => {
  it('constructor still accepts apiKey only', () => {
    const client = new CompressionClient({ apiKey: 'cmp_test' });
    expect(typeof client.compress).toBe('function');
    expect(typeof client.compressBatch).toBe('function');
  });
});

describe('CompressionClient — provider-only llm', () => {
  it('test_provider_only_llm_works — engine has defaultModelName === undefined', () => {
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'anthropic',
        llmApiKey: 'sk-ant-x',
      });
      // Touching .messages triggers engine construction.
      const facade = client.messages;
      expect(typeof facade.create).toBe('function');
      // Reach into the private engine to assert defaultModelName.
      const engine = (client as unknown as { engine?: CompresrEngine }).engine;
      expect(engine).toBeInstanceOf(CompresrEngine);
      expect(engine?.defaultModelName).toBeUndefined();
      expect(engine?.provider).toBe('anthropic');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('run() throws missing_model when no model is set anywhere', async () => {
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'anthropic',
        llmApiKey: 'sk-ant-x',
      });
      await expect(client.run({ prompt: 'hi' })).rejects.toMatchObject({
        code: 'missing_model',
      });
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('run() forwards topP to initChatModel constructor', async () => {
    const initChatModel = vi.fn(
      (_spec: string, _kwargs?: Record<string, unknown>) => ({})
    );
    _setLangChainBindings({
      initChatModel: initChatModel as unknown as (
        spec: string,
        kwargs?: Record<string, unknown>
      ) => unknown,
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'pong',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
            }),
          ],
        }),
      }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'anthropic:claude-haiku-4-5',
      });
      await client.run({ prompt: 'hi', topP: 0.42, temperature: 0.3 });
      expect(initChatModel).toHaveBeenCalledTimes(1);
      expect(initChatModel).toHaveBeenCalledWith(
        'anthropic:claude-haiku-4-5',
        expect.objectContaining({ topP: 0.42, temperature: 0.3 })
      );
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('run() uses call-site model when constructor llm is bare provider', async () => {
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
              content: 'pong',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
            }),
          ],
        }),
      }),
    });
    try {
      const client = new CompressionClient({
        apiKey: 'cmp_test',
        llm: 'anthropic',
        llmApiKey: 'sk-ant-x',
      });
      const result = await client.run({
        prompt: 'hi',
        model: 'claude-haiku-4-5',
      });
      expect(result.text).toBe('pong');
      expect(seenSpecs).toEqual(['anthropic:claude-haiku-4-5']);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});
