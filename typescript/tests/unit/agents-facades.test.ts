/**
 * Unit tests for the Anthropic and OpenAI provider-shape facades.
 *
 * These translate ``NormalizedResult`` -> provider-native shapes; we feed
 * the engine a stubbed LangChain so the only thing under test is the shape
 * mapping itself.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  toAnthropicMessage,
  toChatCompletion,
  type NormalizedResult,
} from '../../src/agents/index.js';
import {
  CompresrEngine,
  _setLangChainBindings,
} from '../../src/agents/engine.js';
import {
  anthropicMessages,
  openaiChatCompletions,
} from '../../src/agents/index.js';
import { CompresrError } from '../../src/errors/index.js';
import { FakeCompressionClient, type FakeAsClient } from './_fake-client.js';

function buildNormalized(
  patch: Partial<NormalizedResult> = {}
): NormalizedResult {
  return {
    text: 'an answer',
    contentBlocks: [],
    toolUses: [],
    citations: [],
    stopReason: 'end_turn',
    usage: { input_tokens: 10, output_tokens: 4 },
    compresrStats: {
      tokensSaved: 6,
      originalTotal: 10,
      compressedTotal: 4,
      byTool: {},
    },
    raw: { id: 'msg_test_1' },
    messages: [],
    ...patch,
  };
}

describe('toAnthropicMessage', () => {
  it('maps text + usage onto the Anthropic shape', () => {
    const msg = toAnthropicMessage(buildNormalized(), {
      model: 'claude-haiku-4-5',
    });
    expect(msg.role).toBe('assistant');
    expect(msg.type).toBe('message');
    expect(msg.model).toBe('claude-haiku-4-5');
    expect(msg.content[0]).toMatchObject({ type: 'text', text: 'an answer' });
    expect(msg.usage.input_tokens).toBe(10);
    expect(msg.usage.output_tokens).toBe(4);
    expect(msg.id).toBe('msg_test_1');
  });

  it('emits tool_use blocks for tool calls', () => {
    const msg = toAnthropicMessage(
      buildNormalized({
        text: '',
        toolUses: [{ id: 'tu1', name: 'search', input: { q: 'x' } }],
      }),
      { model: 'claude-haiku-4-5' }
    );
    expect(msg.content).toHaveLength(1);
    expect(msg.content[0]).toMatchObject({
      type: 'tool_use',
      id: 'tu1',
      name: 'search',
      input: { q: 'x' },
    });
  });

  it('falls back to a synthetic id when raw lacks one', () => {
    const msg = toAnthropicMessage(buildNormalized({ raw: null }), {
      model: 'm',
    });
    expect(msg.id.startsWith('msg_')).toBe(true);
  });

  it('carries the full conversation chain on .messages', () => {
    const msg = toAnthropicMessage(
      buildNormalized({ messages: ['m1', 'm2', 'm3'] }),
      { model: 'claude-haiku-4-5' }
    );
    expect(msg.messages).toEqual(['m1', 'm2', 'm3']);
  });

  it('surfaces aggregate token counts on the Usage block', () => {
    const msg = toAnthropicMessage(
      buildNormalized({
        usage: {
          input_tokens: 600,
          output_tokens: 60,
          cache_read_input_tokens: 120,
          cache_creation_input_tokens: 50,
          ai_message_count: 3,
        },
      }),
      { model: 'claude-haiku-4-5' }
    );
    expect(msg.usage.input_tokens).toBe(600);
    expect(msg.usage.output_tokens).toBe(60);
    expect(msg.usage.cache_read_input_tokens).toBe(120);
    expect(msg.usage.cache_creation_input_tokens).toBe(50);
  });
});

describe('toChatCompletion', () => {
  it('maps text + usage onto the OpenAI shape', () => {
    const cmp = toChatCompletion(buildNormalized(), { model: 'gpt-5' });
    expect(cmp.object).toBe('chat.completion');
    expect(cmp.choices).toHaveLength(1);
    const choice = cmp.choices[0]!;
    expect(choice.message.role).toBe('assistant');
    expect(choice.message.content).toBe('an answer');
    expect(choice.finish_reason).toBe('stop');
    expect(cmp.usage.prompt_tokens).toBe(10);
    expect(cmp.usage.completion_tokens).toBe(4);
    expect(cmp.usage.total_tokens).toBe(14);
  });

  it('emits tool_calls with JSON-encoded arguments', () => {
    const cmp = toChatCompletion(
      buildNormalized({
        text: '',
        toolUses: [{ id: 'tu1', name: 'search', input: { q: 'x' } }],
        stopReason: 'tool_use',
      }),
      { model: 'gpt-5' }
    );
    const choice = cmp.choices[0]!;
    expect(choice.finish_reason).toBe('tool_calls');
    expect(choice.message.tool_calls).toHaveLength(1);
    const call = choice.message.tool_calls[0]!;
    expect(call.type).toBe('function');
    expect(call.function.name).toBe('search');
    expect(JSON.parse(call.function.arguments)).toEqual({ q: 'x' });
  });

  it('carries the full conversation chain on .messages', () => {
    const cmp = toChatCompletion(
      buildNormalized({ messages: ['m1', 'm2'] }),
      { model: 'gpt-5' }
    );
    expect(cmp.messages).toEqual(['m1', 'm2']);
  });

  it('surfaces aggregate token counts on the Usage block', () => {
    const cmp = toChatCompletion(
      buildNormalized({
        usage: { input_tokens: 500, output_tokens: 40, ai_message_count: 4 },
      }),
      { model: 'gpt-5' }
    );
    expect(cmp.usage.prompt_tokens).toBe(500);
    expect(cmp.usage.completion_tokens).toBe(40);
    expect(cmp.usage.total_tokens).toBe(540);
  });
});

describe('anthropicMessages facade', () => {
  it('runs the engine and returns the AnthropicMessage shape', async () => {
    const fake = new FakeCompressionClient();
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'hi back',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: { input_tokens: 1, output_tokens: 2 },
              id: 'msg_x',
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
      const facade = anthropicMessages(engine);
      const msg = await facade.create({
        model: 'claude-haiku-4-5',
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(msg.role).toBe('assistant');
      expect(msg.id).toBe('msg_x');
      expect(msg.content[0]).toMatchObject({ type: 'text', text: 'hi back' });
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('stream() throws not_implemented', async () => {
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
      const facade = anthropicMessages(engine);
      await expect(
        facade.stream({
          model: 'claude-haiku-4-5',
          messages: [{ role: 'user', content: 'hi' }],
        })
      ).rejects.toBeInstanceOf(CompresrError);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('openaiChatCompletions facade', () => {
  it('runs the engine and returns the ChatCompletion shape', async () => {
    const fake = new FakeCompressionClient();
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({
        invoke: async () => ({
          messages: [
            Object.assign(Object.create({ constructor: { name: 'AIMessage' } }), {
              content: 'hi back',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: { input_tokens: 1, output_tokens: 2 },
              id: 'chatcmpl-x',
            }),
          ],
        }),
      }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-5',
      });
      const facade = openaiChatCompletions(engine);
      const cmp = await facade.completions.create({
        model: 'gpt-5',
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(cmp.object).toBe('chat.completion');
      expect(cmp.choices[0]?.message.content).toBe('hi back');
      expect(cmp.id).toBe('chatcmpl-x');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('completions.stream() throws not_implemented', async () => {
    const fake = new FakeCompressionClient();
    _setLangChainBindings({
      initChatModel: () => ({}),
      createAgent: () => ({ invoke: async () => ({ messages: [] }) }),
    });
    try {
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-5',
      });
      const facade = openaiChatCompletions(engine);
      await expect(
        facade.completions.stream({
          model: 'gpt-5',
          messages: [{ role: 'user', content: 'hi' }],
        })
      ).rejects.toBeInstanceOf(CompresrError);
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('facades forward the call-site model to engine.run', () => {
  it('anthropic messages.create forwards model to the engine', async () => {
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
              content: 'hi',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
              id: 'msg_y',
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
      const facade = anthropicMessages(engine);
      const msg = await facade.create({
        model: 'claude-opus-4-5',
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(seenSpecs).toEqual(['anthropic:claude-opus-4-5']);
      expect(msg.model).toBe('claude-opus-4-5');
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('openai completions.create forwards model to the engine', async () => {
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
              content: 'hi',
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
        llm: 'openai',
      });
      const facade = openaiChatCompletions(engine);
      const cmp = await facade.completions.create({
        model: 'gpt-5-mini',
        messages: [{ role: 'user', content: 'hi' }],
      });
      expect(seenSpecs).toEqual(['openai:gpt-5-mini']);
      expect(cmp.model).toBe('gpt-5-mini');
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});

describe('facades forward per-call LLM knobs to initChatModel constructor', () => {
  function installStub(): ReturnType<typeof vi.fn> {
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
              content: 'ok',
              tool_calls: [],
              response_metadata: { stop_reason: 'end_turn' },
              usage_metadata: {},
            }),
          ],
        }),
      }),
    });
    return initChatModel;
  }

  it('anthropic facade forwards temperature, topP, topK, stopSequences to initChatModel', async () => {
    const initChatModel = installStub();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'anthropic:claude-haiku-4-5',
      });
      const facade = anthropicMessages(engine);
      await facade.create({
        model: 'claude-haiku-4-5',
        messages: [{ role: 'user', content: 'hi' }],
        temperature: 0.7,
        topP: 0.95,
        topK: 40,
        stopSequences: ['STOP'],
        maxTokens: 256,
      });
      expect(initChatModel).toHaveBeenCalledTimes(1);
      expect(initChatModel).toHaveBeenCalledWith(
        'anthropic:claude-haiku-4-5',
        expect.objectContaining({
          temperature: 0.7,
          topP: 0.95,
          topK: 40,
          stopSequences: ['STOP'],
          maxTokens: 256,
        })
      );
    } finally {
      _setLangChainBindings(undefined);
    }
  });

  it('openai facade forwards presencePenalty, frequencyPenalty, seed, stop, logprobs to initChatModel', async () => {
    const initChatModel = installStub();
    try {
      const fake = new FakeCompressionClient();
      const engine = new CompresrEngine({
        compresrClient: fake as FakeAsClient,
        llm: 'openai:gpt-5',
      });
      const facade = openaiChatCompletions(engine);
      await facade.completions.create({
        model: 'gpt-5',
        messages: [{ role: 'user', content: 'hi' }],
        presencePenalty: 0.4,
        frequencyPenalty: 0.2,
        seed: 99,
        stop: ['\n\n'],
        logprobs: true,
        topLogprobs: 3,
        temperature: 0.1,
        topP: 0.5,
      });
      expect(initChatModel).toHaveBeenCalledTimes(1);
      expect(initChatModel).toHaveBeenCalledWith(
        'openai:gpt-5',
        expect.objectContaining({
          presencePenalty: 0.4,
          frequencyPenalty: 0.2,
          seed: 99,
          stop: ['\n\n'],
          logprobs: true,
          topLogprobs: 3,
          temperature: 0.1,
          topP: 0.5,
        })
      );
    } finally {
      _setLangChainBindings(undefined);
    }
  });
});
