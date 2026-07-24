/**
 * Anthropic-shaped facade — ``client.messages.create(...)``.
 *
 * Delegates the actual run to a private :class:`CompresrEngine` and remaps the
 * returned :class:`NormalizedResult` into an Anthropic ``Message`` shape.
 *
 * Mirrors Python ``compresr/agents/facades/anthropic.py``.
 */
import { CompresrError } from '../../errors/index.js';
import { CompresrEngine } from '../engine.js';
import {
  toAnthropicMessage,
  type AnthropicMessage,
} from '../schemas/anthropic.js';

export interface AnthropicMessagesCreateOptions {
  /**
   * Optional per-call model. Defaults to the model encoded on the engine's
   * ``llm`` spec (``'anthropic:claude-...'``). If neither is set, the engine
   * throws ``CompresrError("missing_model")``.
   */
  model?: string;
  messages: ReadonlyArray<unknown>;
  maxTokens?: number;
  tools?: ReadonlyArray<unknown>;
  system?: string;
  config?: Record<string, unknown>;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

export interface AnthropicMessagesFacade {
  create(options: AnthropicMessagesCreateOptions): Promise<AnthropicMessage>;
  /** Streaming not yet implemented — throws. */
  stream(options: AnthropicMessagesCreateOptions): Promise<never>;
}

export function anthropicMessages(engine: CompresrEngine): AnthropicMessagesFacade {
  return {
    async create(options: AnthropicMessagesCreateOptions): Promise<AnthropicMessage> {
      const runOpts: Parameters<typeof engine.run>[0] = {
        messages: options.messages,
        tools: options.tools ?? [],
        maxTokens: options.maxTokens ?? 4096,
      };
      if (options.model !== undefined) runOpts.model = options.model;
      if (options.system !== undefined) runOpts.system = options.system;
      if (options.config !== undefined) runOpts.config = options.config;
      if (options.temperature !== undefined) runOpts.temperature = options.temperature;
      if (options.topP !== undefined) runOpts.topP = options.topP;
      if (options.topK !== undefined) runOpts.topK = options.topK;
      if (options.stopSequences !== undefined) {
        runOpts.stopSequences = options.stopSequences;
      }
      const result = await engine.run(runOpts);
      const responseModel = options.model ?? engine.defaultModelName ?? '';
      return toAnthropicMessage(result, { model: responseModel });
    },
    stream(_options: AnthropicMessagesCreateOptions): Promise<never> {
      return Promise.reject(
        new CompresrError(
          'streaming not yet implemented — Phase 2 work item',
          'not_implemented'
        )
      );
    },
  };
}
