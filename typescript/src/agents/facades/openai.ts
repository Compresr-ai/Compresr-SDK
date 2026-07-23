/**
 * OpenAI-shaped facade — ``client.chat.completions.create(...)``.
 *
 * Delegates the actual run to a private :class:`CompresrEngine` and remaps the
 * returned :class:`NormalizedResult` to an OpenAI ``ChatCompletion`` shape.
 *
 * Mirrors Python ``compresr/agents/facades/openai.py``.
 */
import { CompresrError } from '../../errors/index.js';
import { CompresrEngine } from '../engine.js';
import { toChatCompletion, type ChatCompletion } from '../schemas/openai.js';

export interface OpenAICompletionsCreateOptions {
  /**
   * Optional per-call model. Defaults to the model encoded on the engine's
   * ``llm`` spec (``'openai:gpt-...'``). If neither is set, the engine throws
   * ``CompresrError("missing_model")``.
   */
  model?: string;
  messages: ReadonlyArray<unknown>;
  maxTokens?: number;
  tools?: ReadonlyArray<unknown>;
  config?: Record<string, unknown>;
  temperature?: number;
  topP?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  stop?: string[];
  logprobs?: boolean;
  topLogprobs?: number;
}

export interface OpenAICompletionsFacade {
  create(options: OpenAICompletionsCreateOptions): Promise<ChatCompletion>;
  /** Streaming not yet implemented — throws. */
  stream(options: OpenAICompletionsCreateOptions): Promise<never>;
}

export interface OpenAIChatFacade {
  completions: OpenAICompletionsFacade;
}

export function openaiChatCompletions(engine: CompresrEngine): OpenAIChatFacade {
  const completions: OpenAICompletionsFacade = {
    async create(options: OpenAICompletionsCreateOptions): Promise<ChatCompletion> {
      const runOpts: Parameters<typeof engine.run>[0] = {
        messages: options.messages,
        tools: options.tools ?? [],
        maxTokens: options.maxTokens ?? 4096,
      };
      if (options.model !== undefined) runOpts.model = options.model;
      if (options.config !== undefined) runOpts.config = options.config;
      if (options.temperature !== undefined) runOpts.temperature = options.temperature;
      if (options.topP !== undefined) runOpts.topP = options.topP;
      if (options.presencePenalty !== undefined) {
        runOpts.presencePenalty = options.presencePenalty;
      }
      if (options.frequencyPenalty !== undefined) {
        runOpts.frequencyPenalty = options.frequencyPenalty;
      }
      if (options.seed !== undefined) runOpts.seed = options.seed;
      if (options.stop !== undefined) runOpts.stop = options.stop;
      if (options.logprobs !== undefined) runOpts.logprobs = options.logprobs;
      if (options.topLogprobs !== undefined) {
        runOpts.topLogprobs = options.topLogprobs;
      }
      const result = await engine.run(runOpts);
      const responseModel = options.model ?? engine.defaultModelName ?? '';
      return toChatCompletion(result, { model: responseModel });
    },
    stream(_options: OpenAICompletionsCreateOptions): Promise<never> {
      return Promise.reject(
        new CompresrError(
          'streaming not yet implemented — Phase 2 work item',
          'not_implemented'
        )
      );
    },
  };
  return { completions };
}
