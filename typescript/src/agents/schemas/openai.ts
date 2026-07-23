/**
 * OpenAI ``ChatCompletion`` look-alike interfaces.
 *
 * Customers using ``new OpenAI().chat.completions.create(...)`` can swap in
 * ``new CompressionClient(...).chat.completions.create(...)`` without
 * installing ``openai`` or rewriting downstream parsing — the return shape
 * mirrors ``openai.types.chat.ChatCompletion``.
 *
 * Mirrors Python ``compresr/agents/schemas/openai.py``.
 */
import type { CompresrStats, NormalizedResult } from '../normalized.js';

export interface FunctionCall {
  readonly name: string;
  /** JSON-encoded arguments string, per OpenAI's wire shape. */
  readonly arguments: string;
}

export interface ToolCall {
  readonly id: string;
  readonly type: 'function';
  readonly function: FunctionCall;
}

export interface ChatMessage {
  readonly role: 'assistant';
  readonly content: string | null;
  readonly tool_calls: ReadonlyArray<ToolCall>;
}

export interface Choice {
  readonly index: number;
  readonly message: ChatMessage;
  readonly finish_reason: string;
}

export interface OpenAIUsage {
  readonly prompt_tokens: number;
  readonly completion_tokens: number;
  readonly total_tokens: number;
}

export interface ChatCompletion {
  readonly id: string;
  readonly object: 'chat.completion';
  readonly model: string;
  readonly choices: ReadonlyArray<Choice>;
  readonly usage: OpenAIUsage;
  readonly compresr: CompresrStats;
  readonly raw: unknown;
  readonly messages: ReadonlyArray<unknown>;
}

const FINISH_REASON_MAP: Readonly<Record<string, string>> = {
  end_turn: 'stop',
  stop_sequence: 'stop',
  max_tokens: 'length',
  tool_use: 'tool_calls',
  pause_turn: 'tool_calls',
};

function intOrZero(v: unknown): number {
  if (typeof v === 'number' && Number.isFinite(v)) return Math.trunc(v);
  if (typeof v === 'string') {
    const n = Number(v);
    if (Number.isFinite(n)) return Math.trunc(n);
  }
  return 0;
}

function deriveId(result: NormalizedResult, prefix: string): string {
  const raw = result.raw as { id?: string } | null;
  if (raw && typeof raw.id === 'string' && raw.id) return raw.id;
  return `${prefix}generated`;
}

export interface ToChatCompletionOptions {
  model: string;
  completionId?: string;
}

export function toChatCompletion(
  result: NormalizedResult,
  options: ToChatCompletionOptions
): ChatCompletion {
  const toolCalls: ToolCall[] = result.toolUses.map((tu) => ({
    id: tu.id ?? '',
    type: 'function',
    function: {
      name: tu.name ?? '',
      arguments: JSON.stringify(tu.input ?? {}),
    },
  }));
  const message: ChatMessage = {
    role: 'assistant',
    content: result.text || null,
    tool_calls: toolCalls,
  };
  const stop = result.stopReason || '';
  const finishReason = FINISH_REASON_MAP[stop] ?? (stop || 'stop');
  const u = result.usage;
  const promptTokens = intOrZero(
    u['input_tokens'] ?? u['prompt_tokens']
  );
  const completionTokens = intOrZero(
    u['output_tokens'] ?? u['completion_tokens']
  );
  return {
    id: options.completionId ?? deriveId(result, 'chatcmpl-'),
    object: 'chat.completion',
    model: options.model,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
    compresr: result.compresrStats,
    raw: result.raw,
    messages: [...result.messages],
  };
}
