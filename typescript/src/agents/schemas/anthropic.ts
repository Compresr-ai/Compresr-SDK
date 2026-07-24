/**
 * Anthropic ``Message`` look-alike interfaces.
 *
 * These duck-type the Anthropic SDK's response shape (``message.content[0].text``,
 * ``message.usage.input_tokens``, ``message.stop_reason``) so customers can
 * swap ``CompressionClient`` in for their existing ``Anthropic`` client without
 * installing the ``@anthropic-ai/sdk`` package or rewriting downstream parsing.
 *
 * Mirrors Python ``compresr/agents/schemas/anthropic.py``.
 */
import type {
  Citation,
  CompresrStats,
  NormalizedResult,
} from '../normalized.js';

export interface TextBlock {
  readonly type: 'text';
  readonly text: string;
  readonly citations: ReadonlyArray<Citation>;
}

export interface ToolUseBlock {
  readonly type: 'tool_use';
  readonly id: string;
  readonly name: string;
  readonly input: Readonly<Record<string, unknown>>;
}

export type ContentBlock = TextBlock | ToolUseBlock;

export interface AnthropicUsage {
  readonly input_tokens: number;
  readonly output_tokens: number;
  readonly cache_read_input_tokens: number;
  readonly cache_creation_input_tokens: number;
}

export interface AnthropicMessage {
  readonly id: string;
  readonly role: 'assistant';
  readonly type: 'message';
  readonly model: string;
  readonly content: ReadonlyArray<ContentBlock>;
  readonly stop_reason: string;
  readonly usage: AnthropicUsage;
  readonly compresr: CompresrStats;
  readonly raw: unknown;
  readonly messages: ReadonlyArray<unknown>;
}

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

export interface ToAnthropicMessageOptions {
  model: string;
  msgId?: string;
}

export function toAnthropicMessage(
  result: NormalizedResult,
  options: ToAnthropicMessageOptions
): AnthropicMessage {
  const blocks: ContentBlock[] = [];
  if (result.text) {
    blocks.push({
      type: 'text',
      text: result.text,
      citations: [...result.citations],
    });
  }
  for (const tu of result.toolUses) {
    blocks.push({
      type: 'tool_use',
      id: tu.id ?? '',
      name: tu.name ?? '',
      input: { ...(tu.input ?? {}) },
    });
  }
  const u = result.usage;
  return {
    id: options.msgId ?? deriveId(result, 'msg_'),
    role: 'assistant',
    type: 'message',
    model: options.model,
    content: blocks,
    stop_reason: result.stopReason || 'end_turn',
    usage: {
      input_tokens: intOrZero(u['input_tokens']),
      output_tokens: intOrZero(u['output_tokens']),
      cache_read_input_tokens: intOrZero(u['cache_read_input_tokens']),
      cache_creation_input_tokens: intOrZero(u['cache_creation_input_tokens']),
    },
    compresr: result.compresrStats,
    raw: result.raw,
    messages: [...result.messages],
  };
}
