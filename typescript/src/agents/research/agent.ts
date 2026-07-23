/**
 * `ResearchAgent` — strict-output ReAct loop with per-snippet compression.
 *
 * Mirrors the Python implementation in `compresr/agents/research/agent.py`.
 * Bypasses `_Engine.run` so we can set `tool_choice` per step and compress
 * each tool result before it enters the conversation.
 */

import { parseResearchOutput } from './parser.js';
import { DEFAULT_RESEARCH_SYSTEM_PROMPT } from './prompts.js';
import type { Citation, ResearchResult, ResearchUsage, Step } from './types.js';

interface CompresrClientLike {
  compress(opts: {
    context: string;
    query?: string;
    compressionModelName?: string;
  }): Promise<{
    data?: { compressedContext?: string; compressed_context?: string };
  }>;
}

export interface ChatModelLike {
  bindTools(
    tools: ReadonlyArray<unknown>,
    options: { tool_choice: string }
  ): { invoke(messages: unknown[]): Promise<unknown> };
}

export interface SearchToolLike {
  name?: string;
  invoke(args: Record<string, unknown>): Promise<unknown>;
}

export interface ResearchEngine {
  readonly provider: string;
  readonly defaultModelName?: string;
  _resolveModel(model?: string): string;
  _getChatModel(modelName: string): Promise<ChatModelLike>;
  readonly _compresrClient: CompresrClientLike;
  readonly _promptCacheConfig: {
    readonly enabled: boolean;
    readonly ttl: '5m' | '1h';
    readonly minMessages: number;
  };
}

interface AIMessageLike {
  content?: unknown;
  tool_calls?: ReadonlyArray<{
    id?: string;
    name?: string;
    args?: Record<string, unknown>;
  }>;
  usage_metadata?: Record<string, unknown>;
}

const DEFAULT_MAX_CONTEXT_TOKENS = 120_000;
const DEFAULT_MAX_STEPS = 10;
const DEFAULT_MIN_COMPRESS_TOKENS = 100;

const CACHE_READ_ALIASES = [
  'cached_tokens',
  'cache_read',
  'priority_cache_read',
  'flex_cache_read',
  'cached_content_token_count',
] as const;

export interface ResearchAgentOptions {
  engine: ResearchEngine;
  searchTool: SearchToolLike;
  maxSteps?: number;
  systemPrompt?: string;
  compressSnippets?: boolean;
  compressionModel?: string;
  minCompressTokens?: number;
  maxContextTokens?: number;
}

export interface ResearchRunOptions {
  model?: string;
}

/**
 * Internal result of {@link ResearchAgent._runLoop}.
 *
 * Exposed to the engine's solo-WebSearchTool fast-path so it can feed the
 * raw message chain to {@link CompresrEngine._normalize}, while {@link ResearchAgent.run}
 * keeps producing a typed {@link ResearchResult}.
 */
export interface ResearchLoopState {
  messages: unknown[];
  aiHistory: AIMessageLike[];
  trajectory: Step[];
  lastAi: AIMessageLike | null;
}

function estimateTokens(text: string): number {
  return Math.max(1, Math.floor(text.length / 4));
}

function stringifyContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (block && typeof block === 'object' && 'text' in block) {
        const t = (block as { text?: unknown }).text;
        if (typeof t === 'string') parts.push(t);
      } else if (typeof block === 'string') {
        parts.push(block);
      }
    }
    return parts.join('');
  }
  if (content == null) return '';
  if (typeof content === 'number' || typeof content === 'boolean') return String(content);
  return JSON.stringify(content);
}

function readCacheTokens(
  details: Record<string, unknown> | undefined
): { read: number; create: number } {
  if (!details) return { read: 0, create: 0 };
  let read = 0;
  for (const alias of CACHE_READ_ALIASES) {
    const v = details[alias];
    if (typeof v === 'number') read += v;
  }
  // langchain-anthropic zeros ``cache_creation`` when ttl-specific keys are set.
  const ttl5 = typeof details['ephemeral_5m_input_tokens'] === 'number'
    ? (details['ephemeral_5m_input_tokens']) : 0;
  const ttl1 = typeof details['ephemeral_1h_input_tokens'] === 'number'
    ? (details['ephemeral_1h_input_tokens']) : 0;
  const gen = typeof details['cache_creation'] === 'number'
    ? (details['cache_creation']) : 0;
  const create = ttl5 + ttl1 > 0 ? ttl5 + ttl1 : gen;
  return { read, create };
}

export class ResearchAgent {
  private readonly engine: ResearchEngine;
  private readonly searchTool: SearchToolLike;
  private readonly maxSteps: number;
  private readonly systemPrompt: string;
  private readonly compressSnippets: boolean;
  private readonly compressionModel: string;
  private readonly minCompressTokens: number;
  private readonly maxContextTokens: number;

  constructor(options: ResearchAgentOptions) {
    this.engine = options.engine;
    this.searchTool = options.searchTool;
    this.maxSteps = Math.max(1, options.maxSteps ?? DEFAULT_MAX_STEPS);
    this.systemPrompt = options.systemPrompt ?? DEFAULT_RESEARCH_SYSTEM_PROMPT;
    this.compressSnippets = options.compressSnippets ?? true;
    this.compressionModel = options.compressionModel ?? 'latte_v1';
    this.minCompressTokens = options.minCompressTokens ?? DEFAULT_MIN_COMPRESS_TOKENS;
    this.maxContextTokens = options.maxContextTokens ?? DEFAULT_MAX_CONTEXT_TOKENS;
  }

  async run(question: string, runOptions: ResearchRunOptions = {}): Promise<ResearchResult> {
    const state = await this._runLoop(question, runOptions);
    const searchCalls = state.trajectory.filter((s) => s.type === 'search').length;
    const usage = this.aggregateUsage(state.aiHistory, searchCalls);
    const finalText = state.lastAi ? stringifyContent(state.lastAi.content) : '';
    const parsed = parseResearchOutput(finalText);
    const citations = this.collectCitations(parsed.citationUrls, state.trajectory);

    return {
      answer: parsed.answer,
      explanation: parsed.explanation,
      confidence: parsed.confidence,
      text: finalText,
      citations,
      trajectory: state.trajectory,
      usage,
      raw: state.lastAi,
    };
  }

  /**
   * Execute the strict-output ReAct loop and return raw state.
   *
   * Shared by {@link run} (which parses ``ResearchResult``) and the engine's
   * solo-WebSearchTool fast-path (which feeds ``state.messages`` into
   * the engine's normalizer).
   *
   * @internal — public for the engine's fast-path; not a stable surface.
   */
  async _runLoop(
    question: string,
    runOptions: ResearchRunOptions = {}
  ): Promise<ResearchLoopState> {
    const { SystemMessage, HumanMessage, ToolMessage } = await import(
      '@langchain/core/messages'
    );

    const modelName = this.engine._resolveModel(runOptions.model);
    const chat = await this.engine._getChatModel(modelName);

    const messages: unknown[] = [
      new SystemMessage(this.systemPrompt),
      new HumanMessage(question),
    ];
    const aiHistory: AIMessageLike[] = [];
    const trajectory: Step[] = [];
    let lastAi: AIMessageLike | null = null;

    for (let stepIdx = 0; stepIdx < this.maxSteps; stepIdx++) {
      this.maybeTruncate(messages);
      const toolChoice = stepIdx === this.maxSteps - 1 ? 'none' : 'auto';
      const bound = chat.bindTools([this.searchTool], { tool_choice: toolChoice });
      const invokeMessages = this.applyCacheControl(messages);

      const t0 = Date.now();
      let response: AIMessageLike;
      try {
        response = (await bound.invoke(invokeMessages)) as AIMessageLike;
      } catch (exc) {
        trajectory.push({
          type: 'error',
          text: exc instanceof Error ? exc.message : String(exc),
          latencyS: (Date.now() - t0) / 1000,
        });
        break;
      }
      const stepLatencyS = (Date.now() - t0) / 1000;

      aiHistory.push(response);
      lastAi = response;
      messages.push(response);

      const toolCalls = response.tool_calls ?? [];

      if (toolCalls.length === 0) {
        trajectory.push({
          type: 'answer',
          text: stringifyContent(response.content),
          latencyS: stepLatencyS,
        });
        break;
      }

      for (const tc of toolCalls) {
        const args = tc.args ?? {};
        const query = typeof args['query'] === 'string' ? (args['query']) : undefined;
        trajectory.push({ type: 'search', query, latencyS: stepLatencyS });

        const rawResult = await this.invokeSearch(args);
        const finalResult = this.compressSnippets
          ? await this.compress(rawResult, query ?? question)
          : rawResult;
        trajectory.push({ type: 'tool_result', chars: finalResult.length, text: finalResult });

        messages.push(
          new ToolMessage({
            content: finalResult,
            tool_call_id: tc.id ?? '',
            name: this.searchTool.name ?? 'search_web',
          })
        );
      }
    }

    return { messages, aiHistory, trajectory, lastAi };
  }

  /**
   * Mirrors AnthropicPromptCachingMiddleware for the bypass path — research
   * doesn't go through createAgent so we stamp cache_control manually.
   */
  private applyCacheControl(messages: unknown[]): unknown[] {
    if (this.engine.provider !== 'anthropic') return messages;
    const cfg = this.engine._promptCacheConfig;
    if (!cfg?.enabled) return messages;
    if (messages.length < cfg.minMessages) return messages;

    const marker = { type: 'ephemeral', ttl: cfg.ttl };
    const last = messages[messages.length - 1] as {
      content?: unknown;
      constructor?: new (args: unknown) => unknown;
    };
    const content = last?.content;
    let newContent: unknown[];
    if (typeof content === 'string') {
      newContent = [{ type: 'text', text: content, cache_control: marker }];
    } else if (Array.isArray(content) && content.length > 0) {
      newContent = [...(content as unknown[])];
      const tail = newContent[newContent.length - 1];
      if (tail !== null && typeof tail === 'object') {
        newContent[newContent.length - 1] = {
          ...(tail as Record<string, unknown>),
          cache_control: marker,
        };
      } else {
        const tailText = typeof tail === 'string' ? tail : '';
        newContent.push({ type: 'text', text: tailText, cache_control: marker });
      }
    } else {
      return messages;
    }
    const Ctor = last.constructor as new (arg: { content: unknown[] }) => unknown;
    const patched = new Ctor({ content: newContent });
    return [...messages.slice(0, -1), patched];
  }

  private async invokeSearch(args: Record<string, unknown>): Promise<string> {
    try {
      const out = await this.searchTool.invoke(args);
      if (typeof out === 'string') return out;
      if (out == null) return '';
      return JSON.stringify(out);
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : String(exc);
      // eslint-disable-next-line no-console
      console.warn(`[compresr.research] search tool raised: ${msg}; continuing.`);
      return `Search error: ${msg}`;
    }
  }

  private async compress(text: string, query: string): Promise<string> {
    if (!text || estimateTokens(text) < this.minCompressTokens) return text;
    const client = this.engine._compresrClient;
    if (!client?.compress) return text;
    try {
      const resp = await client.compress({
        context: text,
        query,
        compressionModelName: this.compressionModel,
      });
      const cc = resp?.data?.compressedContext ?? resp?.data?.compressed_context;
      return typeof cc === 'string' && cc.length > 0 ? cc : text;
    } catch {
      return text;
    }
  }

  private maybeTruncate(messages: unknown[]): void {
    if (!this.maxContextTokens) return;
    const total = () =>
      messages.reduce<number>(
        (acc, m) => acc + estimateTokens(stringifyContent((m as { content?: unknown })?.content)),
        0
      );
    let cur = total();
    if (cur <= this.maxContextTokens) return;
    let guard = 0;
    while (cur > this.maxContextTokens && messages.length >= 4 && guard < 4) {
      messages.splice(2, 2);
      cur = total();
      guard++;
    }
  }

  private aggregateUsage(aiMessages: AIMessageLike[], searchCalls = 0): ResearchUsage {
    const totals: Record<string, number> = {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: 0,
      cache_creation_tokens: 0,
    };
    for (const m of aiMessages) {
      const um = m?.usage_metadata ?? {};
      const inTok = typeof um['input_tokens'] === 'number' ? (um['input_tokens']) : 0;
      totals['output_tokens'] += typeof um['output_tokens'] === 'number'
        ? (um['output_tokens']) : 0;

      const details = um['input_token_details'] as Record<string, unknown> | undefined;
      const { read, create } = readCacheTokens(details);
      totals['cache_read_tokens'] += read;
      totals['cache_creation_tokens'] += create;

      // langchain-anthropic 1.x inflates input_tokens to fresh+read+create.
      const overcount = read + create;
      totals['input_tokens'] += Math.max(0, inTok - overcount);
    }
    return {
      input_tokens: totals['input_tokens'],
      output_tokens: totals['output_tokens'],
      cache_read_tokens: totals['cache_read_tokens'],
      cache_creation_tokens: totals['cache_creation_tokens'],
      calls: aiMessages.length,
      search_calls: searchCalls,
    };
  }

  private collectCitations(parsedUrls: ReadonlyArray<string>, trajectory: Step[]): Citation[] {
    const urls: string[] = [];
    for (const u of parsedUrls) if (!urls.includes(u)) urls.push(u);
    const urlRe = /https?:\/\/[^\s,;<>"')]+/g;
    for (const step of trajectory) {
      if (step.type !== 'tool_result' || !step.text) continue;
      let m: RegExpExecArray | null;
      while ((m = urlRe.exec(step.text)) !== null) {
        if (!urls.includes(m[0])) urls.push(m[0]);
      }
    }
    return urls.map((url) => ({ url }));
  }
}
