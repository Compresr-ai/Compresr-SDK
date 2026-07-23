/**
 * Native facade — ``client.run({ prompt, tools, ... })``.
 *
 * The native surface returns :class:`NormalizedResult` directly so callers who
 * don't need provider-specific shapes can skip the remappers entirely.
 *
 * Mirrors Python ``compresr/agents/facades/native.py``.
 */
import type { CompresrEngine } from '../engine.js';
import type { NormalizedResult } from '../normalized.js';

export interface NativeRunOptions {
  prompt: string;
  /**
   * Optional per-call model. Defaults to the model encoded on the engine's
   * ``llm`` spec. If neither is set, the engine throws
   * ``CompresrError("missing_model")``.
   */
  model?: string;
  tools?: ReadonlyArray<unknown>;
  system?: string;
  maxTokens?: number;
  config?: Record<string, unknown>;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

export async function nativeRun(
  engine: CompresrEngine,
  options: NativeRunOptions
): Promise<NormalizedResult> {
  const runOpts: Parameters<typeof engine.run>[0] = {
    messages: [{ role: 'user', content: options.prompt }],
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
  return engine.run(runOpts);
}
