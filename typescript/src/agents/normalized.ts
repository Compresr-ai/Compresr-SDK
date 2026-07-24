/**
 * Normalized result types used by Wave 2B's provider-shape facades.
 *
 * These shapes are intentionally provider-agnostic — every backend (Anthropic,
 * OpenAI Responses, Gemini grounding) gets remapped into the same
 * ``NormalizedResult`` by ``CompresrEngine``. Plain interfaces keep the shape
 * immutable on the type-level and free of runtime overhead.
 *
 * Mirrors Python ``compresr/agents/normalized.py``.
 */

/**
 * Aggregate compression stats for a single agent run.
 *
 * ``byTool`` maps tool name -> tokens saved. Wave 2A leaves these at
 * zero; Wave 2B+ will plumb the actual middleware savings.
 */
export interface CompresrStats {
  readonly tokensSaved: number;
  readonly originalTotal: number;
  readonly compressedTotal: number;
  readonly byTool: Readonly<Record<string, number>>;
}

export function defaultCompresrStats(): CompresrStats {
  return {
    tokensSaved: 0,
    originalTotal: 0,
    compressedTotal: 0,
    byTool: {},
  };
}

/**
 * A single citation extracted from a provider response.
 *
 * Raw provider data is preserved in ``providerMetadata`` so callers
 * can recover anything the normalizer dropped.
 */
export interface Citation {
  readonly url: string;
  readonly title?: string;
  readonly citedText?: string;
  readonly providerMetadata: Readonly<Record<string, unknown>>;
}

/** Normalized tool-use shape — ``{ id, name, input }`` triples. */
export interface NormalizedToolUse {
  readonly id?: string;
  readonly name?: string;
  readonly input?: Record<string, unknown>;
}

/**
 * Provider-agnostic agent invocation result.
 *
 * ``raw`` preserves the original (final) ``AIMessage`` so advanced users
 * can reach in for fields the normalizer doesn't surface. ``messages``
 * holds the full conversation chain so callers can walk the trajectory
 * step by step.
 */
export interface NormalizedResult {
  readonly text: string;
  readonly contentBlocks: ReadonlyArray<unknown>;
  readonly toolUses: ReadonlyArray<NormalizedToolUse>;
  readonly citations: ReadonlyArray<Citation>;
  readonly stopReason: string;
  readonly usage: Readonly<Record<string, number>>;
  readonly compresrStats: CompresrStats;
  readonly raw: unknown;
  readonly messages: ReadonlyArray<unknown>;
}

export function emptyNormalizedResult(): NormalizedResult {
  return {
    text: '',
    contentBlocks: [],
    toolUses: [],
    citations: [],
    stopReason: 'end_turn',
    usage: {},
    compresrStats: defaultCompresrStats(),
    raw: null,
    messages: [],
  };
}
