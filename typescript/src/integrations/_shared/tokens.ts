/**
 * Token estimation.
 *
 * Uses a 4-chars-per-token approximation. The JS ecosystem doesn't have a
 * single dominant tokenizer (gpt-tokenizer, js-tiktoken, tiktoken-node all
 * have different install footprints and platform constraints); keeping the
 * default zero-dep is the right tradeoff for an integration package.
 *
 * Users who want precise counts can pass `estimator: (text) => number`
 * into integrations that accept it.
 */

export type TokenEstimator = (text: string) => number;

/**
 * Estimate token count for `text`.
 *
 * Falls back to `Math.max(1, Math.floor(text.length / 4))`. Returns 0 for
 * empty/falsy input.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.floor(text.length / 4));
}
