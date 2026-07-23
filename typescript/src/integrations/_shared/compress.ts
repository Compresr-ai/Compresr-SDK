/**
 * Compression helper — async-only (the TS SDK is async-first, no blocking
 * concern).
 *
 * Mirrors Python `_shared/compress.py::acompress_safe`.
 */
import type { CompressionClient } from '../../clients/compression.js';

import { applyErrorPolicyAsync, DEFAULT_POLICY, type ErrorPolicy } from './errors.js';
import { estimateTokens, type TokenEstimator } from './tokens.js';

export interface CompressSafeOptions {
  context: string;
  query?: string;
  compressionModel?: string;
  targetCompressionRatio?: number;
  coarse?: boolean;
  minTokens?: number;
  onError?: ErrorPolicy;
  contextLabel?: string;
  estimator?: TokenEstimator;
}

/**
 * Compress `context` if it exceeds `minTokens`.
 *
 * Returns the compressed string, or the original on short input / error
 * (when `onError='passthrough'`).
 */
export async function compressSafe(
  client: CompressionClient,
  options: CompressSafeOptions
): Promise<string> {
  const {
    context,
    query,
    compressionModel = 'latte_v1',
    targetCompressionRatio,
    coarse,
    minTokens = 200,
    onError = DEFAULT_POLICY,
    contextLabel,
    estimator = estimateTokens,
  } = options;

  if (typeof context !== 'string' || !context.trim()) {
    return context;
  }
  if (estimator(context) < minTokens) {
    return context;
  }

  return applyErrorPolicyAsync(
    async () => {
      const resp = await client.compress({
        context,
        ...(query !== undefined ? { query } : {}),
        compressionModelName: compressionModel,
        ...(targetCompressionRatio !== undefined
          ? { targetCompressionRatio }
          : {}),
        ...(coarse !== undefined ? { coarse } : {}),
      });
      const out = resp.data?.compressed_context;
      return typeof out === 'string' && out ? out : context;
    },
    {
      fallback: context,
      policy: onError,
      context: { label: contextLabel, model: compressionModel },
    }
  );
}
