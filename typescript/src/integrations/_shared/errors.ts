/**
 * Unified error policy for integrations.
 *
 * Compression should rarely break a user's app. Default is `passthrough`:
 * log and return the original/fallback value. `raise` is opt-in.
 *
 * Mirrors Python `_shared/errors.py`.
 */

import { getLogger } from '../../logger.js';

export type ErrorPolicy = 'raise' | 'passthrough';
export const DEFAULT_POLICY: ErrorPolicy = 'passthrough';

export interface ErrorPolicyOptions<T> {
  fallback: T;
  policy?: ErrorPolicy;
  context?: Record<string, unknown>;
}

/** Run `fn`; on exception, raise (policy='raise') or log+return `fallback`. */
export function applyErrorPolicy<T>(
  fn: () => T,
  options: ErrorPolicyOptions<T>
): T {
  const { fallback, policy = DEFAULT_POLICY, context } = options;
  try {
    return fn();
  } catch (exc) {
    if (policy === 'raise') {
      throw exc;
    }
    getLogger().warn(
      `integrations call failed (${stringifyError(exc)}); passthrough.`,
      context ?? {}
    );
    return fallback;
  }
}

/** Async variant — handles both sync throws and rejected promises. */
export async function applyErrorPolicyAsync<T>(
  fn: () => Promise<T>,
  options: ErrorPolicyOptions<T>
): Promise<T> {
  const { fallback, policy = DEFAULT_POLICY, context } = options;
  try {
    return await fn();
  } catch (exc) {
    if (policy === 'raise') {
      throw exc;
    }
    getLogger().warn(
      `integrations async call failed (${stringifyError(exc)}); passthrough.`,
      context ?? {}
    );
    return fallback;
  }
}

function stringifyError(exc: unknown): string {
  if (exc instanceof Error) return exc.message;
  return String(exc);
}
