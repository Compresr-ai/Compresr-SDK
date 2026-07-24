export interface RetryConfig {
  maxRetries?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
  multiplier?: number;
  /** Fractional symmetric jitter in `[0, 1]` (e.g. `0.25` = ±25%). */
  jitter?: number;
  retryOnStatus?: number[];
  respectRetryAfter?: boolean;
}

export interface ResolvedRetryConfig {
  maxRetries: number;
  initialBackoffMs: number;
  maxBackoffMs: number;
  multiplier: number;
  jitter: number;
  retryOnStatus: ReadonlySet<number>;
  respectRetryAfter: boolean;
}

export const DEFAULT_RETRY_CONFIG: ResolvedRetryConfig = {
  maxRetries: 3,
  initialBackoffMs: 500,
  maxBackoffMs: 30_000,
  multiplier: 2.0,
  jitter: 0.25,
  retryOnStatus: new Set<number>([429, 503]),
  respectRetryAfter: true,
};

export function resolveRetryConfig(cfg?: RetryConfig): ResolvedRetryConfig {
  if (!cfg) return DEFAULT_RETRY_CONFIG;
  const merged: ResolvedRetryConfig = {
    maxRetries: cfg.maxRetries ?? DEFAULT_RETRY_CONFIG.maxRetries,
    initialBackoffMs: cfg.initialBackoffMs ?? DEFAULT_RETRY_CONFIG.initialBackoffMs,
    maxBackoffMs: cfg.maxBackoffMs ?? DEFAULT_RETRY_CONFIG.maxBackoffMs,
    multiplier: cfg.multiplier ?? DEFAULT_RETRY_CONFIG.multiplier,
    jitter: cfg.jitter ?? DEFAULT_RETRY_CONFIG.jitter,
    retryOnStatus: cfg.retryOnStatus
      ? new Set<number>(cfg.retryOnStatus)
      : DEFAULT_RETRY_CONFIG.retryOnStatus,
    respectRetryAfter: cfg.respectRetryAfter ?? DEFAULT_RETRY_CONFIG.respectRetryAfter,
  };
  if (merged.maxRetries < 0) throw new RangeError('maxRetries must be >= 0');
  if (merged.initialBackoffMs < 0) throw new RangeError('initialBackoffMs must be >= 0');
  if (merged.maxBackoffMs < 0) throw new RangeError('maxBackoffMs must be >= 0');
  if (merged.multiplier < 1) throw new RangeError('multiplier must be >= 1');
  if (merged.jitter < 0 || merged.jitter > 1) {
    throw new RangeError('jitter must be in [0, 1]');
  }
  return merged;
}

export function computeBackoffMs(
  attempt: number,
  cfg: ResolvedRetryConfig,
  retryAfterSeconds?: number
): number {
  if (retryAfterSeconds !== undefined && cfg.respectRetryAfter) {
    const ms = Math.max(0, retryAfterSeconds * 1000);
    return Math.min(ms, cfg.maxBackoffMs);
  }
  const raw = cfg.initialBackoffMs * Math.pow(cfg.multiplier, attempt);
  const capped = Math.min(raw, cfg.maxBackoffMs);
  if (cfg.jitter === 0) return capped;
  const spread = capped * cfg.jitter;
  return Math.max(0, capped + (Math.random() * 2 - 1) * spread);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
