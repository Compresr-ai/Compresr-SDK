/**
 * Pluggable logger interface.
 *
 * The SDK emits warnings (cleartext base URL, error-policy passthroughs,
 * unknown provider hints, etc.) via this logger. The default sink writes
 * to `console.warn` / `console.error` with a `[compresr]` prefix. Replace
 * it with `setLogger({ warn, error })` to integrate with pino, winston,
 * or any production logger.
 */

/* eslint-disable no-console -- this module is the canonical console sink. */

export interface CompresrLogger {
  warn(message: string, meta?: Record<string, unknown>): void;
  error(message: string, meta?: Record<string, unknown>): void;
}

const LOG_PREFIX = '[compresr]';

let logger: CompresrLogger = {
  warn: (message, meta) => {
    if (meta && Object.keys(meta).length > 0) {
      console.warn(`${LOG_PREFIX} ${message}`, meta);
    } else {
      console.warn(`${LOG_PREFIX} ${message}`);
    }
  },
  error: (message, meta) => {
    if (meta && Object.keys(meta).length > 0) {
      console.error(`${LOG_PREFIX} ${message}`, meta);
    } else {
      console.error(`${LOG_PREFIX} ${message}`);
    }
  },
};

/** Replace the global Compresr logger. */
export function setLogger(next: CompresrLogger): void {
  logger = next;
}

/** Current logger. Internal callers use this; user-facing modules should not. */
export function getLogger(): CompresrLogger {
  return logger;
}
