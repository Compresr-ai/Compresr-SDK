/** SDK configuration constants. */

export const API_KEY_PREFIX = 'cmp_';
export const DEFAULT_BASE_URL = 'https://api.compresr.ai';
export const DEFAULT_TIMEOUT = 300_000;

/**
 * Convenience model names. The backend is the authority — pass any string
 * as `compressionModelName` and the API will validate.
 */
export const MODELS = {
  LATTE: 'latte_v1',
  LATTE_V1: 'latte_v1',
  LATTE_V2: 'latte_v2',
} as const;

export type Model = (typeof MODELS)[keyof typeof MODELS];

export const HEADERS = {
  API_KEY: 'X-API-Key',
  CONTENT_TYPE: 'Content-Type',
  ACCEPT: 'Accept',
  USER_AGENT: 'User-Agent',
} as const;

export const STATUS_CODES = {
  OK: 200,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  VALIDATION_ERROR: 422,
  RATE_LIMITED: 429,
  SERVER_ERROR: 500,
} as const;
