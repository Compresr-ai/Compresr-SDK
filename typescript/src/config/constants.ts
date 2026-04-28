/**
 * SDK configuration constants
 */

/** API key prefix for validation */
export const API_KEY_PREFIX = 'cmp_';

/** Default base URL for Compresr API */
export const DEFAULT_BASE_URL = 'https://api.compresr.ai';

/** Default request timeout in milliseconds */
export const DEFAULT_TIMEOUT = 60_000;

/** Stream request timeout in milliseconds */
export const STREAM_TIMEOUT = 300_000;

/**
 * Available compression and search models
 */
export const MODELS = {
  // Compression models
  ESPRESSO: 'espresso_v1',
  LATTE: 'latte_v1',

  // Search models
  MACCHIATO: 'macchiato_v1',

  // Agentic models
  AGENTIC_HISTORY_LINGUA: 'agentic_history_lingua',
  AGENTIC_TOOL_OUTPUT_GEMFILTER: 'agentic_tool_output_gemfilter',
  AGENTIC_TOOL_OUTPUT_LINGUA: 'agentic_tool_output_lingua',
  AGENTIC_TOOL_DISCOVERY_SAT: 'agentic_tool_discovery_sat',
} as const;

export type CompressionModel = typeof MODELS.ESPRESSO | typeof MODELS.LATTE;
export type SearchModel = typeof MODELS.MACCHIATO;
export type Model = (typeof MODELS)[keyof typeof MODELS];

export const ALLOWED_COMPRESSION_MODELS: ReadonlySet<string> = new Set([
  MODELS.ESPRESSO,
  MODELS.LATTE,
]);

export const ALLOWED_SEARCH_MODELS: ReadonlySet<string> = new Set([
  MODELS.MACCHIATO,
]);

export const QUERY_REQUIRED_MODELS: ReadonlySet<string> = new Set([
  MODELS.LATTE,
]);

export const COARSE_SUPPORTED_MODELS: ReadonlySet<string> = new Set([
  MODELS.LATTE,
]);

export const AGNOSTIC_ENDPOINT_MODELS: ReadonlySet<string> = new Set([
  MODELS.ESPRESSO,
]);

export const QS_ENDPOINT_MODELS: ReadonlySet<string> = new Set([
  MODELS.LATTE,
]);

/**
 * HTTP header names
 */
export const HEADERS = {
  API_KEY: 'X-API-Key',
  CONTENT_TYPE: 'Content-Type',
  ACCEPT: 'Accept',
  USER_AGENT: 'User-Agent',
} as const;

/**
 * HTTP status codes for error handling
 */
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
