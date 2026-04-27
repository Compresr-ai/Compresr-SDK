/**
 * API endpoint constants
 */
export const ENDPOINTS = {
  // Agnostic compression (no query required)
  COMPRESS_AGNOSTIC: '/api/compress/question-agnostic/',
  COMPRESS_AGNOSTIC_STREAM: '/api/compress/question-agnostic/stream',
  COMPRESS_AGNOSTIC_BATCH: '/api/compress/question-agnostic/batch',

  // Query-specific compression (query required)
  COMPRESS_QS: '/api/compress/question-specific/',
  COMPRESS_QS_STREAM: '/api/compress/question-specific/stream',
  COMPRESS_QS_BATCH: '/api/compress/question-specific/batch',

  // Agentic search
  SEARCH: '/api/compress/search/',
  SEARCH_STREAM: '/api/compress/search/stream',

  // Index management
  SEARCH_INDEX_CREATE: '/api/compress/search/indexes/{indexName}',
  SEARCH_INDEX_TASK: '/api/compress/search/indexes/tasks/{taskId}',
  SEARCH_INDEX_DELETE: '/api/compress/search/indexes/{indexName}',

  // Health
  SEARCH_HEALTH: '/api/compress/search/health',
} as const;

export type Endpoint = (typeof ENDPOINTS)[keyof typeof ENDPOINTS];
