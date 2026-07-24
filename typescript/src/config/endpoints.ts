export const ENDPOINTS = {
  COMPRESS: '/api/compress/question-specific/',
  COMPRESS_STREAM: '/api/compress/question-specific/stream',
  COMPRESS_BATCH: '/api/compress/question-specific/batch',
} as const;

export type Endpoint = (typeof ENDPOINTS)[keyof typeof ENDPOINTS];
