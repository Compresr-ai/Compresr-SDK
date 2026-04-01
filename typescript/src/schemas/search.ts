/**
 * Agentic Search schemas
 *
 * Schemas for agentic search endpoints (macchiato_v1).
 */
import { z } from 'zod';
import { BaseResponseSchema } from './common.js';

// =============================================================================
// Request Schemas
// =============================================================================

/**
 * Search request schema
 */
export const SearchRequestSchema = z.object({
  query: z.string().min(1, 'query must not be empty'),
  index_name: z.string().min(1, 'index_name must not be empty'),
  compression_model_name: z.string(),
  max_time_s: z.number().min(0.1).max(30).default(4.5),
  source: z.string().default('sdk:typescript'),
});

export type SearchRequest = z.infer<typeof SearchRequestSchema>;

/**
 * Index creation payload
 */
export const IndexCreatePayloadSchema = z.object({
  chunks: z.array(z.string()),
  candidate_questions: z.array(z.string()).optional(),
  source_docs: z.record(z.string(), z.unknown()).optional(),
});

export type IndexCreatePayload = z.infer<typeof IndexCreatePayloadSchema>;

// =============================================================================
// Result Schemas
// =============================================================================

/**
 * Search result
 */
export const SearchResultSchema = z.object({
  chunks: z.array(z.string()),
  chunk_ids: z.array(z.string()).default([]),
  chunks_count: z.number(),
  original_tokens: z.number(),
  compressed_tokens: z.number(),
  compression_ratio: z.number(),
  duration_ms: z.number().default(0),
  index_name: z.string(),
  cost: z.record(z.string(), z.unknown()).optional(),
});

export type SearchResult = z.infer<typeof SearchResultSchema>;

/**
 * Index task status
 */
export const IndexTaskStatusSchema = z.object({
  status: z.enum(['pending', 'processing', 'completed', 'failed']),
  task_id: z.string(),
  index_name: z.string().optional(),
  num_chunks: z.number().optional(),
  error: z.string().optional(),
});

export type IndexTaskStatus = z.infer<typeof IndexTaskStatusSchema>;

// =============================================================================
// Response Schemas
// =============================================================================

/**
 * Search response
 */
export const SearchResponseSchema = BaseResponseSchema.extend({
  data: SearchResultSchema.nullable(),
});

export type SearchResponse = z.infer<typeof SearchResponseSchema>;

/**
 * Index creation response
 */
export const IndexCreateResponseSchema = z.object({
  task_id: z.string(),
  status: z.string(),
});

export type IndexCreateResponse = z.infer<typeof IndexCreateResponseSchema>;
