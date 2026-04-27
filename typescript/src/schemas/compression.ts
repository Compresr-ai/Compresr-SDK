/**
 * Compression schemas
 *
 * Schemas for compression endpoints, matching backend exactly.
 */
import { z } from 'zod';
import { BaseResponseSchema } from './common.js';

// =============================================================================
// Request Schemas
// =============================================================================

/**
 * Compress request schema (single context only)
 * For multiple contexts, use the /batch endpoint.
 */
export const CompressRequestSchema = z.object({
  context: z.string().min(1, 'context must not be empty'),
  compression_model_name: z.string(),
  query: z.string().min(1, 'query must not be empty').optional(),
  target_compression_ratio: z.number().nonnegative().optional(),
  coarse: z.boolean().optional(),
  source: z.string().default('sdk:typescript'),
});

export type CompressRequest = z.infer<typeof CompressRequestSchema>;

/**
 * Agnostic batch input (no query required)
 */
export const AgnosticBatchInputSchema = z.object({
  context: z.string().min(1, 'context must not be empty'),
});

export type AgnosticBatchInput = z.infer<typeof AgnosticBatchInputSchema>;

/**
 * Agnostic batch compression request
 */
export const AgnosticBatchRequestSchema = z.object({
  inputs: z.array(AgnosticBatchInputSchema).min(1).max(100),
  compression_model_name: z.string(),
  target_compression_ratio: z.number().nonnegative().optional(),
  source: z.string().default('sdk:typescript'),
});

export type AgnosticBatchRequest = z.infer<typeof AgnosticBatchRequestSchema>;

/**
 * Query-specific batch compression input (query required)
 */
export const CompressBatchInputSchema = z.object({
  context: z.string().min(1, 'context must not be empty'),
  query: z.string().min(1, 'query must not be empty'),
});

export type CompressBatchInput = z.infer<typeof CompressBatchInputSchema>;

/**
 * Batch compression request
 */
export const CompressBatchRequestSchema = z.object({
  inputs: z.array(CompressBatchInputSchema).min(1).max(100),
  compression_model_name: z.string(),
  target_compression_ratio: z.number().nonnegative().optional(),
  coarse: z.boolean().optional(),
  source: z.string().default('sdk:typescript'),
});

export type CompressBatchRequest = z.infer<typeof CompressBatchRequestSchema>;

// =============================================================================
// Result Schemas
// =============================================================================

/**
 * Single compression result
 */
export const CompressResultSchema = z.object({
  original_context: z.string(),
  compressed_context: z.string(),
  original_tokens: z.number(),
  compressed_tokens: z.number(),
  actual_compression_ratio: z.number(),
  tokens_saved: z.number(),
  duration_ms: z.number(),
  target_compression_ratio: z.number().nullish(),
});

export type CompressResult = z.infer<typeof CompressResultSchema>;

/**
 * Batch item result
 */
export const CompressBatchItemResultSchema = z.object({
  original_context: z.string(),
  compressed_context: z.string(),
  original_tokens: z.number(),
  compressed_tokens: z.number(),
  actual_compression_ratio: z.number(),
  tokens_saved: z.number(),
  duration_ms: z.number(),
});

export type CompressBatchItemResult = z.infer<typeof CompressBatchItemResultSchema>;

/**
 * Batch compression result with aggregated metrics
 */
export const CompressBatchResultSchema = z.object({
  results: z.array(CompressBatchItemResultSchema),
  total_original_tokens: z.number(),
  total_compressed_tokens: z.number(),
  total_tokens_saved: z.number(),
  average_compression_ratio: z.number(),
  count: z.number(),
});

export type CompressBatchResult = z.infer<typeof CompressBatchResultSchema>;

// =============================================================================
// Response Schemas
// =============================================================================

/**
 * Single compression response
 */
export const CompressResponseSchema = BaseResponseSchema.extend({
  data: CompressResultSchema.nullable(),
});

export type CompressResponse = z.infer<typeof CompressResponseSchema>;

/**
 * Batch compression response
 */
export const CompressBatchResponseSchema = BaseResponseSchema.extend({
  data: CompressBatchResultSchema.nullable(),
});

export type CompressBatchResponse = z.infer<typeof CompressBatchResponseSchema>;
