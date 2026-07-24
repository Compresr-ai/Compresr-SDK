/** Compression schemas — mirror backend. */
import { z } from 'zod';

import { BaseResponseSchema } from './common.js';

export const CompressRequestSchema = z.object({
  context: z.string().min(1, 'context must not be empty'),
  query: z.string().min(1, 'query must not be empty').optional(),
  compression_model_name: z.string(),
  target_compression_ratio: z.number().nonnegative().optional(),
  coarse: z.boolean().optional(),
  heuristic_chunking: z.boolean().optional(),
  disable_placeholders: z.boolean().optional(),
  // latte_v2-only knobs. Backend returns 422 if sent to a model that
  // doesn't support them (currently latte_v2 only).
  dynamic: z.boolean().optional(),
  dynamic_min_ratio: z.number().optional(),
  dynamic_max_ratio: z.number().optional(),
  source: z.string().default('sdk:typescript'),
});

export type CompressRequest = z.infer<typeof CompressRequestSchema>;

export const CompressBatchInputSchema = z.object({
  context: z.string().min(1, 'context must not be empty'),
  query: z.string().min(1, 'query must not be empty').optional(),
});

export type CompressBatchInput = z.infer<typeof CompressBatchInputSchema>;

export const CompressBatchRequestSchema = z.object({
  inputs: z.array(CompressBatchInputSchema).min(1).max(100),
  compression_model_name: z.string(),
  target_compression_ratio: z.number().nonnegative().optional(),
  coarse: z.boolean().optional(),
  heuristic_chunking: z.boolean().optional(),
  disable_placeholders: z.boolean().optional(),
  // latte_v2-only knobs (shared across the whole batch).
  dynamic: z.boolean().optional(),
  dynamic_min_ratio: z.number().optional(),
  dynamic_max_ratio: z.number().optional(),
  source: z.string().default('sdk:typescript'),
});

export type CompressBatchRequest = z.infer<typeof CompressBatchRequestSchema>;

export const CompressResultSchema = z.object({
  original_context: z.string().nullish(),
  compressed_context: z.string(),
  original_tokens: z.number(),
  compressed_tokens: z.number(),
  actual_compression_ratio: z.number(),
  tokens_saved: z.number(),
  duration_ms: z.number(),
  target_compression_ratio: z.number().nullish(),
});

export type CompressResult = z.infer<typeof CompressResultSchema>;

export const CompressBatchItemResultSchema = z.object({
  original_context: z.string().nullish(),
  compressed_context: z.string(),
  original_tokens: z.number(),
  compressed_tokens: z.number(),
  actual_compression_ratio: z.number(),
  tokens_saved: z.number(),
  duration_ms: z.number(),
});

export type CompressBatchItemResult = z.infer<typeof CompressBatchItemResultSchema>;

export const CompressBatchResultSchema = z.object({
  results: z.array(CompressBatchItemResultSchema),
  total_original_tokens: z.number(),
  total_compressed_tokens: z.number(),
  total_tokens_saved: z.number(),
  average_compression_ratio: z.number(),
  count: z.number(),
});

export type CompressBatchResult = z.infer<typeof CompressBatchResultSchema>;

export const CompressResponseSchema = BaseResponseSchema.extend({
  data: CompressResultSchema.nullable(),
});

export type CompressResponse = z.infer<typeof CompressResponseSchema>;

export const CompressBatchResponseSchema = BaseResponseSchema.extend({
  data: CompressBatchResultSchema.nullable(),
});

export type CompressBatchResponse = z.infer<typeof CompressBatchResponseSchema>;
