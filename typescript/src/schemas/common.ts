/**
 * Common/shared schemas
 */
import { z } from 'zod';

/**
 * Base API response schema
 */
export const BaseResponseSchema = z.object({
  success: z.boolean(),
});

export type BaseResponse = z.infer<typeof BaseResponseSchema>;

/**
 * Stream chunk schema for SSE responses
 */
export const StreamChunkSchema = z.object({
  content: z.string(),
  done: z.boolean().default(false),
  error: z.string().optional(),
});

export type StreamChunk = z.infer<typeof StreamChunkSchema>;

/**
 * Health check response
 */
export const HealthResponseSchema = z.object({
  status: z.string(),
  timestamp: z.string().optional(),
});

export type HealthResponse = z.infer<typeof HealthResponseSchema>;
