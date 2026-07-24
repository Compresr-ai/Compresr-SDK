/** Client construction + integration defaults. */
import { CompressionClient } from '../../clients/compression.js';
import type { HttpClientOptions } from '../../http/client.js';

export const DEFAULT_MODEL = 'latte_v1';
export const DEFAULT_RATIO = 0.5;
export const DEFAULT_MIN_TOKENS = 200;
export const BATCH_LIMIT = 100;

export interface BuildClientOptions {
  apiKey?: string;
  baseUrl?: string;
  caller?: string;
}

export function buildClient(options: BuildClientOptions): CompressionClient {
  const { apiKey, baseUrl, caller = 'Compresr integration' } = options;
  if (!apiKey) {
    throw new Error(`Either \`apiKey\` or \`client\` is required for ${caller}.`);
  }
  const clientOpts: HttpClientOptions = { apiKey };
  if (baseUrl) clientOpts.baseUrl = baseUrl;
  return new CompressionClient(clientOpts);
}
