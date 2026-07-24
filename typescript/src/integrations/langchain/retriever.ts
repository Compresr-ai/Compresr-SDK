/**
 * `BaseDocumentCompressor` implementation for `ContextualCompressionRetriever`.
 *
 * Mirrors Python `CompresrExtractor`. One Compresr batch call
 * replaces N LLM extraction calls.
 */
import { Document } from '@langchain/core/documents';

import type { CompressionClient } from '../../clients/compression.js';
import { getLogger } from '../../logger.js';
import {
  BATCH_LIMIT,
  buildClient,
  DEFAULT_MIN_TOKENS,
  DEFAULT_MODEL,
  DEFAULT_POLICY,
  DEFAULT_RATIO,
  type ErrorPolicy,
  estimateTokens,
} from '../_shared/index.js';

type DocumentType = Document;

export interface CompresrExtractorOptions {
  apiKey?: string;
  client?: CompressionClient;
  baseUrl?: string;
  compressionModel?: string;
  targetCompressionRatio?: number;
  minTokens?: number;
  coarse?: boolean;
  onError?: ErrorPolicy;
  /** Drop documents whose compressed content is empty. */
  dropBelowMin?: boolean;
}

/**
 * Compress retrieved `Document`s with query-aware compression.
 *
 * @example
 * ```ts
 * import { ContextualCompressionRetriever } from '@langchain/classic/retrievers/contextual_compression';
 * import { CompresrExtractor } from '@compresr/sdk/integrations/langchain';
 *
 * const compressor = new CompresrExtractor({
 *   apiKey: process.env.COMPRESR_API_KEY!,
 * });
 * const retriever = new ContextualCompressionRetriever({
 *   baseCompressor: compressor,
 *   baseRetriever: vectorStore.asRetriever({ k: 8 }),
 * });
 * ```
 */
export class CompresrExtractor {
  private readonly compresr: CompressionClient;
  private readonly compressionModel: string;
  private readonly targetCompressionRatio: number;
  private readonly minTokens: number;
  private readonly coarse: boolean | undefined;
  private readonly onError: ErrorPolicy;
  private readonly dropBelowMin: boolean;

  constructor(options: CompresrExtractorOptions) {
    const {
      apiKey,
      client,
      baseUrl,
      compressionModel = DEFAULT_MODEL,
      targetCompressionRatio = DEFAULT_RATIO,
      minTokens = DEFAULT_MIN_TOKENS,
      coarse,
      onError = DEFAULT_POLICY,
      dropBelowMin = false,
    } = options;

    this.compresr =
      client ?? buildClient({ apiKey, baseUrl, caller: 'CompresrExtractor' });
    this.compressionModel = compressionModel;
    this.targetCompressionRatio = targetCompressionRatio;
    this.minTokens = minTokens;
    this.coarse = coarse;
    this.onError = onError;
    this.dropBelowMin = dropBelowMin;
  }

  private partition(
    documents: readonly DocumentType[]
  ): { indices: number[]; eligible: DocumentType[] } {
    const indices: number[] = [];
    const eligible: DocumentType[] = [];
    documents.forEach((doc, i) => {
      const pc = (doc as { pageContent?: unknown }).pageContent;
      if (typeof pc === 'string' && estimateTokens(pc) >= this.minTokens) {
        indices.push(i);
        eligible.push(doc);
      }
    });
    return { indices, eligible };
  }

  private emit(doc: DocumentType, content: string): DocumentType {
    const md = (doc as { metadata?: Record<string, unknown> }).metadata ?? {};
    return new Document({
      pageContent: content,
      metadata: { ...md, compresr: true },
    });
  }

  async compressDocuments(
    documents: readonly DocumentType[],
    query: string
  ): Promise<DocumentType[]> {
    if (documents.length === 0) return [];

    const { indices, eligible } = this.partition(documents);
    if (eligible.length === 0) return [...documents];

    const out: DocumentType[] = [...documents];

    for (let start = 0; start < eligible.length; start += BATCH_LIMIT) {
      const chunk = eligible.slice(start, start + BATCH_LIMIT);
      let results: { compressed_context: string }[];
      try {
        const resp = await this.compresr.compressBatch({
          contexts: chunk.map((d) => (d as { pageContent: string }).pageContent),
          queries: query,
          compressionModelName: this.compressionModel,
          targetCompressionRatio: this.targetCompressionRatio,
          ...(this.coarse !== undefined ? { coarse: this.coarse } : {}),
        });
        results = (resp.data?.results ?? []);
      } catch (exc) {
        if (this.onError === 'raise') throw exc;
        getLogger().warn(
          `batch compress failed (${stringifyError(exc)}); passthrough ${chunk.length} docs.`
        );
        continue;
      }

      results.slice(0, chunk.length).forEach((item, offset) => {
        const globalIdx = indices[start + offset];
        const newText = item?.compressed_context;
        if (typeof newText === 'string' && newText && globalIdx !== undefined) {
          const doc = documents[globalIdx];
          if (doc !== undefined) {
            out[globalIdx] = this.emit(doc, newText);
          }
        }
      });
    }

    if (this.dropBelowMin) {
      return out.filter((d) => {
        const pc = (d as { pageContent?: unknown }).pageContent;
        return typeof pc === 'string' && pc.trim().length > 0;
      });
    }
    return out;
  }
}

function stringifyError(exc: unknown): string {
  return exc instanceof Error ? exc.message : String(exc);
}
