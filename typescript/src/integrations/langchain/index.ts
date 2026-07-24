/**
 * LangChain.js integration for Compresr.
 *
 *     npm install @langchain/core langchain   # (peers)
 */
export {
  compresrToolMiddleware,
  compresrSummarizationMiddleware,
  compresrPromptMiddleware,
  type CompresrMiddleware,
  type CompresrToolMiddlewareOptions,
  type CompresrSummarizationMiddlewareOptions,
  type CompresrPromptMiddlewareOptions,
  type ModelRequest,
  type ModelHandler,
  type ToolCall,
  type ToolCallRequest,
  type ToolHandler,
  type ToolQueryExtractor,
  type SummaryQueryExtractor,
} from './middleware.js';
export {
  wrapToolWithCompression,
  compressToolOutput,
  type WrapToolOptions,
  type ToolDecorator,
} from './wrappers.js';
export {
  CompresrExtractor,
  type CompresrExtractorOptions,
} from './retriever.js';

export type { ErrorPolicy } from '../_shared/index.js';
