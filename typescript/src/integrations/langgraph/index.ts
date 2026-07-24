/**
 * LangGraph.js integration for Compresr.
 *
 *     npm install @langchain/langgraph @langchain/core   # (peers)
 *
 * LangGraph 1.0+ uses the same middleware mechanism as LangChain's
 * `createAgent`. Both middlewares below are re-exported from
 * `@compresr/sdk/integrations/langchain` so they're discoverable here
 * too — import either path you prefer.
 */
export {
  compresrToolMiddleware,
  compresrSummarizationMiddleware,
  compresrPromptMiddleware,
  type CompresrMiddleware,
  type CompresrToolMiddlewareOptions,
  type CompresrSummarizationMiddlewareOptions,
  type CompresrPromptMiddlewareOptions,
} from '../langchain/middleware.js';

export { makeCompresrNode, type MakeCompresrNodeOptions } from './nodes.js';
export { makeCompresrNode as compresrNode } from './nodes.js';
export {
  CompresrCheckpointSerializer,
  type CompresrCheckpointSerializerOptions,
} from './checkpoint.js';
export { compresrHandoffTool, type CompresrHandoffToolOptions } from './handoff.js';
export { CompresrStore, type CompresrStoreOptions } from './store.js';

export type { ErrorPolicy } from '../_shared/index.js';
