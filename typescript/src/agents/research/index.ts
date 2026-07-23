/**
 * Compresr research agent — multi-step web search with per-snippet compression.
 *
 * Public entry: `client.research.run(question)`.
 * Loop structure adapted from Perplexity `search_evals` (MIT,
 * https://github.com/perplexityai/search_evals).
 */

export { ResearchAgent } from './agent.js';
export type { ResearchAgentOptions, ResearchRunOptions as ResearchAgentRunOptions } from './agent.js';
export { ResearchFacade } from './facade.js';
export type { ResearchRunOptions } from './facade.js';
export { parseResearchOutput } from './parser.js';
export type { ParsedResearch } from './parser.js';
export { DEFAULT_RESEARCH_SYSTEM_PROMPT } from './prompts.js';
export type {
  Citation,
  ResearchResult,
  ResearchUsage,
  Step,
  StepKind,
} from './types.js';
