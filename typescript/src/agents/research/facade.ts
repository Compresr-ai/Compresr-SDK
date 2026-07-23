/** `client.research` facade — thin wrapper over `ResearchAgent`. */

import { ResearchAgent } from './agent.js';
import type { ResearchAgentOptions } from './agent.js';
import type { ResearchResult } from './types.js';

const SEARCH_PROVIDER_ENV: Record<string, ReadonlyArray<string>> = {
  tavily: ['TAVILY_API_KEY'],
  brave: ['BRAVE_SEARCH_API_KEY', 'BRAVE_API_KEY'],
};

export interface ResearchRunOptions {
  search?: string | ResearchAgentOptions['searchTool'];
  maxSteps?: number;
  model?: string;
  compressSnippets?: boolean;
  compressionModel?: string;
  minCompressTokens?: number;
  maxContextTokens?: number;
  systemPrompt?: string;
}

export class ResearchFacade {
  private readonly engine: ResearchAgentOptions['engine'];

  constructor(engine: ResearchAgentOptions['engine']) {
    this.engine = engine;
  }

  async run(question: string, options: ResearchRunOptions = {}): Promise<ResearchResult> {
    const tool = await this.resolveSearchTool(options.search ?? 'tavily');
    const agent = new ResearchAgent({
      engine: this.engine,
      searchTool: tool,
      maxSteps: options.maxSteps ?? 10,
      ...(options.systemPrompt !== undefined ? { systemPrompt: options.systemPrompt } : {}),
      compressSnippets: options.compressSnippets ?? true,
      compressionModel: options.compressionModel ?? 'latte_v1',
      minCompressTokens: options.minCompressTokens ?? 100,
      maxContextTokens: options.maxContextTokens ?? 120_000,
    });
    return agent.run(question, options.model !== undefined ? { model: options.model } : {});
  }

  /** Single-shot: one search + forced answer (`maxSteps=2`). */
  async search(question: string, options: Omit<ResearchRunOptions, 'maxSteps'> = {}): Promise<ResearchResult> {
    return this.run(question, { ...options, maxSteps: 2 });
  }

  private async resolveSearchTool(
    search: string | ResearchAgentOptions['searchTool']
  ): Promise<ResearchAgentOptions['searchTool']> {
    if (typeof search !== 'string') return search;
    const provider = search.toLowerCase();
    if (!(provider in SEARCH_PROVIDER_ENV)) {
      throw new Error(
        `unsupported search provider '${search}'; expected one of ${Object.keys(SEARCH_PROVIDER_ENV).join(', ')} or a LangChain tool`
      );
    }
    const { WebSearchTool } = await import('../tools/web-search.js');
    let apiKey: string | undefined;
    for (const env of SEARCH_PROVIDER_ENV[provider]) {
      const v = process.env[env];
      if (v) {
        apiKey = v;
        break;
      }
    }
    const opts = apiKey !== undefined ? { apiKey } : {};
    const tool =
      provider === 'tavily'
        ? await WebSearchTool.tavily(opts)
        : await WebSearchTool.brave(opts);
    return tool as ResearchAgentOptions['searchTool'];
  }
}
