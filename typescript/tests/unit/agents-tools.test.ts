/**
 * Unit tests for ``WebSearchTool`` and ``createWebSearchTool`` — verifies the
 * Tavily/Brave factory paths and the clear error when a peer dep is missing.
 *
 * The Tavily/Brave packages aren't installed in this repo, so we use
 * vitest ``vi.mock`` to stub the dynamic imports. The shape we return is a
 * minimal duck-typed BaseTool ({name, invoke}) — what matters is the wiring.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  WebSearchTool,
  createWebSearchTool,
} from '../../src/agents/tools/web-search.js';
import { CompresrError } from '../../src/errors/index.js';

// Captures init kwargs for assertion (the public tool is a wrapper now).
const lastTavilyInit: Record<string, unknown> = {};
let tavilyInvokeReturn: unknown = { query: '', results: [] };

class FakeTavilySearch {
  readonly maxResults: number;
  readonly tavilyApiKey?: string;
  readonly includeDomains?: string[];
  readonly excludeDomains?: string[];
  constructor(init: Record<string, unknown>) {
    // Replace contents (don't reassign — keep the same object reference for tests).
    for (const k of Object.keys(lastTavilyInit)) delete lastTavilyInit[k];
    Object.assign(lastTavilyInit, init);
    this.maxResults = (init['maxResults'] as number) ?? 5;
    if (typeof init['tavilyApiKey'] === 'string') {
      this.tavilyApiKey = init['tavilyApiKey'];
    }
    if (Array.isArray(init['includeDomains'])) {
      this.includeDomains = init['includeDomains'] as string[];
    }
    if (Array.isArray(init['excludeDomains'])) {
      this.excludeDomains = init['excludeDomains'] as string[];
    }
  }
  async invoke(_args: { query: string }): Promise<unknown> {
    return tavilyInvokeReturn;
  }
}

class FakeBraveSearch {
  readonly apiKey: string;
  readonly searchKwargs: Record<string, unknown>;
  constructor(init: Record<string, unknown>) {
    this.apiKey = init['apiKey'] as string;
    this.searchKwargs = init['searchKwargs'] as Record<string, unknown>;
  }
}

vi.mock('@langchain/tavily', () => ({
  TavilySearch: FakeTavilySearch,
}));

vi.mock('@langchain/community/tools/brave_search', () => ({
  BraveSearch: FakeBraveSearch,
}));

describe('WebSearchTool.tavily', () => {
  it('threads apiKey + maxResults into the underlying TavilySearch', async () => {
    const prevTavilyKey = process.env['TAVILY_API_KEY'];
    delete process.env['TAVILY_API_KEY'];
    try {
      const tool = await WebSearchTool.tavily({
        apiKey: 'tvly-test',
        maxResults: 7,
      });
      // Wrapper has the canonical name + schema (parity with brave).
      expect(tool.name).toBe('tavily_search');
      expect(tool.schema).toBeDefined();
      // Init kwargs reached TavilySearch.
      expect(lastTavilyInit['maxResults']).toBe(7);
      expect(lastTavilyInit['tavilyApiKey']).toBe('tvly-test');
      // Env stays clean — the key is passed via constructor, not env.
      expect(process.env['TAVILY_API_KEY']).toBeUndefined();
    } finally {
      if (prevTavilyKey !== undefined) {
        process.env['TAVILY_API_KEY'] = prevTavilyKey;
      }
    }
  });

  it('threads allowed/blocked domains as include/exclude', async () => {
    await WebSearchTool.tavily({
      apiKey: 'tvly-test',
      allowedDomains: ['nytimes.com'],
      blockedDomains: ['example.org'],
    });
    expect(lastTavilyInit['includeDomains']).toEqual(['nytimes.com']);
    expect(lastTavilyInit['excludeDomains']).toEqual(['example.org']);
  });

  it('flattens the dict response into plain text blocks', async () => {
    // latte_v1 no-ops on JSON-shaped input, so the wrapper must flatten.
    tavilyInvokeReturn = {
      query: 'x',
      results: [
        {
          title: 'SAS 9.4 cohort',
          url: 'https://example.org/a',
          content: 'All analyses used SAS 9.4 (Cary, NC).',
        },
        {
          title: 'BRCA1 mutation study',
          url: 'https://example.org/b',
          content: 'Logistic regression with SAS 9.1.',
        },
      ],
    };
    const tool = await WebSearchTool.tavily({ apiKey: 'tvly-test' });
    const out = (await tool.invoke({ query: 'anything' })) as string;
    expect(typeof out).toBe('string');
    expect(out.trimStart().startsWith('{')).toBe(false);
    expect(out).toContain('SAS 9.4 cohort');
    expect(out).toContain('https://example.org/a');
    expect(out).toContain('All analyses used SAS 9.4');
    expect(out).toContain('BRCA1 mutation study');
    // Blocks separated by blank lines.
    expect(out).toContain('\n\n');
  });

  it('falls back to JSON.stringify for unknown shapes', async () => {
    tavilyInvokeReturn = { error: 'rate-limited', code: 429 };
    const tool = await WebSearchTool.tavily({ apiKey: 'tvly-test' });
    const out = (await tool.invoke({ query: 'anything' })) as string;
    expect(typeof out).toBe('string');
    expect(out).toContain('rate-limited');
  });
});

describe('WebSearchTool.brave', () => {
  it('returns a Brave tool with api key threaded through', async () => {
    const tool = (await WebSearchTool.brave({
      apiKey: 'brave-test',
      maxResults: 4,
    })) as unknown as FakeBraveSearch;
    expect(tool).toBeInstanceOf(FakeBraveSearch);
    expect(tool.apiKey).toBe('brave-test');
    expect(tool.searchKwargs).toEqual({ count: 4 });
  });

  it('throws a clear error when apiKey is missing and no env var is set', async () => {
    delete process.env['BRAVE_SEARCH_API_KEY'];
    delete process.env['BRAVE_API_KEY'];
    await expect(WebSearchTool.brave({})).rejects.toBeInstanceOf(CompresrError);
  });
});

describe('createWebSearchTool', () => {
  it('dispatches to tavily provider', async () => {
    const tool = await createWebSearchTool('tavily', { apiKey: 'tvly-test' });
    // Now wrapped — the dispatch path mints the canonical tavily_search tool.
    expect(tool.name).toBe('tavily_search');
  });

  it('dispatches to brave provider', async () => {
    const tool = (await createWebSearchTool('brave', {
      apiKey: 'brave-test',
    })) as unknown as FakeBraveSearch;
    expect(tool).toBeInstanceOf(FakeBraveSearch);
  });
});
