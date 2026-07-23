/**
 * ``WebSearchTool`` — backed by Tavily, Brave, or Amazon Bedrock AgentCore.
 *
 * Returns a LangChain.js ``BaseTool`` whose output flows through Compresr's
 * ``compresrToolMiddleware`` automatically when used with the SDK's engine
 * (``client.messages.create`` / ``client.chat.completions.create`` /
 * ``client.run``).
 *
 * Provider-native server tools (Anthropic ``web_search_20250305``, OpenAI
 * ``web_search_preview``, Gemini ``google_search``) are intentionally NOT
 * supported here — they execute server-side and return opaque/encrypted
 * content that Compresr cannot read or compress. Use a real search API
 * (Tavily or Brave) so the result is plaintext we can compress.
 *
 * Mirrors Python ``compresr/agents/tools/web_search.py``.
 */
import type { StructuredToolInterface } from '@langchain/core/tools';

import { CompresrError } from '../../errors/index.js';
import {
  buildAgentCoreClient,
  type AgentCoreConfig,
} from './_agentcore.js';

/** Public return type for the web-search factories — any tool that satisfies
 * LangChain.js's ``StructuredToolInterface`` is acceptable. We use the
 * interface form (not the abstract class) so we avoid generic parameter
 * gymnastics at the call site. */
export type WebSearchToolInstance = StructuredToolInterface;

export interface TavilyOptions {
  apiKey?: string;
  maxResults?: number;
  allowedDomains?: ReadonlyArray<string>;
  blockedDomains?: ReadonlyArray<string>;
  /** Additional kwargs forwarded to the underlying LangChain tool. */
  extra?: Record<string, unknown>;
}

export interface BraveOptions {
  apiKey?: string;
  maxResults?: number;
  /** Additional kwargs forwarded to the underlying LangChain tool. */
  extra?: Record<string, unknown>;
}

/**
 * Options for ``WebSearchTool.agentcore`` (Amazon Bedrock AgentCore web
 * search). Each gateway field is resolved with precedence
 * **explicit option → environment variable**:
 *
 * | Field          | Env var (fallback)                                |
 * |----------------|---------------------------------------------------|
 * | gatewayUrl     | AGENTCORE_GATEWAY_MCP_URL (GATEWAY_MCP_URL)       |
 * | cognitoTokenUrl| AGENTCORE_COGNITO_TOKEN_URL (COGNITO_TOKEN_URL)   |
 * | clientId       | AGENTCORE_COGNITO_CLIENT_ID (COGNITO_CLIENT_ID)   |
 * | clientSecret   | AGENTCORE_COGNITO_CLIENT_SECRET (COGNITO_CLIENT_SECRET) |
 * | scope          | AGENTCORE_COGNITO_SCOPE (COGNITO_SCOPE)           |
 *
 * ``allowedDomains`` / ``blockedDomains`` are accepted for signature parity
 * with Tavily/Brave but are **unsupported** by AgentCore web search (no native
 * domain filter) — use Tavily if you need domain filtering.
 */
export interface AgentCoreOptions {
  gatewayUrl?: string;
  cognitoTokenUrl?: string;
  clientId?: string;
  clientSecret?: string;
  scope?: string;
  maxResults?: number;
  /** Unsupported by AgentCore — accepted for parity, otherwise ignored. */
  allowedDomains?: ReadonlyArray<string>;
  /** Unsupported by AgentCore — accepted for parity, otherwise ignored. */
  blockedDomains?: ReadonlyArray<string>;
  /** Reserved for forward compatibility; currently unused. */
  extra?: Record<string, unknown>;
}

/** Flatten ``{results: [{title, url, content}, ...]}`` into blank-line
 * -separated plain-text blocks. ``latte_v1`` no-ops on JSON-shaped input,
 * so this reshape is what makes Tavily output compressible by the middleware. */
function flattenSearchResults(out: unknown): string {
  if (typeof out === 'string') return out;
  if (out && typeof out === 'object' && 'results' in out) {
    const results = (out as { results?: unknown }).results;
    if (Array.isArray(results)) {
      const parts: string[] = [];
      for (const r of results) {
        if (!r || typeof r !== 'object') continue;
        const rr = r as Record<string, unknown>;
        const title = typeof rr['title'] === 'string' ? rr['title'].trim() : '';
        const url = typeof rr['url'] === 'string' ? rr['url'].trim() : '';
        const content =
          typeof rr['content'] === 'string'
            ? rr['content'].trim()
            : typeof rr['snippet'] === 'string'
              ? rr['snippet'].trim()
              : '';
        const block = [title, url, content].filter(Boolean).join('\n');
        if (block) parts.push(block);
      }
      if (parts.length > 0) return parts.join('\n\n');
    }
  }
  return JSON.stringify(out);
}

async function buildTavily(options: TavilyOptions): Promise<WebSearchToolInstance> {
  let mod: Record<string, unknown>;
  try {
    mod = await import('@langchain/tavily');
  } catch {
    throw new CompresrError(
      "WebSearchTool.tavily requires '@langchain/tavily'. " +
        'Install with: npm install @langchain/tavily',
      'missing_peer_dependency'
    );
  }
  const TavilySearch = (mod['TavilySearch'] ?? mod['default']) as
    | (new (init: Record<string, unknown>) => unknown)
    | undefined;
  if (typeof TavilySearch !== 'function') {
    throw new CompresrError(
      "'@langchain/tavily' is missing TavilySearch — upgrade the package.",
      'missing_peer_dependency'
    );
  }
  const init: Record<string, unknown> = {
    maxResults: options.maxResults ?? 5,
  };
  if (options.apiKey) {
    init['tavilyApiKey'] = options.apiKey;
  }
  if (options.allowedDomains !== undefined) {
    init['includeDomains'] = [...options.allowedDomains];
  }
  if (options.blockedDomains !== undefined) {
    init['excludeDomains'] = [...options.blockedDomains];
  }
  // Merge ``extra`` first so trusted fields (apiKey, includeDomains, etc.)
  // win on collision — see audit TS-M3.
  const merged = { ...options.extra, ...init };
  const base = new TavilySearch(merged) as {
    invoke: (args: { query: string }) => Promise<unknown>;
  };

  // Wrap with a tool() so the returned ToolMessage content is plain text
  // (latte_v1 no-ops on JSON), and so an explicit ``query`` schema reaches
  // the LLM (parity with brave).
  const { tool } = (await import('@langchain/core/tools')) as {
    tool: (
      fn: (input: { query: string }) => Promise<string>,
      meta: { name: string; description: string; schema: unknown }
    ) => WebSearchToolInstance;
  };
  const { z } = (await import('zod')) as { z: typeof import('zod').z };

  return tool(
    async ({ query }) => flattenSearchResults(await base.invoke({ query })),
    {
      name: 'tavily_search',
      description:
        'Tavily web search. Use to answer questions about current events.',
      schema: z.object({
        query: z.string().max(2000).describe('The search query string.'),
      }),
    }
  );
}

async function buildBrave(options: BraveOptions): Promise<WebSearchToolInstance> {
  let mod: Record<string, unknown>;
  try {
    mod = (await import(
      // @ts-expect-error — optional peer; types not guaranteed at compile time
      '@langchain/community/tools/brave_search'
    )) as Record<string, unknown>;
  } catch {
    throw new CompresrError(
      "WebSearchTool.brave requires '@langchain/community'. " +
        'Install with: npm install @langchain/community',
      'missing_peer_dependency'
    );
  }
  const BraveSearch = (mod['BraveSearch'] ?? mod['default']) as
    | (new (init: Record<string, unknown>) => unknown)
    | undefined;
  if (typeof BraveSearch !== 'function') {
    throw new CompresrError(
      "'@langchain/community' is missing BraveSearch — upgrade the package.",
      'missing_peer_dependency'
    );
  }
  const key =
    options.apiKey ??
    process.env['BRAVE_SEARCH_API_KEY'] ??
    process.env['BRAVE_API_KEY'];
  if (!key) {
    throw new CompresrError(
      'Brave requires apiKey or BRAVE_SEARCH_API_KEY env var.',
      'missing_api_key'
    );
  }
  const init: Record<string, unknown> = {
    apiKey: key,
    searchKwargs: { count: options.maxResults ?? 5 },
  };
  // Merge ``extra`` first so trusted fields (apiKey, searchKwargs) win on
  // collision — see audit TS-M3.
  const merged = { ...options.extra, ...init };
  return new BraveSearch(merged) as WebSearchToolInstance;
}

/** Clamp the requested result count into AgentCore's supported 1–25 range. */
function clampMaxResults(value: number | undefined): number {
  const n = value ?? 5;
  if (!Number.isFinite(n)) return 5;
  return Math.min(25, Math.max(1, Math.trunc(n)));
}

/** Each AgentCore config field and the env vars it falls back to. */
const AGENTCORE_ENV: ReadonlyArray<{
  key: keyof Omit<AgentCoreConfig, 'maxResults'>;
  optionLabel: string;
  envVars: ReadonlyArray<string>;
}> = [
  {
    key: 'gatewayUrl',
    optionLabel: 'gatewayUrl',
    envVars: ['AGENTCORE_GATEWAY_MCP_URL', 'GATEWAY_MCP_URL'],
  },
  {
    key: 'cognitoTokenUrl',
    optionLabel: 'cognitoTokenUrl',
    envVars: ['AGENTCORE_COGNITO_TOKEN_URL', 'COGNITO_TOKEN_URL'],
  },
  {
    key: 'clientId',
    optionLabel: 'clientId',
    envVars: ['AGENTCORE_COGNITO_CLIENT_ID', 'COGNITO_CLIENT_ID'],
  },
  {
    key: 'clientSecret',
    optionLabel: 'clientSecret',
    envVars: ['AGENTCORE_COGNITO_CLIENT_SECRET', 'COGNITO_CLIENT_SECRET'],
  },
  {
    key: 'scope',
    optionLabel: 'scope',
    envVars: ['AGENTCORE_COGNITO_SCOPE', 'COGNITO_SCOPE'],
  },
];

/** URL-valued config fields that must use ``https://`` to avoid leaking the
 * Cognito client secret / bearer token in cleartext. */
const AGENTCORE_URL_FIELDS: ReadonlyArray<keyof Omit<AgentCoreConfig, 'maxResults'>> =
  ['gatewayUrl', 'cognitoTokenUrl'];

/** Resolve config with precedence explicit option → env var; collect misses. */
function resolveAgentCoreConfig(options: AgentCoreOptions): AgentCoreConfig {
  const resolved: Record<string, string> = {};
  const missing: string[] = [];
  for (const { key, optionLabel, envVars } of AGENTCORE_ENV) {
    const fromOption = options[key];
    let value: string | undefined =
      typeof fromOption === 'string' && fromOption.length > 0
        ? fromOption
        : undefined;
    if (value === undefined) {
      for (const envVar of envVars) {
        const candidate = process.env[envVar];
        if (candidate) {
          value = candidate;
          break;
        }
      }
    }
    if (value === undefined) {
      missing.push(`${optionLabel} (env: ${envVars.join(' or ')})`);
    } else {
      resolved[key] = value;
    }
  }
  if (missing.length > 0) {
    throw new CompresrError(
      'WebSearchTool.agentcore is missing required config: ' +
        missing.join('; ') +
        '.',
      'missing_config'
    );
  }
  for (const field of AGENTCORE_URL_FIELDS) {
    if (!resolved[field].startsWith('https://')) {
      throw new CompresrError(
        `WebSearchTool.agentcore: '${field}' must use https:// ` +
          '(refusing to send credentials in cleartext).',
        'invalid_config'
      );
    }
  }
  return {
    gatewayUrl: resolved['gatewayUrl'],
    cognitoTokenUrl: resolved['cognitoTokenUrl'],
    clientId: resolved['clientId'],
    clientSecret: resolved['clientSecret'],
    scope: resolved['scope'],
    maxResults: clampMaxResults(options.maxResults),
  };
}

async function buildAgentCore(
  options: AgentCoreOptions
): Promise<WebSearchToolInstance> {
  const config = resolveAgentCoreConfig(options);
  const client = buildAgentCoreClient(config);

  const { tool } = (await import('@langchain/core/tools')) as {
    tool: (
      fn: (input: { query: string }) => Promise<string>,
      meta: { name: string; description: string; schema: unknown }
    ) => WebSearchToolInstance;
  };
  const { z } = (await import('zod')) as { z: typeof import('zod').z };

  return tool(
    async ({ query }) =>
      flattenSearchResults({
        results: await client.search(query, config.maxResults),
      }),
    {
      name: 'agentcore_web_search',
      description:
        'Amazon Bedrock AgentCore web search. Use to answer questions ' +
        'about current events.',
      schema: z.object({
        query: z.string().max(2000).describe('The search query string.'),
      }),
    }
  );
}

/**
 * Factory namespace for web-search tools — pick a provider via
 * ``WebSearchTool.tavily(...)``, ``WebSearchTool.brave(...)``, or
 * ``WebSearchTool.agentcore(...)``.
 *
 * Each method returns a LangChain.js ``StructuredToolInterface`` ready to
 * pass into the engine's ``tools: []`` array. The interface form is used
 * (instead of the abstract ``StructuredTool`` class) so callers can drop
 * in any compatible tool implementation without hitting generic-parameter
 * mismatches.
 */
export const WebSearchTool = {
  tavily(options: TavilyOptions = {}): Promise<WebSearchToolInstance> {
    return buildTavily(options);
  },
  brave(options: BraveOptions = {}): Promise<WebSearchToolInstance> {
    return buildBrave(options);
  },
  agentcore(options: AgentCoreOptions = {}): Promise<WebSearchToolInstance> {
    return buildAgentCore(options);
  },
} as const;

export type WebSearchTool = typeof WebSearchTool;

/** Direct factory for the Tavily variant — preferred when ``tools: [...]``
 * needs a synchronous reference. Returns a promise that resolves to a tool. */
export function createWebSearchTool(
  provider: 'tavily',
  options?: TavilyOptions
): Promise<WebSearchToolInstance>;
export function createWebSearchTool(
  provider: 'brave',
  options?: BraveOptions
): Promise<WebSearchToolInstance>;
export function createWebSearchTool(
  provider: 'agentcore',
  options?: AgentCoreOptions
): Promise<WebSearchToolInstance>;
export function createWebSearchTool(
  provider: 'tavily' | 'brave' | 'agentcore',
  options: TavilyOptions | BraveOptions | AgentCoreOptions = {}
): Promise<WebSearchToolInstance> {
  if (provider === 'tavily') return buildTavily(options);
  if (provider === 'brave') return buildBrave(options);
  if (provider === 'agentcore') return buildAgentCore(options);
  throw new CompresrError(
    `Unknown web search provider: '${String(provider)}'. ` +
      "Use 'tavily', 'brave', or 'agentcore'.",
    'invalid_provider'
  );
}
