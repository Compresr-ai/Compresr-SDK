/**
 * Unit tests for ``WebSearchTool.agentcore`` — the Amazon Bedrock AgentCore
 * provider. The Cognito token ``fetch`` and the ``@modelcontextprotocol/sdk``
 * client are both stubbed; there are NO live AWS / network calls.
 *
 * Each scenario re-imports the SUT after ``vi.resetModules()`` + ``vi.doMock``
 * so the dynamic ``import('@modelcontextprotocol/sdk/...')`` inside the client
 * resolves our fake. Because of the module reset, ``CompresrError`` thrown by
 * the fresh SUT is a different class identity than a top-level import, so we
 * assert on ``code`` / ``name`` rather than ``instanceof``.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

interface McpScenario {
  failImport?: boolean;
  emptyExports?: boolean;
  toolNames?: ReadonlyArray<string>;
  toolText?: string;
  unauthorizeFirstConnect?: boolean;
  toolIsError?: boolean;
}

interface McpCallLog {
  connectCount: number;
  authHeaders: string[];
  lastCall: { name: string; arguments: Record<string, unknown> } | null;
}

const VALID_OPTIONS = {
  gatewayUrl: 'https://gw.example.com/mcp',
  cognitoTokenUrl: 'https://auth.example.com/oauth2/token',
  clientId: 'client-id',
  clientSecret: 'super-secret',
  scope: 'gateway/invoke',
};

const DEFAULT_PAYLOAD = JSON.stringify({
  results: [
    {
      title: 'AgentCore overview',
      url: 'https://example.com/a',
      text: 'Amazon Bedrock AgentCore provides managed web search.',
      publishedDate: '2025-01-01',
    },
    {
      title: 'MCP streamable HTTP',
      url: 'https://example.com/b',
      text: 'The gateway speaks the MCP streamable-HTTP transport.',
      publishedDate: null,
    },
  ],
});

let tokenFetchCount = 0;

function stubTokenFetch(): void {
  tokenFetchCount = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      tokenFetchCount += 1;
      return new Response(
        JSON.stringify({ access_token: `tok-${tokenFetchCount}` }),
        { status: 200, headers: { 'content-type': 'application/json' } }
      );
    })
  );
}

async function setupAgentCore(
  scenario: McpScenario = {}
): Promise<{
  mod: typeof import('../../src/agents/tools/web-search.js');
  calls: McpCallLog;
}> {
  vi.resetModules();
  const calls: McpCallLog = { connectCount: 0, authHeaders: [], lastCall: null };
  const toolNames = scenario.toolNames ?? [
    'gateway___WebSearch',
    'x_amz_bedrock_agentcore_search',
  ];
  const toolText = scenario.toolText ?? DEFAULT_PAYLOAD;

  vi.doMock('@modelcontextprotocol/sdk/client/index.js', () => {
    if (scenario.failImport) throw new Error('Cannot find module');
    if (scenario.emptyExports) return { Client: undefined };
    class FakeClient {
      constructor(_info: { name: string; version: string }) {}
      async connect(): Promise<void> {
        calls.connectCount += 1;
        if (scenario.unauthorizeFirstConnect && calls.connectCount === 1) {
          throw Object.assign(new Error('HTTP 401 Unauthorized'), {
            code: 401,
          });
        }
      }
      async listTools(): Promise<{ tools: Array<{ name: string }> }> {
        return { tools: toolNames.map((name) => ({ name })) };
      }
      async callTool(params: {
        name: string;
        arguments: Record<string, unknown>;
      }): Promise<{ content: unknown; isError?: boolean }> {
        calls.lastCall = params;
        return {
          content: [{ type: 'text', text: toolText }],
          isError: scenario.toolIsError ?? false,
        };
      }
      async close(): Promise<void> {}
    }
    return { Client: FakeClient };
  });

  vi.doMock('@modelcontextprotocol/sdk/client/streamableHttp.js', () => {
    if (scenario.failImport) throw new Error('Cannot find module');
    if (scenario.emptyExports) return { StreamableHTTPClientTransport: undefined };
    class FakeTransport {
      constructor(
        _url: URL,
        opts?: { requestInit?: { headers?: Record<string, string> } }
      ) {
        calls.authHeaders.push(opts?.requestInit?.headers?.['Authorization'] ?? '');
      }
      async close(): Promise<void> {}
    }
    return { StreamableHTTPClientTransport: FakeTransport };
  });

  const mod = await import('../../src/agents/tools/web-search.js');
  return { mod, calls };
}

const AGENTCORE_ENV_VARS = [
  'AGENTCORE_GATEWAY_MCP_URL',
  'GATEWAY_MCP_URL',
  'AGENTCORE_COGNITO_TOKEN_URL',
  'COGNITO_TOKEN_URL',
  'AGENTCORE_COGNITO_CLIENT_ID',
  'COGNITO_CLIENT_ID',
  'AGENTCORE_COGNITO_CLIENT_SECRET',
  'COGNITO_CLIENT_SECRET',
  'AGENTCORE_COGNITO_SCOPE',
  'COGNITO_SCOPE',
] as const;

function clearAgentCoreEnv(): Record<string, string | undefined> {
  const saved: Record<string, string | undefined> = {};
  for (const key of AGENTCORE_ENV_VARS) {
    saved[key] = process.env[key];
    delete process.env[key];
  }
  return saved;
}

function restoreEnv(saved: Record<string, string | undefined>): void {
  for (const [key, value] of Object.entries(saved)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe('WebSearchTool.agentcore — config resolution', () => {
  it('throws missing_config naming the absent keys when nothing is provided', async () => {
    const saved = clearAgentCoreEnv();
    try {
      const { mod } = await setupAgentCore();
      await expect(mod.WebSearchTool.agentcore({})).rejects.toMatchObject({
        name: 'CompresrError',
        code: 'missing_config',
      });
      // The message should name each missing field + its env-var fallbacks.
      await mod.WebSearchTool.agentcore({}).catch((err: unknown) => {
        const message = err instanceof Error ? err.message : '';
        expect(message).toContain('gatewayUrl');
        expect(message).toContain('AGENTCORE_GATEWAY_MCP_URL');
        expect(message).toContain('clientSecret');
      });
    } finally {
      restoreEnv(saved);
    }
  });

  it('builds the tool from explicit options', async () => {
    const saved = clearAgentCoreEnv();
    try {
      const { mod } = await setupAgentCore();
      const tool = await mod.WebSearchTool.agentcore({ ...VALID_OPTIONS });
      expect(tool.name).toBe('agentcore_web_search');
      expect(tool.schema).toBeDefined();
    } finally {
      restoreEnv(saved);
    }
  });

  it('resolves config from environment variables (incl. legacy fallbacks)', async () => {
    const saved = clearAgentCoreEnv();
    try {
      process.env['AGENTCORE_GATEWAY_MCP_URL'] = VALID_OPTIONS.gatewayUrl;
      process.env['AGENTCORE_COGNITO_TOKEN_URL'] = VALID_OPTIONS.cognitoTokenUrl;
      process.env['AGENTCORE_COGNITO_CLIENT_ID'] = VALID_OPTIONS.clientId;
      // Legacy fallback names for the remaining two.
      process.env['COGNITO_CLIENT_SECRET'] = VALID_OPTIONS.clientSecret;
      process.env['COGNITO_SCOPE'] = VALID_OPTIONS.scope;
      stubTokenFetch();
      const { mod, calls } = await setupAgentCore();
      const tool = await mod.WebSearchTool.agentcore({});
      expect(tool.name).toBe('agentcore_web_search');
      const out = (await tool.invoke({ query: 'q' })) as string;
      expect(out).toContain('AgentCore overview');
      expect(calls.lastCall?.arguments['query']).toBe('q');
    } finally {
      restoreEnv(saved);
    }
  });
});

describe('WebSearchTool.agentcore — search behavior', () => {
  it('returns plaintext blocks (not JSON) and ignores the meta tool', async () => {
    const saved = clearAgentCoreEnv();
    try {
      stubTokenFetch();
      const { mod, calls } = await setupAgentCore();
      const tool = await mod.WebSearchTool.agentcore({ ...VALID_OPTIONS });
      const out = (await tool.invoke({ query: 'agentcore' })) as string;
      expect(typeof out).toBe('string');
      expect(out.trimStart().startsWith('{')).toBe(false);
      expect(out).toContain('AgentCore overview');
      expect(out).toContain('https://example.com/a');
      expect(out).toContain('Amazon Bedrock AgentCore provides managed web search.');
      expect(out).toContain('MCP streamable HTTP');
      expect(out).toContain('\n\n');
      // The semantic tool-search meta tool must never be picked.
      expect(calls.lastCall?.name).toBe('gateway___WebSearch');
    } finally {
      restoreEnv(saved);
    }
  });

  it('never leaks the client secret or bearer token into output', async () => {
    const saved = clearAgentCoreEnv();
    try {
      stubTokenFetch();
      const { mod, calls } = await setupAgentCore();
      const tool = await mod.WebSearchTool.agentcore({ ...VALID_OPTIONS });
      const out = (await tool.invoke({ query: 'q' })) as string;
      expect(out).not.toContain('super-secret');
      expect(out).not.toContain('tok-1');
      // The bearer token reached the transport header, not the output.
      expect(calls.authHeaders[0]).toBe('Bearer tok-1');
    } finally {
      restoreEnv(saved);
    }
  });

  it('clamps maxResults into the 1–25 range', async () => {
    const saved = clearAgentCoreEnv();
    try {
      stubTokenFetch();
      const high = await setupAgentCore();
      const toolHigh = await high.mod.WebSearchTool.agentcore({
        ...VALID_OPTIONS,
        maxResults: 100,
      });
      await toolHigh.invoke({ query: 'q' });
      expect(high.calls.lastCall?.arguments['maxResults']).toBe(25);

      stubTokenFetch();
      const low = await setupAgentCore();
      const toolLow = await low.mod.WebSearchTool.agentcore({
        ...VALID_OPTIONS,
        maxResults: 0,
      });
      await toolLow.invoke({ query: 'q' });
      expect(low.calls.lastCall?.arguments['maxResults']).toBe(1);
    } finally {
      restoreEnv(saved);
    }
  });

  it('surfaces the real diagnostic text when the gateway returns isError:true', async () => {
    const saved = clearAgentCoreEnv();
    try {
      stubTokenFetch();
      const { mod } = await setupAgentCore({
        toolIsError: true,
        toolText: 'invalid maxResults: must be between 1 and 25',
      });
      const tool = await mod.WebSearchTool.agentcore({ ...VALID_OPTIONS });
      await expect(tool.invoke({ query: 'q' })).rejects.toMatchObject({
        name: 'CompresrError',
        code: 'agentcore_tool_error',
      });
      await tool.invoke({ query: 'q' }).catch((err: unknown) => {
        const message = err instanceof Error ? err.message : '';
        expect(message).toContain('invalid maxResults');
      });
    } finally {
      restoreEnv(saved);
    }
  });

  it('rejects http:// gateway/token URLs to avoid leaking credentials', async () => {
    const saved = clearAgentCoreEnv();
    try {
      const { mod } = await setupAgentCore();
      await expect(
        mod.WebSearchTool.agentcore({
          ...VALID_OPTIONS,
          gatewayUrl: 'http://gw.example.com/mcp',
        })
      ).rejects.toMatchObject({
        name: 'CompresrError',
        code: 'invalid_config',
      });
    } finally {
      restoreEnv(saved);
    }
  });

  it('re-mints the token once and retries on an MCP 401', async () => {
    const saved = clearAgentCoreEnv();
    try {
      stubTokenFetch();
      const { mod, calls } = await setupAgentCore({
        unauthorizeFirstConnect: true,
      });
      const tool = await mod.WebSearchTool.agentcore({ ...VALID_OPTIONS });
      const out = (await tool.invoke({ query: 'q' })) as string;
      expect(out).toContain('AgentCore overview');
      // Two connects: the 401 attempt + the retry with a fresh token.
      expect(calls.connectCount).toBe(2);
      expect(calls.authHeaders[0]).toBe('Bearer tok-1');
      expect(calls.authHeaders[1]).toBe('Bearer tok-2');
    } finally {
      restoreEnv(saved);
    }
  });
});

describe('WebSearchTool.agentcore — missing peer dependency', () => {
  it('throws missing_peer_dependency when the MCP SDK import fails', async () => {
    const saved = clearAgentCoreEnv();
    try {
      stubTokenFetch();
      const { mod } = await setupAgentCore({ failImport: true });
      const tool = await mod.WebSearchTool.agentcore({ ...VALID_OPTIONS });
      await expect(tool.invoke({ query: 'q' })).rejects.toMatchObject({
        name: 'CompresrError',
        code: 'missing_peer_dependency',
      });
    } finally {
      restoreEnv(saved);
    }
  });

  it('throws missing_peer_dependency when client exports are absent', async () => {
    const saved = clearAgentCoreEnv();
    try {
      stubTokenFetch();
      const { mod } = await setupAgentCore({ emptyExports: true });
      const tool = await mod.WebSearchTool.agentcore({ ...VALID_OPTIONS });
      await expect(tool.invoke({ query: 'q' })).rejects.toMatchObject({
        code: 'missing_peer_dependency',
      });
    } finally {
      restoreEnv(saved);
    }
  });
});

describe('createWebSearchTool — agentcore dispatch', () => {
  it('dispatches to the agentcore provider', async () => {
    const saved = clearAgentCoreEnv();
    try {
      const { mod } = await setupAgentCore();
      const tool = await mod.createWebSearchTool('agentcore', {
        ...VALID_OPTIONS,
      });
      expect(tool.name).toBe('agentcore_web_search');
    } finally {
      restoreEnv(saved);
    }
  });

  it('lists agentcore in the invalid_provider message', async () => {
    const { mod } = await setupAgentCore();
    expect(() =>
      (
        mod.createWebSearchTool as unknown as (
          p: string,
          o?: unknown
        ) => Promise<unknown>
      )('nope', {})
    ).toThrow(/agentcore/);
  });
});
