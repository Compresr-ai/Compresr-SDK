/**
 * AgentCore web-search client — Cognito OAuth + MCP streamable-HTTP.
 *
 * Ported from the reference ``websearch_client.py`` (``AgentCoreWebSearch``).
 * Reaching Amazon Bedrock AgentCore web search requires two hops that the
 * off-the-shelf Tavily/Brave tools handle for us but AgentCore does not:
 *
 *   1. A Cognito **client-credentials** OAuth grant to mint a bearer token.
 *   2. An **MCP streamable-HTTP** session to the gateway, calling the
 *      ``...WebSearch`` tool with ``{query, maxResults}`` and parsing the
 *      single JSON text block it returns.
 *
 * ``@modelcontextprotocol/sdk`` is an **optional** peer: a bare install must
 * import fine; the ``missing_peer_dependency`` error only fires when
 * ``.agentcore()`` is actually invoked. Never log the client secret or the
 * bearer token.
 *
 * Mirrors Python ``compresr/agents/tools/_agentcore.py``.
 */
import { CompresrError } from '../../errors/index.js';

/**
 * The gateway also exposes this built-in *tool*-search (semantic search over
 * the gateway's own tools). It is NOT web search — always ignore it.
 */
const TOOL_SEARCH_META = 'x_amz_bedrock_agentcore_search';

/** Cognito token POST timeout (ms). */
const TOKEN_TIMEOUT_MS = 30_000;

/** MCP tool-call timeout (ms) — bounds how long a hung gateway can stall a search. */
const TOOL_CALL_TIMEOUT_MS = 30_000;

/**
 * Upper bound (bytes, UTF-8) on the MCP tool-result text before it is
 * buffered and JSON.parse()'d. Guards against an unbounded/misbehaving
 * gateway response causing unbounded memory allocation. Mirrors Python's
 * ``MAX_RESPONSE_BYTES``.
 */
const MAX_RESPONSE_BYTES = 1_000_000;

/** Resolved AgentCore gateway configuration (camelCase). */
export interface AgentCoreConfig {
  gatewayUrl: string;
  cognitoTokenUrl: string;
  clientId: string;
  clientSecret: string;
  scope: string;
  /** Default result count; clamped 1–25 by the caller. */
  maxResults: number;
}

/** One web-search hit. ``content`` is AWS's extracted snippet, not full text. */
export interface AgentCoreSearchResult {
  title: string | null;
  url: string | null;
  content: string;
  publishedDate: string | null;
}

/** Minimal client surface — ``search`` mints/caches the token as needed. */
export interface AgentCoreClient {
  search(query: string, maxResults: number): Promise<AgentCoreSearchResult[]>;
}

// --- minimal structural types for the optional MCP SDK ---------------------
// The SDK is an optional peer, so we narrow to only the surface we use rather
// than depending on its exported types at compile time.

interface McpTransport {
  close(): Promise<void>;
}

interface McpTransportCtor {
  new (
    url: URL,
    opts?: {
      requestInit?: { headers?: Record<string, string>; signal?: AbortSignal };
    }
  ): McpTransport;
}

interface McpToolResult {
  content: unknown;
  /** Per the MCP spec, a failed tool call still returns HTTP 200 with
   * ``isError: true`` and the error message in the text block. */
  isError?: boolean;
}

interface McpClient {
  connect(transport: McpTransport): Promise<void>;
  listTools(): Promise<{ tools: Array<{ name: string }> }>;
  callTool(params: {
    name: string;
    arguments: Record<string, unknown>;
  }): Promise<McpToolResult>;
  close(): Promise<void>;
}

interface McpClientCtor {
  new (info: { name: string; version: string }): McpClient;
}

interface LoadedMcp {
  Client: McpClientCtor;
  StreamableHTTPClientTransport: McpTransportCtor;
}

/**
 * Dynamically import the optional ``@modelcontextprotocol/sdk``. The exact
 * client subpaths were verified against the installed package (v1.29);
 * ``./client/index.js`` and ``./client/streamableHttp.js`` resolve via the
 * package's ``./*`` export wildcard.
 */
async function loadMcp(): Promise<LoadedMcp> {
  let clientMod: Record<string, unknown>;
  let transportMod: Record<string, unknown>;
  try {
    clientMod = await import('@modelcontextprotocol/sdk/client/index.js');
    transportMod = await import(
      '@modelcontextprotocol/sdk/client/streamableHttp.js'
    );
  } catch {
    throw new CompresrError(
      "WebSearchTool.agentcore requires '@modelcontextprotocol/sdk'. " +
        'Install with: npm install @modelcontextprotocol/sdk',
      'missing_peer_dependency'
    );
  }
  const Client = clientMod['Client'] as McpClientCtor | undefined;
  const StreamableHTTPClientTransport = transportMod[
    'StreamableHTTPClientTransport'
  ] as McpTransportCtor | undefined;
  if (
    typeof Client !== 'function' ||
    typeof StreamableHTTPClientTransport !== 'function'
  ) {
    throw new CompresrError(
      "'@modelcontextprotocol/sdk' is missing expected client exports — " +
        'upgrade the package.',
      'missing_peer_dependency'
    );
  }
  return { Client, StreamableHTTPClientTransport };
}

/** Normalize a tool name for fuzzy matching (drop case, dashes, underscores). */
function normalizeName(name: string): string {
  return name.toLowerCase().replace(/[-_]/g, '');
}

/** Pick the web-search tool, ignoring the semantic tool-search meta tool. */
function pickWebSearchTool(names: ReadonlyArray<string>): string {
  const candidates = names.filter((n) => n !== TOOL_SEARCH_META);
  for (const name of candidates) {
    if (normalizeName(name).includes('websearch')) return name;
  }
  if (candidates.length === 0) {
    throw new CompresrError(
      `No web-search tool found among [${names.join(', ')}].`,
      'agentcore_no_tool'
    );
  }
  return candidates[0];
}

/** Extract the ``access_token`` string from an untrusted token response. */
function extractAccessToken(data: unknown): string | null {
  if (data && typeof data === 'object' && 'access_token' in data) {
    const token = (data as { access_token?: unknown }).access_token;
    if (typeof token === 'string' && token.length > 0) return token;
  }
  return null;
}

/** Concatenate the ``text`` of all text content blocks. */
function extractText(content: unknown): string {
  if (!Array.isArray(content)) return '';
  let out = '';
  for (const block of content) {
    if (block && typeof block === 'object' && 'text' in block) {
      const text = (block as { text?: unknown }).text;
      if (typeof text === 'string') out += text;
    }
  }
  return out;
}

/** Parse the gateway's JSON text block into normalized result rows. */
function parseResults(text: string): AgentCoreSearchResult[] {
  if (!text) return [];
  if (Buffer.byteLength(text, 'utf8') > MAX_RESPONSE_BYTES) {
    throw new CompresrError(
      'AgentCore web search response exceeded the size guard.',
      'agentcore_bad_response'
    );
  }
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new CompresrError(
      'AgentCore web search returned a non-JSON payload.',
      'agentcore_bad_response'
    );
  }
  if (!payload || typeof payload !== 'object' || !('results' in payload)) {
    return [];
  }
  const results = (payload as { results?: unknown }).results;
  if (!Array.isArray(results)) return [];
  const out: AgentCoreSearchResult[] = [];
  for (const item of results) {
    if (!item || typeof item !== 'object') continue;
    const row = item as Record<string, unknown>;
    out.push({
      title: typeof row['title'] === 'string' ? row['title'] : null,
      url: typeof row['url'] === 'string' ? row['url'] : null,
      content: typeof row['text'] === 'string' ? row['text'] : '',
      publishedDate:
        typeof row['publishedDate'] === 'string' ? row['publishedDate'] : null,
    });
  }
  return out;
}

/** Best-effort detection of an HTTP 401 / unauthorized failure. */
function isUnauthorized(err: unknown): boolean {
  if (err && typeof err === 'object') {
    const code = (err as { code?: unknown }).code;
    if (code === 401) return true;
    const message = (err as { message?: unknown }).message;
    if (typeof message === 'string') {
      const lowered = message.toLowerCase();
      return lowered.includes('401') || lowered.includes('unauthorized');
    }
  }
  return false;
}

/**
 * Build an {@link AgentCoreClient}. The bearer token is cached across calls and
 * transparently re-minted once on an MCP 401. Secrets and the token itself are
 * never logged or surfaced in error messages.
 */
export function buildAgentCoreClient(config: AgentCoreConfig): AgentCoreClient {
  let cachedToken: string | null = null;
  // Cache the in-flight mint promise (not just the resolved token) so
  // concurrent first-use callers await the same Cognito request instead of
  // each firing their own (redundant credential-bearing requests / self
  // -inflicted rate limiting).
  let inFlightMint: Promise<string> | null = null;

  async function mintTokenUncached(): Promise<string> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TOKEN_TIMEOUT_MS);
    let resp: Response;
    try {
      resp = await fetch(config.cognitoTokenUrl, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          grant_type: 'client_credentials',
          client_id: config.clientId,
          client_secret: config.clientSecret,
          scope: config.scope,
        }),
        signal: controller.signal,
      });
    } catch (err: unknown) {
      // Do NOT rethrow the original error — it (and any .cause it carries)
      // may reference the request, including the client_secret in its body.
      // Covers both network-level failures (DNS/TLS/connection reset) and
      // the abort fired by the timeout above.
      const timedOut = err instanceof Error && err.name === 'AbortError';
      throw new CompresrError(
        timedOut
          ? 'AgentCore Cognito token request timed out.'
          : 'AgentCore Cognito token request failed (network error).',
        'agentcore_auth_error'
      );
    } finally {
      clearTimeout(timer);
    }
    if (!resp.ok) {
      // Do NOT include the response body — it may echo request credentials.
      throw new CompresrError(
        `AgentCore Cognito token request failed (HTTP ${resp.status}).`,
        'agentcore_auth_error'
      );
    }
    const data: unknown = await resp.json();
    const token = extractAccessToken(data);
    if (!token) {
      throw new CompresrError(
        'AgentCore Cognito response did not contain an access_token.',
        'agentcore_auth_error'
      );
    }
    cachedToken = token;
    return token;
  }

  async function mintToken(): Promise<string> {
    if (inFlightMint) return inFlightMint;
    const promise = mintTokenUncached().finally(() => {
      inFlightMint = null;
    });
    inFlightMint = promise;
    return promise;
  }

  async function getToken(refresh: boolean): Promise<string> {
    if (cachedToken && !refresh) return cachedToken;
    return mintToken();
  }

  async function callOnce(
    mcp: LoadedMcp,
    query: string,
    maxResults: number,
    refresh: boolean
  ): Promise<AgentCoreSearchResult[]> {
    const token = await getToken(refresh);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TOOL_CALL_TIMEOUT_MS);
    const transport = new mcp.StreamableHTTPClientTransport(
      new URL(config.gatewayUrl),
      {
        requestInit: {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        },
      }
    );
    const client = new mcp.Client({
      name: 'compresr-agentcore-client',
      version: '1.0.0',
    });
    try {
      await client.connect(transport);
      const listed = await client.listTools();
      const toolName = pickWebSearchTool(listed.tools.map((t) => t.name));
      const result = await client.callTool({
        name: toolName,
        arguments: { query, maxResults },
      });
      const text = extractText(result.content);
      if (result.isError) {
        // Gateway tool-level errors (bad query, throttling, auth/scope
        // error) come back as plain text, NOT JSON — surface the real
        // diagnostic text (and keep it intact for isUnauthorized() below).
        throw new CompresrError(
          `AgentCore web search tool returned an error: ${text}`,
          'agentcore_tool_error'
        );
      }
      return parseResults(text);
    } finally {
      clearTimeout(timer);
      await client.close().catch(() => undefined);
    }
  }

  async function search(
    query: string,
    maxResults: number
  ): Promise<AgentCoreSearchResult[]> {
    const mcp = await loadMcp();
    try {
      return await callOnce(mcp, query, maxResults, false);
    } catch (err: unknown) {
      if (isUnauthorized(err)) {
        // Token may have expired — re-mint once and retry.
        return callOnce(mcp, query, maxResults, true);
      }
      throw err;
    }
  }

  return { search };
}
