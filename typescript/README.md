# Compresr TypeScript SDK

Query-aware LLM context compression — reduce LLM API costs by 30-70%.

## Install

```bash
npm install @compresr/sdk
```

Get an API key at [compresr.ai](https://compresr.ai).

## Quick start

```typescript
import { CompressionClient } from '@compresr/sdk';

const client = new CompressionClient({ apiKey: 'cmp_...' });

const result = await client.compress({
  context: 'Long passage to compress...',
  query: 'What is the main conclusion?',
  targetCompressionRatio: 0.5,
});

console.log(`Saved ${result.data?.tokens_saved} tokens`);
console.log(result.data?.compressed_context);
```

The default model is `latte_v2` (query-aware). Pass any other model name your
account has access to via `compressionModelName: '...'` — the backend
validates.

## Adaptive (dynamic) compression — `latte_v2` only

`latte_v2` can pick the keep-ratio per document instead of holding a fixed
target. Useful for RAG / tool-output flows where chunk density varies a lot:
dense docs keep more, sparse docs compress hard.

```typescript
// Server adapts per-doc, capped between 1.5x and 10x by default
const result = await client.compress({
  context: '...',
  query: '...',
  compressionModelName: 'latte_v2',
  dynamic: true,
});

// Tighter band — never less than 2x, never more than 6x
const tuned = await client.compress({
  context: '...',
  query: '...',
  compressionModelName: 'latte_v2',
  dynamic: true,
  dynamicMinRatio: 2.0,
  dynamicMaxRatio: 6.0,
});
```

When `dynamic: true`, `targetCompressionRatio` is ignored. Sending
`dynamic: true` to a model that doesn't support it (e.g. `latte_v1`) returns
a 422 from the API.

## Batch

```typescript
const batch = await client.compressBatch({
  contexts: ['Doc 1...', 'Doc 2...', 'Doc 3...'],
  queries: 'What is self-attention?',
  targetCompressionRatio: 0.5,
});

console.log(`Total saved: ${batch.data?.total_tokens_saved} tokens`);
```

## Streaming

```typescript
for await (const chunk of client.compressStream({
  context: '...',
  query: '...',
})) {
  process.stdout.write(chunk.content);
}
```

## Compression options

| Param | Purpose |
|---|---|
| `query` | Question the LLM is trying to answer — drives `latte_v2` compression |
| `targetCompressionRatio` | `0-1` strength or `>1` for Nx factor (max 200) |
| `coarse` | `true` = paragraph-level (default, faster); `false` = token-level (fine-grained) |
| `heuristicChunking` | Structure-preserving chunking |
| `disablePlaceholders` | Disable placeholder tokens in output |

## Error handling

```typescript
import {
  AuthenticationError,
  RateLimitError,
  ValidationError,
  CompresrError,
} from '@compresr/sdk';

try {
  await client.compress({ context, query });
} catch (err) {
  if (err instanceof AuthenticationError) console.error('Invalid API key');
  else if (err instanceof RateLimitError) console.error('Rate limited');
  else if (err instanceof ValidationError) console.error('Bad request:', err);
  else if (err instanceof CompresrError) console.error('API error:', err);
}
```

## Drop-in agent client

`@compresr/sdk/agents` ships a tiny provider-shape facade layer on top of
LangChain.js. One `CompressionClient` exposes three call surfaces — Anthropic
`messages.create`, OpenAI `chat.completions.create`, and a native `run` — all
backed by the same engine (`createAgent` + `CompresrToolMiddleware`). Every
tool output above `minTokens` is auto-compressed before it re-enters the
model's context.

- Pass `temperature`, `topP`, `maxTokens`, `stopSequences`, `presencePenalty`,
  `frequencyPenalty`, `seed`, etc. to any facade — they're forwarded to the
  underlying chat model per call via `.bind(...)` (no cache pollution).

```typescript
import { CompressionClient } from '@compresr/sdk';
import { WebSearchTool } from '@compresr/sdk/agents';

const client = new CompressionClient({
  apiKey: process.env.COMPRESR_API_KEY!,
  llm: 'anthropic',                           // bare provider — model lives at the call site
  llmApiKey: process.env.ANTHROPIC_API_KEY!,
  compression: { targetCompressionRatio: 0.5, minTokens: 300 },
});
```

Use `llm: 'anthropic:claude-haiku-4-5'` if you want a default — but the call-site `model:` always wins.

### Anthropic shape

```typescript
const msg = await client.messages.create({
  model: 'claude-haiku-4-5',
  maxTokens: 512,
  messages: [{ role: 'user', content: "What's the latest AI news?" }],
  tools: [search],
});
```

### OpenAI shape

```typescript
const completion = await client.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: "Summarize today's top story." }],
  tools: [search],
});
```

### Native shape

```typescript
const result = await client.run({ prompt: 'Hello', tools: [search], maxTokens: 512 });
console.log(result.text, result.citations, result.compresrStats);
```

### Tools

`WebSearchTool` returns a LangChain.js `BaseTool` — usable across all three
facades.

```typescript
import { WebSearchTool } from '@compresr/sdk/agents';

const tavily = await WebSearchTool.tavily({
  apiKey: process.env.TAVILY_API_KEY!,
  maxResults: 5,
  allowedDomains: ['arxiv.org'],
});

const brave = await WebSearchTool.brave({
  apiKey: process.env.BRAVE_API_KEY!,
  maxResults: 5,
});
```

#### Amazon Bedrock AgentCore

`WebSearchTool.agentcore` reaches AgentCore web search through a Cognito
client-credentials OAuth grant + an MCP streamable-HTTP session. Install the
optional MCP peer first:

```bash
npm install @modelcontextprotocol/sdk        # for WebSearchTool.agentcore
```

Provide the gateway config explicitly, or via env vars (precedence is
**explicit option → env var**):

| Option            | Env var (legacy fallback)                          |
| ----------------- | -------------------------------------------------- |
| `gatewayUrl`      | `AGENTCORE_GATEWAY_MCP_URL` (`GATEWAY_MCP_URL`)     |
| `cognitoTokenUrl` | `AGENTCORE_COGNITO_TOKEN_URL` (`COGNITO_TOKEN_URL`) |
| `clientId`        | `AGENTCORE_COGNITO_CLIENT_ID` (`COGNITO_CLIENT_ID`) |
| `clientSecret`    | `AGENTCORE_COGNITO_CLIENT_SECRET` (`COGNITO_CLIENT_SECRET`) |
| `scope`           | `AGENTCORE_COGNITO_SCOPE` (`COGNITO_SCOPE`)         |

```typescript
// All five fields can come from the env vars above instead.
const agentcore = await WebSearchTool.agentcore({
  gatewayUrl: process.env.AGENTCORE_GATEWAY_MCP_URL!,
  cognitoTokenUrl: process.env.AGENTCORE_COGNITO_TOKEN_URL!,
  clientId: process.env.AGENTCORE_COGNITO_CLIENT_ID!,
  clientSecret: process.env.AGENTCORE_COGNITO_CLIENT_SECRET!,
  scope: process.env.AGENTCORE_COGNITO_SCOPE!,
  maxResults: 5, // clamped to 1–25
});
```

The tool is named `agentcore_web_search` and returns plaintext (compressible by
the middleware). AgentCore has **no native domain filter**, so `allowedDomains`
/ `blockedDomains` are accepted for parity but ignored — use Tavily if you need
domain filtering. The client secret and bearer token are never logged.

`WebSearchTool` is also re-exported from `@compresr/sdk` for convenience, but
`@compresr/sdk/agents` is the recommended import path — heavy LangChain peers
stay dynamic, so the subpath only pulls them in when you touch the surface.

### Custom tools

Any LangChain.js tool works. Use the standard `tool({...})` decorator and
Compresr compresses its output the same way:

```typescript
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

const fetchDocs = tool(
  async ({ query }) => fetchInternalDocs(query),
  {
    name: 'fetch_docs',
    description: 'Search the internal docs corpus',
    schema: z.object({ query: z.string() }),
  },
);

await client.run({ prompt: 'How does billing work?', tools: [fetchDocs] });
```

### Provider switching

Swap providers by changing one line — the facades stay the same:

```typescript
new CompressionClient({ apiKey, llm: 'anthropic:claude-haiku-4-5', llmApiKey });
new CompressionClient({ apiKey, llm: 'openai:gpt-4o-mini',         llmApiKey });
new CompressionClient({ apiKey, llm: 'google_genai:gemini-2.5-flash', llmApiKey });
```

Streaming (`messages.stream`, `chat.completions.stream`) is a Phase-2 item
and currently throws `CompresrError('streaming not yet implemented')`.

### Why no provider-native server search?

Anthropic `web_search_20250305`, OpenAI `web_search_preview`, and Gemini
`google_search` execute server-side and return encrypted or opaque content
that Compresr can't read — so it can't compress them either. Use
`WebSearchTool.tavily` / `WebSearchTool.brave` if you want compressible
search output in the agent loop.

## Research agent

`client.research.run(question)` runs a multi-step web-research loop with
per-snippet `latte_v1` compression on tool results and multi-provider prompt
caching. Loop structure adapted from Perplexity `search_evals` (MIT).

```typescript
import { CompressionClient } from "@compresr/sdk";

const client = new CompressionClient({
  apiKey: "cmp_...",
  llm: "anthropic:claude-sonnet-4-6",
  llmApiKey: "sk-ant-...",
});

const result = await client.research.run(
  "What was the latest stable Python version released in 2025?",
  { search: "tavily", maxSteps: 10 }     // "tavily" | "brave" | a LangChain tool
);

console.log(result.answer);              // parsed Exact Answer field
console.log(result.explanation);         // parsed Explanation field
console.log(result.confidence);          // parsed 0-1 confidence
console.log(result.citations);           // ReadonlyArray<{ url: string }>
console.log(result.usage.cache_read_tokens, result.usage.calls);
```

Single-shot mode: `await client.research.search(question)` runs one search +
a forced final answer (equivalent to `run(..., { maxSteps: 2 })`).

The agent respects all the prompt-cache options on `CompressionClient`
(`enablePromptCache`, `promptCacheTtl`, `openaiPromptCacheKey`). Tavily /
Brave keys are read from `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY` (falls
back to `BRAVE_API_KEY`).

## Framework integrations

Optional peer dependencies — install only what you use:

```bash
npm install langchain @langchain/core          # engine for the agents layer
npm install @langchain/anthropic               # for anthropic:... models
npm install @langchain/openai                  # for openai:... models
npm install @langchain/google-genai            # for google_genai:... models
npm install @langchain/tavily                  # for WebSearchTool.tavily
npm install @langchain/community               # for WebSearchTool.brave
npm install @modelcontextprotocol/sdk          # for WebSearchTool.agentcore
npm install @langchain/langgraph               # for LangGraph integration
npm install llamaindex                         # for LlamaIndex integration
```

### LangChain — middleware, tool wrapper, retriever

```typescript
import { createAgent } from 'langchain';
import {
  compresrToolMiddleware,
  wrapToolWithCompression,
  CompresrExtractor,
} from '@compresr/sdk/integrations/langchain';

const agent = createAgent({
  model,
  tools: [webSearch],
  middleware: [
    compresrToolMiddleware({
      apiKey: process.env.COMPRESR_API_KEY!,
      queryArg: 'query',
    }),
  ],
});
```

### LangGraph — compression as a graph node

```typescript
import { makeCompresrNode } from '@compresr/sdk/integrations/langgraph';

graph.addNode(
  'compress',
  makeCompresrNode<State>({
    apiKey: process.env.COMPRESR_API_KEY!,
    contextKey: 'retrieved_text',
    queryKey: 'user_question',
  })
);
```

### LlamaIndex — node postprocessor for RAG

```typescript
import { CompresrNodePostprocessor } from '@compresr/sdk/integrations/llamaindex';

const queryEngine = index.asQueryEngine({
  nodePostprocessors: [
    new CompresrNodePostprocessor({
      apiKey: process.env.COMPRESR_API_KEY!,
    }),
  ],
});
```

### Unified query API

Every integration that accepts a query exposes the same three knobs:

| Param | Purpose |
|---|---|
| `query` | Static query — same for every call |
| `queryExtractor` | Callable that derives the query from the call context |
| `queryArg` / `queryKey` | Name of the tool arg / state key to use as the query |

Priority: `query` > `queryExtractor` > `queryArg`/`queryKey` > smart-pick
from common arg keys (`query`, `question`, `search_query`, ...) > last user
message in history.

### Tutorials

Runnable end-to-end examples under `tutorial/`:

- `01-quickstart.ts` — core `CompressionClient`.
- `02-langchain.ts` — middleware + tool wrapper + retriever.
- `03-langgraph.ts` — compression node in a 3-node graph.
- `04-llamaindex.ts` — node postprocessor + tool wrapper.
- `05-agents.md` — drop-in agent client (Anthropic / OpenAI / native facades).

Run any with `npx tsx --env-file=../.env tutorial/01-quickstart.ts`.

## Requirements

- Node.js 20+ (uses native `fetch`)
- TypeScript 5.0+ (optional, recommended)
- Optional peers: `@langchain/core>=0.3`, `langchain>=1.0`,
  `@langchain/anthropic>=1.0`, `@langchain/openai>=1.0`,
  `@langchain/google-genai>=0.2`, `@langchain/tavily>=0.1`,
  `@langchain/community>=0.3`, `@modelcontextprotocol/sdk>=1.0`,
  `@langchain/langgraph>=0.2`, `llamaindex>=0.8` (install only what you use)

## Development

```bash
npm install
npm test                 # unit tests
npm run test:integration # live tests (requires COMPRESR_API_KEY)
npm run build
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Support

- Docs: [compresr.ai/docs](https://compresr.ai/docs)
- Issues: [GitHub](https://github.com/Compresr-ai/Compresr-SDK/issues)
- Email: support@compresr.ai
