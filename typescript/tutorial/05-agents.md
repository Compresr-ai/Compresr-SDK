# Compresr agents (TypeScript)

The `@compresr/sdk/agents` subpath ships provider-shaped facades over LangChain.js so you can swap the Anthropic / OpenAI SDK for `CompressionClient` without changing your call sites — tool output is auto-compressed by Compresr's tool middleware.

**Customer scenario in this guide:** a venture analyst writing an investment memo on Anthropic. The agent runs live web searches and looks up internal investment-memo policies — both tools return multi-kilobyte payloads that would otherwise blow up the model context. Plug `CompressionClient` in and the tool output gets ~50% smaller before reaching the LLM with zero change at the call site.

## The mental model

One `CompressionClient`, three customer-facing surfaces, one engine. Every tool the LLM calls flows through `CompresrToolMiddleware`; outputs above `minTokens` are compressed via the Compresr backend before re-entering the model context.

| Surface | Mirrors |
|---|---|
| `client.messages.create(...)` | `new Anthropic().messages.create(...)` |
| `client.chat.completions.create(...)` | `new OpenAI().chat.completions.create(...)` |
| `client.run({ prompt, tools, ... })` | Compresr-native `NormalizedResult` |

## Install

```bash
npm install @compresr/sdk langchain @langchain/core
npm install @langchain/anthropic        # for anthropic:... models
npm install @langchain/openai           # for openai:... models
npm install @langchain/tavily           # for WebSearchTool.tavily
npm install @langchain/community        # for WebSearchTool.brave (optional)
```

## Anthropic-shaped facade

```typescript
import { CompressionClient, WebSearchTool, createWebSearchTool } from "@compresr/sdk";

// Two equivalent ways to build a web-search tool — pick whichever reads better:
const search = await WebSearchTool.tavily({                  // namespace factory
  apiKey: process.env.TAVILY_API_KEY!,
  maxResults: 5,
});

const searchAlt = await createWebSearchTool("tavily", {      // function form — useful when provider is dynamic
  apiKey: process.env.TAVILY_API_KEY!,
  maxResults: 5,
});

// Brave works the same way — bring the BRAVE_API_KEY and either:
//   await WebSearchTool.brave({ apiKey: process.env.BRAVE_API_KEY!, maxResults: 5 })
//   await createWebSearchTool("brave", { apiKey: process.env.BRAVE_API_KEY!, maxResults: 5 })

const client = new CompressionClient({
  apiKey: process.env.COMPRESR_API_KEY!,
  llm: "anthropic",                           // bare provider — model lives at the call site
  llmApiKey: process.env.ANTHROPIC_API_KEY!,
  compression: { targetCompressionRatio: 0.5, minTokens: 300 },
});

const msg = await client.messages.create({
  model: "claude-haiku-4-5",                  // model belongs here, like @anthropic-ai/sdk
  maxTokens: 512,
  messages: [{ role: "user", content: "What did Anthropic announce at their latest dev day? Cite the source." }],
  tools: [search],
});

console.log(msg.content[0]);                  // { type: "text", text: "...", citations: [...] }
console.log(msg.usage.input_tokens);
```

Use `llm: 'anthropic:claude-haiku-4-5'` if you want a default — but the call-site `model:` always wins.

## Custom tools with `@langchain/core/tools`

Any function you wrap with LangChain's `tool({...})` is automatically piped through `CompresrToolMiddleware`. You write zero compression code.

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const INVESTMENT_MEMOS: Record<string, string> = {
  anthropic: "ANTHROPIC INVESTMENT MEMO: ".repeat(200),         // ~5KB of internal due-diligence notes
  openai:    "OPENAI INVESTMENT MEMO: ".repeat(200),
};

const memoLookup = tool(
  async ({ company }: { company: string }) =>
    INVESTMENT_MEMOS[company] ?? `No memo on file for '${company}'.`,
  {
    name: "memo_lookup",
    description: "Look up the partnership's full investment memo. Companies: anthropic, openai.",
    schema: z.object({ company: z.string() }),
  },
);

const msg = await client.messages.create({
  model: "claude-haiku-4-5",
  maxTokens: 256,
  messages: [{ role: "user", content: "Summarize our existing thesis on Anthropic from the memo." }],
  tools: [memoLookup],
});
```

## OpenAI-shaped facade

Switching providers is one constructor argument; the tools above (Tavily + `kbLookup`) are unchanged.

```typescript
const openaiClient = new CompressionClient({
  apiKey: process.env.COMPRESR_API_KEY!,
  llm: "openai:gpt-4o-mini",
  llmApiKey: process.env.OPENAI_API_KEY!,
});

const completion = await openaiClient.chat.completions.create({
  model: "gpt-4o-mini",
  maxTokens: 256,
  messages: [{ role: "user", content: "Summarize our existing thesis on Anthropic from the memo." }],
  tools: [memoLookup],
});

console.log(completion.choices[0]?.message.content);
console.log(completion.usage.prompt_tokens, completion.usage.completion_tokens);
```

## Native shape

```typescript
const result = await client.run({ prompt: "Hello", tools: [search] });
console.log(result.text, result.citations, result.compresrStats);
```

## Verifying compression actually fires

Spy on `CompressionClient.compress` to confirm the middleware is calling the backend. Useful for the first integration to prove the value-add concretely.

```typescript
import { CompressionClient } from "@compresr/sdk";

const calls: { query?: string; contextLen: number }[] = [];
const original = client.compress.bind(client);
client.compress = async (opts) => {
  calls.push({ query: opts.query, contextLen: opts.context?.length ?? 0 });
  return original(opts);
};

await client.messages.create({
  model: "claude-haiku-4-5",
  maxTokens: 256,
  messages: [{ role: "user", content: "Summarize our existing thesis on OpenAI from the memo." }],
  tools: [memoLookup],
});

console.log(`compress() fired ${calls.length} time(s)`);
calls.forEach((c) => console.log(`  query=${c.query}, raw_context_len=${c.contextLen}`));
```

## Compression knobs

All compression behaviour lives in `compression: {...}` on the `CompressionClient` constructor.

| Key | Default | Effect |
|---|---|---|
| `targetCompressionRatio` | `0.5` | 0–1 fraction to remove. `0.7` is aggressive. Values `> 1` mean `Nx` compression. |
| `minTokens` | `200` | Outputs shorter than this skip compression. |
| `coarse` | server default `true` | `true` = paragraph-level (faster). `false` = token-level (finer). |
| `compressionModelName` | `"latte_v2"` | Which Compresr model. The backend validates. |
| `allowTools` | `undefined` | If set, only these tool names get compressed. |
| `ignoreTools` | `undefined` | These tool names always pass through uncompressed. |
| `onError` | `"passthrough"` | `"raise"` to fail loudly on backend errors instead of returning the original output. |

## Per-call LLM knobs

Pass standard chat-model parameters straight into any facade — they're forwarded to the underlying LangChain.js chat model via `.bind(...)` per call, so the engine's cached chat instance stays clean across requests.

```typescript
const msg = await client.messages.create({
  model: "claude-sonnet-4-6",
  maxTokens: 512,
  temperature: 0.2,
  topP: 0.9,
  messages: [{ role: "user", content: "Summarize this RFC..." }],
});
```

Supported keys (forwarded across `messages.create`, `chat.completions.create`, and native `run`):

| Key | Notes |
|---|---|
| `temperature`, `topP`, `topK` | Sampling. `topK` is mostly Anthropic / Gemini. |
| `maxTokens` | For Gemini, auto-aliased to `maxOutputTokens`. |
| `stopSequences`, `stop` | Anthropic uses `stopSequences`; OpenAI uses `stop`. |
| `presencePenalty`, `frequencyPenalty`, `seed` | OpenAI-style controls. |
| `logprobs`, `topLogprobs` | OpenAI-style controls. |

Unsupported keys (i.e. a key the provider's LangChain.js binding doesn't recognize) are silently ignored upstream — and any non-LLM keys never reach `.bind(...)` at all.

## When *not* to use this

If your tool outputs are short (<200 tokens), there's nothing to compress — the middleware skips them. Don't add the agents layer for tools that return one-liners; use `langchain` + `createAgent` directly.

Streaming (`messages.stream`, `chat.completions.stream`) is a Phase-2 work item and currently throws `CompresrError("streaming not yet implemented")`.
