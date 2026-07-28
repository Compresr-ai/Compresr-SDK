# Changelog

This monorepo ships two SDKs with independent versioning.

- Python — `compresr` on PyPI
- TypeScript — `@compresr/sdk` on npm

## Python — 2.9.1

- **fix(hermes): harden the fallback redactor.** When Hermes's `agent.redact` is
  unavailable, the built-in fallback now also masks PEM private-key blocks, Google
  API keys / OAuth tokens, and credentials embedded in connection-string URIs —
  closing the gaps a finite pattern list left open (Greptile P1).
- **fix(hermes): correct the recovery-cache wording.** The on-disk recovery cache
  stores a copy of the original with **secrets/PII masked** (not a raw verbatim
  copy). The recovery footer and docstrings now say so, matching the actual
  security behaviour (Greptile P1).
- **docs(python): fix the install matrix.** The README no longer claims the agents
  layer ships in the base install — `pip install compresr` is client-only; the
  provider chat models and web search tools require `compresr[agents]`. Aligns the
  README with `pyproject.toml` and the 2.9.0 CHANGELOG.

## Python — 2.9.0

- **feat(hermes): Hermes agent integration.** Query-aware **context + tool-output
  compression** for the Hermes agent, shipped as `compresr.integrations.hermes`
  (opt-in plugin via the `hermes_agent.plugins` entry point; inert without an API
  key). Cuts the token cost of large tool outputs and mid-conversation context;
  fail-open, with secret/PII redaction before anything leaves the process and a
  recovery cache (secrets/PII masked).
- **BREAKING — `langchain*` moved to the `agents` extra.** `pip install compresr`
  is now `httpx` + `pydantic` only. The agents layer (engine, Anthropic / OpenAI /
  Gemini providers, Tavily / Brave search) now requires
  **`pip install compresr[agents]`**. This keeps the base install from
  force-upgrading a host's pinned `openai` / `anthropic`. Back-compat extras
  (`compresr[langchain]`, `compresr[agents-all]`, …) resolve to `compresr[agents]`.
  If you use the agents/`CompressionClient(llm=...)` surfaces, add `[agents]` to
  your install.
- **BREAKING — minimum Python raised to 3.10** (was 3.9).
- **fix(deps): security.** Remediated fast-uri (CVE-2026-16221) and
  @hono/node-server (GHSA-frvp-7c67-39w9) in the TypeScript workspace; pinned
  ruff's rule set so an unpinned linter can't drift CI.

## Python — 2.8.3 / TypeScript — 1.7.1

- **chore(release): first tag-triggered CI publish.** Releases are now cut by
  pushing `python-vX.Y.Z` / `typescript-vX.Y.Z` tags on `main`; CI gates on
  tag-on-main + tag==version, runs unit tests, publishes to PyPI/npm, and
  creates the GitHub Release. Nothing is published from laptops anymore.
- **TypeScript: runtime `zod` bumped 3.x → 4.4.3** (Dependabot #39);
  `mapZodError` migrated to the zod-4 API (`ZodError.issues` — the v3
  `.errors` alias was removed — and path segments stringified before joining).
- TypeScript devDependencies: dotenv 17, @types/node 26,
  @typescript-eslint/parser 8.62.1, @vitest/coverage-v8 4.1.10; lockfile
  repaired after the Dependabot merge sequence.
- Python: no functional changes since 2.8.2; republished so the artifact is
  provably built from `main` by CI.

## Python — 2.8.0 / TypeScript — 1.7.0

- **feat(agents): add `WebSearchTool.agentcore` (Amazon Bedrock AgentCore web
  search) provider for Python (`compresr` 2.8.0) & TypeScript (`@compresr/sdk`
  1.7.0).** A third first-class web-search provider alongside `tavily` /
  `brave`, exposed as `WebSearchTool.agentcore(...)` (Python classmethod) and
  `WebSearchTool.agentcore({...})` (TS method), plus `provider="agentcore"`
  dispatch. The LLM-facing tool is named **`agentcore_web_search`** with a
  single required **`query`** input, and returns **plaintext** (shared
  `_flatten_search_results` / `flattenSearchResults`) so `latte_v1` and
  `CompresrToolMiddleware` compress it the same as the other providers.
- **Cognito OAuth + MCP under the hood.** Unlike Tavily/Brave there is no
  off-the-shelf LangChain tool, so the provider mints a Cognito
  client-credentials bearer token and opens an MCP streamable-HTTP session to
  the gateway, picking the web-search tool by normalized name and ignoring the
  semantic `x_amz_bedrock_agentcore_search` meta tool. The client secret and
  bearer token are never logged or echoed into error messages.
- **Optional dependency, not base.** A bare `pip install compresr` /
  `npm i @compresr/sdk` still imports fine; the dependency error only fires
  when `.agentcore()` is actually called.
  - Python: new `compresr[agentcore]` extra (`mcp>=1.0`, `nest-asyncio>=1.6`),
    also folded into the `all` extra. Token POST reuses the base `httpx` dep;
    `mcp` / `nest_asyncio` are imported lazily and raise a clear
    `ImportError` with the `pip install compresr[agentcore]` hint when absent.
  - TypeScript: `@modelcontextprotocol/sdk` added as an **optional peer
    dependency** (`peerDependenciesMeta.optional = true`); the token POST uses
    built-in `fetch`. A missing peer throws
    `CompresrError(..., 'missing_peer_dependency')`.
- **Config contract (both languages, identical).** Explicit arg → env-var
  precedence for the five required fields — `gatewayUrl`/`gateway_url`
  (`AGENTCORE_GATEWAY_MCP_URL` → `GATEWAY_MCP_URL`), `cognitoTokenUrl`
  (`AGENTCORE_COGNITO_TOKEN_URL` → `COGNITO_TOKEN_URL`), `clientId`
  (`AGENTCORE_COGNITO_CLIENT_ID` → `COGNITO_CLIENT_ID`), `clientSecret`
  (`AGENTCORE_COGNITO_CLIENT_SECRET` → `COGNITO_CLIENT_SECRET`), `scope`
  (`AGENTCORE_COGNITO_SCOPE` → `COGNITO_SCOPE`) — plus `maxResults`/`max_results`
  (arg only, default 5, clamped 1–25). Missing values raise a clear error
  naming each missing key and its fallbacks (`ValueError` Python /
  `CompresrError(..., 'missing_config')` TS). `allowedDomains`/`blockedDomains`
  are accepted for signature parity but warn that AgentCore has no native
  domain filter (use Tavily).
- TS: `src/version.ts` (`SDK_VERSION`) bumped `1.6.8` → `1.7.0` to match
  `package.json` (it feeds the SDK User-Agent header).

## Python — 2.7.9

- **New: latte_v2 adaptive (dynamic) compression.** Three optional kwargs added
  to every compression method (`compress`, `compress_async`, `compress_stream`,
  `compress_batch`, `compress_batch_async`) and to `CompressRequest` /
  `CompressBatchRequest`:
  - `dynamic: bool | None` — when True, the server picks the keep-count per
    document from the score elbow (Kneedle) instead of a fixed ratio. When
    enabled, `target_compression_ratio` is ignored.
  - `dynamic_min_ratio: float | None` — floor on adaptive compression (server
    default 1.5x). Caps how *little* the doc is compressed.
  - `dynamic_max_ratio: float | None` — ceiling on adaptive compression
    (server default 10x). Caps how *aggressively* the doc is compressed.
  All three default to `None` so unset values never reach the wire (`model_dump
  (exclude_none=True)`). Backend is the authority — it returns 422 if the
  fields are sent to a model that doesn't support them (currently latte_v2
  only). No client-side validation, matching the existing SDK philosophy.
- **`MODELS.LATTE_V1` and `MODELS.LATTE_V2`** convenience constants added
  alongside the existing `MODELS.LATTE` (= `"latte_v1"`).
- 5 new unit tests in `tests/unit/test_schemas.py::TestLatteV2DynamicFields`
  covering defaults, explicit values, wire-format exclusion, batch-request
  parity, and a regression assertion that internal/debug knobs
  (`aggregation`, `include_tokens`) stay off the SDK surface.

## TypeScript — 1.6.8

- **Parity with Python 2.7.9: latte_v2 adaptive (dynamic) compression.**
  Three optional fields on `CompressOptions` / `CompressBatchOptions`
  (camelCase) → mapped to snake_case on the wire by the Zod schemas:
  - `dynamic: boolean` — Kneedle elbow selection; overrides
    `targetCompressionRatio` when true.
  - `dynamicMinRatio: number` — floor on adaptive compression (server
    default 1.5x).
  - `dynamicMaxRatio: number` — ceiling on adaptive compression (server
    default 10x).
  All three default to undefined; the wire stays clean (Zod `.optional()`).
  Backend is the authority — 422 on a model that doesn't support them.
- **`MODELS.LATTE_V1` and `MODELS.LATTE_V2`** constants added alongside
  the existing `MODELS.LATTE` (= `"latte_v1"`).
- Version sync: `src/version.ts` was lagging at `1.6.5` — bumped to
  `1.6.8` to match `package.json`.
- Tests: 5 new schema cases (defaults, explicit values, snake_case wire
  mapping for both single + batch, internal-knob exclusion).

## Python — 2.7.8

- **Docs / tutorials overhaul.** Tutorials under `python/tutorial/` now use
  realistic customer scenarios (stitched multi-document corpora) instead of
  toy examples, with a GitHub-style diff visualization for compressed output.
  TS↔Python parity for the same tutorial set so customers can compare SDKs
  side-by-side. No API changes — wheel contents are identical to 2.7.7 apart
  from tutorial assets and the version string.
- **CI.** Black / isort pass on `python/tutorial/_demo_utils.py`; lockfile-
  adjacent fixes on the TS side don't affect Python install.

## TypeScript — 1.6.7

- **Docs / tutorials overhaul.** Mirror of the Python 2.7.8 tutorial work —
  realistic customer scenarios under `typescript/tutorial/`, GitHub-style
  diff rendering, TS↔Python parity.
- **CI / build.** `package-lock.json` regenerated so `npm ci` syncs cleanly
  on Node 20 / 22 in CI (nested `p-retry@6.2.1` for `@llamaindex/workflow`,
  top-level `@emnapi/core` / `@emnapi/runtime`). Removed an unused
  `@ts-expect-error` in `src/agents/tools/web-search.ts` that became
  redundant once eslint dropped an unnecessary type assertion. No public
  API changes.

## Python — 2.7.7

- **New: user-configurable transport-layer retry.** Defaults retry `429` and
  `503` from the proxy up to 3 times with exponential backoff + jitter,
  honoring the `Retry-After` header / body field. Surfaces as
  `CompressionClient(..., retry_config=RetryConfig(...))`. Pass
  `RetryConfig(max_retries=0)` to opt out. Without this the sync transport
  raised `ServiceUnavailableError` on the first 503 — when the on-prem proxy
  hit its in-flight cap during a burst, customers saw 40-60% failure rates
  instead of brief queueing. Applies to both sync (`urllib`) and async
  (`httpx`) paths. Includes 17 unit tests covering defaults, success-after-
  retry, no-retry-on-4xx, exhaustion, opt-out, and `Retry-After` honoring.

## Python — 2.7.6

- Version-only republish on top of 2.7.5 — issued after the GitHub branch
  cleanup landed (closed Dependabot PRs #12-#19 and feat/* PRs #21, #22 once
  their content was confirmed integrated into `oussama`). **No source change
  vs 2.7.5**; the wheel contents are byte-identical except for the version
  string. Use 2.7.6 to ensure a clean install pinned to the post-cleanup
  release point.

## Python — 2.7.5

- **New: custom HTTP client for the LLM provider transport.**
  `CompressionClient(..., llm_http_client=..., llm_http_async_client=...)` lets
  callers pass their own `httpx.Client` / `httpx.AsyncClient` for the downstream
  Anthropic / OpenAI calls — needed behind TLS-intercepting corporate proxies
  and for custom CA bundles, mTLS, or `verify=False`. The clients are baked
  into every chat-model build; a per-call `run(http_client=...)` overrides
  them. The `llm_` prefix is deliberate: these target the LLM provider
  transport, not the Compresr API transport. OpenAI accepts these natively;
  for Anthropic the SDK shims support on versions of `langchain-anthropic`
  that predate the native `http_client` field (see langchain issue #36056) by
  swapping the cached `anthropic.Anthropic` client for one built around the
  supplied httpx client, reusing langchain's own client params. The shim is a
  no-op once the upstream field ships. `google_genai` is not wired (different
  transport). Includes unit tests (passthrough / override / shim) and a
  hermetic TLS-intercepting-proxy reproducer under `python/tests/manual/`.
  Cleanup tracking doc at `python/HTTP_CLIENT_FIX_ONCE_PR_APPROVED.md`.
  Patch-level bump (2.7.4 → 2.7.5) because the new kwargs are additive and
  the default behaviour is unchanged.
- **Internal**: comment cleanup across `compresr/` source. Removed ~120 lines
  of WHAT-narration and decorative section headers; preserved all docstrings
  (API contract) and WHY-comments (workarounds, invariants, external issue
  refs). No public API or behaviour change.
- **Tooling**: `python/deploy_python_sdk.sh` (gitignored, per-maintainer
  script) now whitelists tests skipped due to Gemini quota exhaustion
  (`RESOURCE_EXHAUSTED` / `quota` / test path contains `gemini`) so a depleted
  Google AI Studio prepayment doesn't block the deploy. Any other skip still
  aborts. **Manual sync required** for other maintainers — the script is not
  version-controlled.

## Python — 2.7.3

- **Packaging fix (CRITICAL).** Switch `python/pyproject.toml` from a hand-maintained `[tool.setuptools]` `packages = [...]` allowlist to `[tool.setuptools.packages.find]` with `include = ["compresr*"]`. The old allowlist silently dropped any subpackage that wasn't added to it by hand — `compresr.agents.research` (added in 2.7.0) was never in the list, so **the 2.7.0, 2.7.1, and 2.7.2 PyPI wheels shipped without the research-agent module**. `client.research.run(...)` and `client.research.search(...)` raise `ModuleNotFoundError` on a fresh `pip install` of any of those versions, even though source-based / editable installs work fine. **`2.7.2` has been yanked from PyPI; use `2.7.3` instead.** The same applies retroactively to `2.7.0` and `2.7.1` — those releases were also broken but the bug was not noticed until Kamel's audit on 2026-06-08. Editable installs (`pip install -e .`) ignore the packages list, which is why this regression survived several local test runs.

## Python — 2.7.2 — YANKED (use 2.7.3)

> Yanked on PyPI 2026-06-08: this wheel is missing `compresr.agents.research` due to the setuptools packaging-allowlist regression. See the 2.7.3 entry above. The source code on this tag is otherwise correct; the bug is in `pyproject.toml` only.

## Python — 2.7.2

- **New: `ResearchUsage.search_calls`.** `client.research.run(...)` /
  `client.research.search(...)` now reports the number of times the search
  tool fired during the research loop, alongside the existing token + LLM-call
  counts. Populated by counting `Step.type == "search"` entries in the
  trajectory during usage aggregation. Independent of `usage.calls` (which
  counts LLM round-trips), because one LLM turn can request multiple tool
  calls. The SDK intentionally stays counts-only — no USD pricing — because
  neither Anthropic/OpenAI nor Brave/Tavily return dollar cost on their
  responses; cost rollups must happen at the call site against a
  caller-owned price table.

## Python — 2.7.1

- **New: solo-WebSearchTool auto-route.** When the generic-agent facades (`client.messages.create`, `client.chat.completions.create`, `client.run`) are invoked with exactly the Compresr `WebSearchTool` and nothing else (`tools=[WebSearchTool.tavily(...)]` or `tools=[WebSearchTool.brave(...)]`), the engine silently routes the call through the strict-output research loop instead of LangChain's `create_agent` middleware path. The caller still receives their facade's expected shape (Anthropic `Message` / OpenAI `ChatCompletion` / `NormalizedResult`); only the loop semantics switch — `tool_choice="none"` on the final step (forces the model to commit), per-snippet `latte_v1` compression against the live tool-call query, 10-step + 120k-token caps, and manual Anthropic `cache_control` stamping. Caller's `system=` wins when supplied; otherwise the research-agent's `DEFAULT_RESEARCH_SYSTEM_PROMPT` is used. Multi-tool calls keep the existing path. Detection is by tool name (`tavily_search` / `brave_search`) since `WebSearchTool.__new__` returns a `StructuredTool`.
- Internal: `ResearchAgent.run` factored into `_run_loop(question, *, model, extra_chat_kwargs)` + the public parser — the loop is now reusable from the engine's fast-path without duplicating the ReAct logic.

## Python — 2.7.0

- **New: `client.research.run(question)` / `client.research.search(question)`** — first-class web-research agent with per-snippet `latte_v1` compression on tool results and provider-aware prompt caching. Multi-step ReAct loop; forces commit on the final step (`tool_choice="none"`); strict `Explanation / Exact Answer / Confidence / Citations` output schema parsed into a typed `ResearchResult`. Loop structure adapted from Perplexity `search_evals` (MIT). See README for an end-to-end example.
- **New: multi-provider prompt-cache surface.** `CompressionClient(..., enable_prompt_cache=True, prompt_cache_ttl="5m"|"1h", openai_prompt_cache_key=...)` now drives Anthropic middleware, OpenAI `prompt_cache_key` routing + `prompt_cache_retention="24h"` mapping (for `ttl="1h"`), and Gemini `cached_content_token_count` aggregation. The same three kwargs work across providers; OpenAI / Gemini surfaces degrade gracefully to no-op when not applicable.
- **Bugfix**: `_aggregate_usage` previously double-counted `cache_read_input_tokens` for `langchain-anthropic` 1.x (the adapter inflates `input_tokens` to `fresh + cache_read + cache_creation`). The aggregator now subtracts cache tokens from `input_tokens` so the field means fresh-only downstream.
- **Bugfix**: when `cache_control` carries a TTL, `langchain-anthropic` stores the write count in `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` and zeroes `cache_creation`. The aggregator now sums both TTL buckets.
- New public types: `ResearchResult`, `ResearchUsage`, `Citation`, `Step`, `StepKind`, `ResearchAgent`, `ResearchFacade`, `parse_research_output`, `DEFAULT_RESEARCH_SYSTEM_PROMPT`.
- Pin: `langchain-anthropic>=1.0` (needed for the prompt-cache middleware module).

## Python — 2.6.5

- **Bugfix**: Streaming requests now honour the same timeout as non-stream calls. Previously the SDK shipped a `STREAM_TIMEOUT` constant in `compresr.config` that was never referenced by any transport — the streaming code path already used `self._timeout` (5 min default), so the dead constant was confusing but had no runtime effect. Removed it for clarity. `CompressionClient(api_key=..., timeout=<seconds>)` now is the single way to control timeout across sync POST, sync stream, and async POST.

## TypeScript — 1.6.6

- **New: user-configurable transport-layer retry.** Mirrors the Python
  `2.7.7` release. Defaults retry HTTP `429` and `503` from the proxy up to
  3 times with exponential backoff + jitter, honoring `Retry-After` (header
  or `retry_after` body field). Surfaces as
  `new CompressionClient({ retry: { maxRetries, initialBackoffMs, ... } })`.
  Pass `{ maxRetries: 0 }` to opt out.
- **Bugfix**: raw `503` responses with no `code` field now map to
  `ServiceUnavailableError` (matching Python's contract) instead of the
  generic `ServerError`. This unblocks the retry classifier for on-prem
  proxies that surface backpressure without a structured error code.
- 15 unit tests added (`tests/unit/http-retry.test.ts`) mirroring the
  Python suite.

## TypeScript — 1.6.5

- Version-parity bump with Python `2.7.5`. The Python `llm_http_client` /
  `llm_http_async_client` feature has no TypeScript counterpart in this
  release — `@compresr/sdk` already accepts a per-client `httpClient` via the
  underlying LangChain factories, so no surface change was needed.
- **Internal**: comment cleanup across `src/`. Removed ~100 lines of
  WHAT-narration and decorative section headers; preserved all JSDoc on
  exported APIs and WHY-comments. Also realigned `src/version.ts` from the
  drifted `1.5.4` back to `1.6.5` so `package.json` and the runtime export
  agree again.

## TypeScript — 1.6.4

- **Expose the research agent from the public surface.** `1.6.2` / `1.6.3` shipped the research module under `dist/` but neither re-exported it from `./index.js` (`@compresr/sdk`) nor declared a subpath in `package.json` `exports`, so `import { ResearchAgent } from '@compresr/sdk'` resolved to `undefined` and `import { ResearchAgent } from '@compresr/sdk/agents/research'` raised `ERR_PACKAGE_PATH_NOT_EXPORTED`. The `client.research.run(...)` runtime worked, but consumers couldn't type the result or import the standalone agent. Fix: re-export `ResearchAgent`, `ResearchFacade`, `parseResearchOutput`, `DEFAULT_RESEARCH_SYSTEM_PROMPT`, and the related types (`ResearchResult`, `ResearchUsage`, `Step`, `StepKind`, `ResearchAgentOptions`, `ResearchAgentRunOptions`, `ResearchRunOptions`, `ParsedResearch`) from `@compresr/sdk` and `@compresr/sdk/agents`; declare a `./agents/research` subpath. The research-specific `Citation` is aliased to `ResearchCitation` on the public surface to avoid clashing with the normalized-result `Citation` (different shape).

## TypeScript — 1.6.3

- Version-parity bump with Python `2.7.3`. **No code changes vs `1.6.2`** — the TypeScript SDK does not ship via npm using a setuptools-style allowlist, so it was unaffected by the Python packaging regression. The bump exists only to keep release pairs aligned (`compresr X.Y.Z` ↔ `@compresr/sdk A.B.C` with synchronised z's).

## TypeScript — 1.6.2

- **New: `ResearchUsage.search_calls`.** Mirror of the Python 2.7.2 change.
  `client.research.run({...})` / `client.research.search({...})` now reports
  the number of times the search tool fired during the research loop on the
  returned `ResearchUsage` (`search_calls: number`). Counted from
  `state.trajectory.filter(s => s.type === 'search')`. Same rationale as
  Python: the SDK stays counts-only; callers price Brave/Tavily/Anthropic
  surcharges externally against their own price table.

## TypeScript — 1.6.1

- **New: solo-WebSearchTool auto-route.** Mirror of the Python 2.7.1 change. When `client.messages.create({ tools: [WebSearchTool.tavily(...)] })` (or `chat.completions.create` / `client.run`) is called with exactly the Compresr `WebSearchTool` and nothing else, the engine routes through the deep-research loop instead of LangChain's `createAgent` path. Same facade shapes preserved (Anthropic `Message` / OpenAI `ChatCompletion` / `NormalizedResult`); same loop semantics (`tool_choice="none"` on the final step, per-snippet `latte_v1` compression against the live tool-call query, 10-step + 120k-token caps, manual Anthropic `cache_control` stamping). Caller's `system` wins when supplied; otherwise the research-agent default prompt is used. Multi-tool calls keep the standard path. New exports: `isSoloWebSearch`, `extractLastUserText`, `WEB_SEARCH_TOOL_NAMES` from `@compresr/sdk/agents`.
- Internal: `ResearchAgent.run` factored into `_runLoop(question, options)` + the public parser, so the engine's fast-path can reuse the loop without duplicating it.
- Exports types `ChatModelLike` / `SearchToolLike` / `ResearchEngine` so external callers can build structural mocks of the engine for testing.

## TypeScript — 1.6.0

- **New: `await client.research.run(question)` / `await client.research.search(question)`** — mirror of the Python research agent. Same loop, same compression, same output shape. `client.research.run(question, { search: "tavily", maxSteps: 10 })` returns a typed `ResearchResult`.
- **New: multi-provider prompt-cache surface.** `new CompressionClient({ enablePromptCache, promptCacheTtl, openaiPromptCacheKey, ... })` drives Anthropic middleware (already shipped in 1.5.x), OpenAI `prompt_cache_key` routing + retention mapping, and Gemini `cached_content_token_count` aggregation. Surface parity with the Python SDK.
- **Bugfix**: aggregator now correctly subtracts cache tokens from `input_tokens` for `@langchain/anthropic` 1.x, and reads `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` for cache writes. Matches the Python fix.
- **Bugfix**: Brave env-var lookup now tries `BRAVE_SEARCH_API_KEY` first then falls back to `BRAVE_API_KEY` (matches the underlying `WebSearchTool.brave` order). Same fix in Python.
- New public types: `ResearchResult`, `ResearchUsage`, `Citation`, `Step`, `StepKind`, `ResearchAgent`, `ResearchFacade`, `parseResearchOutput`, `DEFAULT_RESEARCH_SYSTEM_PROMPT`.

## TypeScript — 1.5.4

- **Bugfix**: `CompressionClient.compressStream(...)` now honours the per-client `timeout` option. It used to ignore the constructor's `timeout` and abort against the standalone `STREAM_TIMEOUT` constant — so passing `new CompressionClient({ apiKey, timeout: 600_000 })` correctly extended `post()` but silently kept `stream()` at the 5-minute default. Both endpoints now read `this.timeout`, with the same 5-minute default and constructor override.
- Regression test (`http-timeout.test.ts`) spies on `setTimeout` to assert both `post()` and `stream()` schedule abort at the same per-client delay.

## Python — 2.6.2

- **Bugfix**: `WebSearchTool.brave(...)` was usable for constructing a tool but broke at runtime — `BraveSearch._run() missing 1 required positional argument: 'query'`. Root cause: `langchain_community.tools.BraveSearch` ships with `args_schema=None`, so the LLM's tool_use payload never contained a `query` field. The wrapper now declares an explicit Pydantic schema (`{query: str}`) before handing the tool back to the customer. Regression test asserts the schema is present.

## Python — 2.6.1

- **Bugfix**: `max_tokens`, `temperature`, and other per-call LLM kwargs were silently dropped **whenever any tool was passed** to `messages.create` / `chat.completions.create` / `run`. Root cause: LangChain's `bind_tools(...)` (called internally by `create_agent`) strips a prior `chat.bind(...)`'s kwargs. The engine now bakes per-call knobs into the `init_chat_model(...)` constructor instead. Cache is keyed by `(model_name, sorted_kwargs)` so distinct combos reuse correctly. Wire-verified via a new integration test (`test_anthropic_max_tokens_caps_output_WITH_TOOLS`).
- `chat.bind(...)` is no longer called anywhere in the engine.

## Python — 2.6.0

- **Provider/model split**: `CompressionClient(llm="anthropic")` is enough — the model lives at the call site (`messages.create(model="claude-haiku-4-5")`), matching Anthropic's and OpenAI's own SDKs. `llm="anthropic:claude-haiku-4-5"` still works as a default; the call-site `model=` always wins. Chat models are cached per model name.
- **Agents layer in the base install**: `pip install compresr` now ships the engine, all three provider chat models (Anthropic / OpenAI / Gemini), and both web search tools (Tavily + Brave). Old `[agents-*]` brackets kept as no-op aliases for back-compat.
- **`max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `presence_penalty`, `frequency_penalty`, `seed`, `logprobs`, `top_logprobs`** now flow through to the underlying chat model via per-call `.bind(...)` — confirmed at the wire (Anthropic + OpenAI integration tests cap `output_tokens` at the value passed; Gemini auto-aliases `max_tokens` → `max_output_tokens`).
- **`WebSearchTool.tavily(...)` / `WebSearchTool.brave(...)`** classmethod factories alongside the original `WebSearchTool(provider="...")` constructor.
- 251 unit tests + 12 live integration tests.

## Python — 2.5.2

- **New**: agents layer — `CompressionClient.messages.create` (Anthropic shape), `client.chat.completions.create` (OpenAI shape), `client.run` (native `NormalizedResult`).
- **New**: `WebSearchTool.tavily(...)` and `WebSearchTool.brave(...)` — LangChain `BaseTool` factories. Output flows through `CompresrToolMiddleware` automatically for compression.
- **New**: optional extras — `agents`, `agents-anthropic`, `agents-openai`, `agents-gemini`, `agents-tavily`, `agents-brave`, `agents-all`.
- Built on LangChain 1.0 `init_chat_model` + `create_agent`. Compression of tool outputs is automatic via `CompresrToolMiddleware`; the kernel `ToolOutputCompressor` lives in `integrations/_shared/`.
- 225 unit tests + 8 integration tests against live Anthropic, OpenAI, and Tavily.

## TypeScript — 1.5.1

- **Bugfix**: same as Python 2.6.1 — `maxTokens` / `temperature` / `topP` / etc. were silently dropped when any tool was bound. `CompresrEngine.getChatModel(modelName, extraKwargs?)` now bakes the LLM kwargs into `initChatModel(...)` and caches per `(modelName, stableStringify(kwargs))`. No more post-hoc `chat.bind()` call.

## TypeScript — 1.5.0

- **Provider/model split**: `new CompressionClient({ llm: "anthropic" })` is enough — `messages.create({ model: "claude-haiku-4-5" })` carries the model. Constructor's `llm: "anthropic:claude-haiku-4-5"` still resolves as a default. Chat models cached per name.
- **`maxTokens`, `temperature`, `topP`, `topK`, `stopSequences`, `presencePenalty`, `frequencyPenalty`, `seed`, `logprobs`, `topLogprobs`** all forward to the chat model via per-call `.bind(...)`. Gemini auto-aliases `maxTokens` → `maxOutputTokens`.
- `SDK_VERSION` synced to `1.5.0` (User-Agent header matches `package.json`).
- 199 unit tests pass; ESM + CJS build green; lint exits 0 with 0 errors/0 warnings.

## TypeScript — 1.4.0

- **New**: agents layer mirroring Python. `client.messages.create`, `client.chat.completions.create`, `client.run`.
- **New**: `WebSearchTool.tavily({...})` / `WebSearchTool.brave({...})` namespace factories + `createWebSearchTool(provider, opts)` function form.
- **New**: subpath export `@compresr/sdk/agents`. Optional peer deps: `@langchain/anthropic`, `@langchain/openai`, `@langchain/google-genai`, `@langchain/tavily`, `@langchain/community`.
- `parseLlmSpec` accepts both colon (`"anthropic:claude-..."`) and slash (`"anthropic/claude-..."`) forms.
- Streaming methods (`messages.stream`, `chat.completions.stream`) throw `"streaming not yet implemented"` — Phase 2.
- 178 unit tests.
