/**
 * Compresr agents — provider-shape facades over LangChain.js.
 *
 * Public surface is intentionally minimal. Heavy LangChain imports stay
 * dynamic so ``import "@compresr/sdk/agents"`` stays cheap when the user
 * only needs the type names.
 *
 *     npm install langchain @langchain/core     # engine peers
 *     npm install @langchain/anthropic          # for anthropic:...
 *     npm install @langchain/openai             # for openai:...
 *     npm install @langchain/google-genai       # for google_genai:...
 *     npm install @langchain/tavily             # for WebSearchTool.tavily
 *     npm install @langchain/community          # for WebSearchTool.brave
 *     npm install @modelcontextprotocol/sdk     # for WebSearchTool.agentcore
 */
export {
  CompresrEngine,
  parseLlmSpec,
  type CompresrEngineOptions,
  type CompressionPolicyOptions,
  type RunOptions,
} from './engine.js';

export {
  WebSearchTool,
  createWebSearchTool,
  type TavilyOptions,
  type BraveOptions,
  type AgentCoreOptions,
} from './tools/index.js';

export {
  type Citation,
  type CompresrStats,
  type NormalizedResult,
  type NormalizedToolUse,
  defaultCompresrStats,
  emptyNormalizedResult,
} from './normalized.js';

export {
  anthropicMessages,
  type AnthropicMessagesFacade,
  type AnthropicMessagesCreateOptions,
} from './facades/anthropic.js';

export {
  openaiChatCompletions,
  type OpenAIChatFacade,
  type OpenAICompletionsFacade,
  type OpenAICompletionsCreateOptions,
} from './facades/openai.js';

export { nativeRun, type NativeRunOptions } from './facades/native.js';

export {
  type AnthropicMessage,
  type ContentBlock,
  type TextBlock,
  type ToolUseBlock,
  type AnthropicUsage,
  toAnthropicMessage,
} from './schemas/anthropic.js';

export {
  type ChatCompletion,
  type ChatMessage,
  type Choice,
  type FunctionCall,
  type ToolCall,
  type OpenAIUsage,
  toChatCompletion,
} from './schemas/openai.js';

// `Citation` is aliased to `ResearchCitation` to avoid clashing with the
// normalized-result `Citation` exported above (different shape).
export {
  ResearchAgent,
  ResearchFacade,
  parseResearchOutput,
  DEFAULT_RESEARCH_SYSTEM_PROMPT,
  type ResearchAgentOptions,
  type ResearchAgentRunOptions,
  type ResearchRunOptions,
  type ParsedResearch,
  type ResearchResult,
  type ResearchUsage,
  type Step,
  type StepKind,
  type Citation as ResearchCitation,
} from './research/index.js';
