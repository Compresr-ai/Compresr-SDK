/**
 * 05 — Compresr agents: VC analyst research workflow.
 *
 * Customer scenario: a venture analyst writing an investment memo on
 * Anthropic. The agent runs a live web search (Tavily) and then fetches a
 * long primary-source bundle (~30k tokens stitched from 4 related
 * Wikipedia articles) and produces a memo-ready briefing. We benchmark the
 * page-fetch workflow off-vs-on Compresr on two providers (Anthropic and
 * OpenAI), print measured latency / tokens / cost, and render a colored CLI
 * diff showing exactly which words `latte_v1` dropped from the tool output.
 *
 * Run from the SDK root:
 *   npx tsx --env-file=.env typescript/tutorial/05-agents.ts
 *
 * Once published: import { CompressionClient, WebSearchTool } from '@compresr/sdk';
 */
import { tool } from '@langchain/core/tools';
import { initChatModel } from 'langchain/chat_models/universal';
import { createAgent } from 'langchain';
import { z } from 'zod';

import { CompressionClient, WebSearchTool } from '../src/index.js';
import { fetchWikipedia, printCompresrDiff } from './_demo_utils.js';

const COMPRESR_API_KEY = process.env.COMPRESR_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const TAVILY_API_KEY = process.env.TAVILY_API_KEY;
if (!COMPRESR_API_KEY) throw new Error('Set COMPRESR_API_KEY.');
if (!ANTHROPIC_API_KEY) throw new Error('Set ANTHROPIC_API_KEY.');
if (!OPENAI_API_KEY) throw new Error('Set OPENAI_API_KEY.');
if (!TAVILY_API_KEY) throw new Error('Set TAVILY_API_KEY.');

const RESEARCH_TITLES = ['Anthropic', 'Claude (language model)', 'Dario Amodei', 'Constitutional AI'];

const researchCorpus = tool(
  async () => {
    const parts: string[] = [];
    for (const title of RESEARCH_TITLES) {
      const text = await fetchWikipedia(title);
      if (text) parts.push(`# ${title}\n\n${text}`);
    }
    return parts.join('\n\n');
  },
  {
    name: 'research_corpus',
    description:
      'Fetch a primary-source research corpus on Anthropic. Returns a long stitched bundle of related Wikipedia articles (~30k tokens). Use this when asked to research a company for an investment memo.',
    schema: z.object({ topic: z.string() }),
  },
);

const PROMPT =
  'You are a venture analyst writing an investment memo on Anthropic. ' +
  'Use the research_corpus tool (pass topic="Anthropic") to fetch primary-source material. ' +
  'Based ONLY on the fetched corpus, produce a memo-ready briefing with three sections: ' +
  '(1) Funding history — every round you can find with date, amount, and lead investors; ' +
  '(2) Key people — founders and notable hires with their prior roles; ' +
  '(3) Product launches — Claude model releases with dates. ' +
  'Use only what research_corpus returns; do not search the web separately.';

const PRICING: Record<string, { input: number; output: number }> = {
  'claude-sonnet-4-6': { input: 3.0, output: 15.0 },
  'gpt-4o-mini': { input: 0.15, output: 0.6 },
};

function usageCost(model: string, usage: { input_tokens?: number; output_tokens?: number } | undefined): number {
  const p = PRICING[model];
  if (!p || !usage) return 0;
  return ((usage.input_tokens ?? 0) * p.input + (usage.output_tokens ?? 0) * p.output) / 1_000_000;
}

interface RunOutput {
  text: string;
  usage: { input_tokens?: number; output_tokens?: number };
  latency: number;
}

async function runBare(provider: 'anthropic' | 'openai', model: string, apiKey: string): Promise<RunOutput> {
  const chat = await initChatModel(`${provider}:${model}`, { apiKey });
  const agent = createAgent({ model: chat, tools: [researchCorpus] });
  const t0 = performance.now();
  const state = (await agent.invoke({ messages: [{ role: 'user', content: PROMPT }] })) as {
    messages: Array<{ content: unknown; usage_metadata?: { input_tokens?: number; output_tokens?: number } }>;
  };
  const latency = (performance.now() - t0) / 1000;
  const ai = state.messages[state.messages.length - 1]!;
  const text = typeof ai.content === 'string' ? ai.content : JSON.stringify(ai.content);
  return { text, usage: ai.usage_metadata ?? {}, latency };
}

async function runCompresr(compClient: CompressionClient, model: string): Promise<RunOutput> {
  const t0 = performance.now();
  const r = await compClient.run({ prompt: PROMPT, model, tools: [researchCorpus] as never, maxTokens: 600 });
  const latency = (performance.now() - t0) / 1000;
  return {
    text: r.text ?? '',
    usage: (r.usage as { input_tokens?: number; output_tokens?: number } | undefined) ?? {},
    latency,
  };
}

function report(label: string, model: string, bare: RunOutput, comp: RunOutput): void {
  const bareCost = usageCost(model, bare.usage);
  const compCost = usageCost(model, comp.usage);
  console.log(`${label.padEnd(22)} ${'without Compresr'.padStart(16)}    ${'with Compresr'.padStart(16)}`);
  console.log('-'.repeat(60));
  const row = (k: string, a: string | number, b: string | number) =>
    console.log(`${k.padEnd(22)} ${String(a).padStart(16)}    ${String(b).padStart(16)}`);
  row('latency (s)', bare.latency.toFixed(2), comp.latency.toFixed(2));
  const bareIn = bare.usage.input_tokens ?? 0;
  const compIn = comp.usage.input_tokens ?? 0;
  row('input_tokens', bareIn, compIn);
  row('output_tokens', bare.usage.output_tokens ?? 0, comp.usage.output_tokens ?? 0);
  row('cost (USD)', `$${bareCost.toFixed(5)}`, `$${compCost.toFixed(5)}`);
  const saved = bareCost - compCost;
  const inputPct = ((bareIn - compIn) * 100) / Math.max(1, bareIn);
  const pct = (saved * 100) / Math.max(bareCost, 1e-9);
  console.log();
  console.log(`Input tokens saved: ${bareIn - compIn} (${inputPct.toFixed(1)}%)`);
  console.log(`Cost saved: $${saved.toFixed(5)} (${pct.toFixed(1)}%)`);
  console.log(`Projected savings per 1,000 requests @ list price: $${(saved * 1000).toFixed(2)}`);
  console.log();
  console.log('--- Without Compresr (first 320 chars) ---');
  console.log(bare.text.slice(0, 320));
  console.log();
  console.log('--- With Compresr (first 320 chars) ---');
  console.log(comp.text.slice(0, 320));
}

async function main() {
  console.log('=== 1. Web search tool — Tavily live ===');
  const tavily = await WebSearchTool.tavily({ apiKey: TAVILY_API_KEY!, maxResults: 5 });
  const anthropicForSearch = new CompressionClient({
    apiKey: COMPRESR_API_KEY!,
    llm: 'anthropic',
    llmApiKey: ANTHROPIC_API_KEY!,
    compression: { compressionModelName: 'latte_v1', targetCompressionRatio: 0.5, minTokens: 500 },
  });
  try {
    const searchResp = await anthropicForSearch.messages.create({
      model: 'claude-haiku-4-5',
      maxTokens: 256,
      messages: [
        {
          role: 'user',
          content:
            'I am a VC analyst writing an investment memo on Anthropic. Use tavily_search to find one recent news item from the past 90 days. Summarize the top result in one sentence and cite its URL.',
        },
      ],
      tools: [tavily],
    });
    const first = searchResp.content[0];
    console.log(first && 'text' in first ? first.text : JSON.stringify(first));
  } catch (e) {
    console.log(`(Tavily live call failed — likely a plan/rate-limit issue: ${(e as Error).message})`);
  }

  console.log('\n=== 2. Page-fetch tool — Anthropic, off vs on (full benchmark) ===');
  const anthropicComp = new CompressionClient({
    apiKey: COMPRESR_API_KEY!,
    llm: 'anthropic',
    llmApiKey: ANTHROPIC_API_KEY!,
    compression: { compressionModelName: 'latte_v1', targetCompressionRatio: 0.5, minTokens: 500 },
  });
  const captures: Array<{ raw: string; cmp: string }> = [];
  const origAnthropicCompress = anthropicComp.compress.bind(anthropicComp);
  anthropicComp.compress = (async (opts: Parameters<typeof origAnthropicCompress>[0]) => {
    const r = await origAnthropicCompress(opts);
    captures.push({ raw: opts.context ?? '', cmp: r.data?.compressed_context ?? '' });
    return r;
  }) as typeof anthropicComp.compress;

  const [aBare, aComp] = await Promise.all([
    runBare('anthropic', 'claude-sonnet-4-6', ANTHROPIC_API_KEY!),
    runCompresr(anthropicComp, 'claude-sonnet-4-6'),
  ]);
  report('Anthropic Sonnet 4.6', 'claude-sonnet-4-6', aBare, aComp);

  if (captures.length > 0) {
    console.log(`\nSpy captured ${captures.length} compress() call(s) on the Anthropic run.`);
    console.log(`First tool output: ${captures[0]!.raw.length.toLocaleString()} chars raw -> ${captures[0]!.cmp.length.toLocaleString()} chars compressed`);
    printCompresrDiff(captures[0]!.raw, captures[0]!.cmp);
  } else {
    console.log('\n(No compressions captured — tool output may have been below min_tokens.)');
  }

  console.log('\n=== 3. Same benchmark — OpenAI provider ===');
  const openaiComp = new CompressionClient({
    apiKey: COMPRESR_API_KEY!,
    llm: 'openai',
    llmApiKey: OPENAI_API_KEY!,
    compression: { compressionModelName: 'latte_v1', targetCompressionRatio: 0.5, minTokens: 500 },
  });
  const [oBare, oComp] = await Promise.all([
    runBare('openai', 'gpt-4o-mini', OPENAI_API_KEY!),
    runCompresr(openaiComp, 'gpt-4o-mini'),
  ]);
  report('OpenAI gpt-4o-mini', 'gpt-4o-mini', oBare, oComp);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
