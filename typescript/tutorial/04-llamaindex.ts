/**
 * 04 — Compresr × LlamaIndex.TS
 *
 * Customer scenario: a financial analyst building a memo from Apple's
 * FY2024 Form 10-K filing (~218k chars, ~54k tokens fetched live from
 * SEC EDGAR). They ask narrow revenue / segment questions; the 10-K is a
 * dense dump of mostly-irrelevant prose for any single question. Three
 * groups of integrations cover the three places a LlamaIndex.TS agent
 * blows up:
 *
 *   A. Postprocessor   — CompresrNodePostprocessor (RAG)
 *   B. Tool wrapper    — wrapToolWithCompresr
 *   C. Memory          — CompresrMemoryBlock
 *
 * Section A retrieves chunks from the 10-K, then runs a real gpt-4o-mini
 * call without and with the postprocessor and prints a colored CLI diff
 * showing exactly which words were dropped.
 *
 * Run from the SDK root:
 *   npx tsx --env-file=.env typescript/tutorial/04-llamaindex.ts
 *
 * Once published: import { ... } from '@compresr/sdk/integrations/llamaindex';
 */
import OpenAI from 'openai';

import {
  CompresrMemoryBlock,
  CompresrNodePostprocessor,
  wrapToolWithCompresr,
} from '../src/integrations/llamaindex/index.js';
import { printCompresrDiff, printSavingsTable } from './_demo_utils.js';

const apiKey = process.env.COMPRESR_API_KEY;
if (!apiKey) throw new Error('Set COMPRESR_API_KEY.');
if (!process.env.OPENAI_API_KEY) throw new Error('Set OPENAI_API_KEY.');

const FILING_URL = 'https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm';
const FILING_TITLE = 'Apple Inc. Form 10-K (fiscal year ended September 28, 2024)';
const QUERY = 'What were Apple total net sales and Services revenue for fiscal year 2024?';
const oai = new OpenAI();

async function fetchSecFiling(url: string): Promise<string> {
  const r = await fetch(url, {
    headers: { 'User-Agent': 'compresr-sdk-tutorial compresr.founders@gmail.com' },
  });
  if (!r.ok) throw new Error(`SEC fetch failed: ${r.status}`);
  let html = await r.text();
  html = html.replace(/<script[\s\S]*?<\/script>/gi, ' ');
  html = html.replace(/<style[\s\S]*?<\/style>/gi, ' ');
  html = html.replace(/<[^>]+>/g, ' ');
  html = html.replace(/&nbsp;|&#160;/g, ' ');
  html = html.replace(/&amp;/g, '&');
  html = html.replace(/&[a-zA-Z#0-9]+;/g, ' ');
  html = html.replace(/[ \t]+/g, ' ');
  html = html.replace(/\n\s*\n+/g, '\n\n');
  return html.trim();
}

const toTokens = (s: string) => s.length >> 2;
const pct = (raw: number, cmp: number) => ((1 - cmp / raw) * 100).toFixed(1);

// LlamaIndex.TS schema stand-in — just enough for the postprocessor.
interface MutableNode {
  text: string;
  metadata: Record<string, unknown>;
  getContent(): string;
  setContent(t: string): void;
}

function makeNode(text: string, title: string, score: number) {
  const node: MutableNode = {
    text,
    metadata: { title },
    getContent() {
      return this.text;
    },
    setContent(t: string) {
      this.text = t;
    },
  };
  return { node, score };
}

// ---------- main -------------------------------------------------------------

async function main() {
  console.log('Compresr × LlamaIndex.TS — three groups, before / after.\n');

  console.log(`Fetching ${FILING_TITLE} from SEC EDGAR…`);
  const filingText = await fetchSecFiling(FILING_URL);
  console.log(`Filing: ${filingText.length.toLocaleString()} chars (~${toTokens(filingText).toLocaleString()} tokens)\n`);

  // ========================================================================
  // A. Postprocessor — CompresrNodePostprocessor
  // ========================================================================
  //
  // What it does: `BaseNodePostprocessor` that batch-compresses every
  // retrieved `NodeWithScore` before it reaches the LLM.
  //
  // When to use it: any RAG pipeline. Drop-in for `LongLLMLinguaPostprocessor`,
  // hosted (no GPU), batch-compresses all retrieved nodes in one call.
  console.log('--- A. CompresrNodePostprocessor ---');

  const CHUNK_SIZE = 8_000;
  const chunks: string[] = [];
  for (let i = 0; i < Math.min(8 * CHUNK_SIZE, filingText.length); i += CHUNK_SIZE) {
    chunks.push(filingText.slice(i, i + CHUNK_SIZE));
  }
  const nodes = chunks.map((text, i) => makeNode(text, `10-K chunk ${i + 1}`, 0.95 - 0.05 * i));
  const rawTexts = nodes.map((n) => n.node.text);
  const beforePerNode = rawTexts.map((t) => t.length);

  const pp = new CompresrNodePostprocessor({
    apiKey,
    compressionModel: 'latte_v2',
    targetCompressionRatio: 0.5,
    minTokens: 100,
  });
  const out = (await pp.postprocessNodes(nodes as never, {
    queryStr: QUERY,
  })) as Array<{ node: MutableNode }>;

  const withoutTotal = beforePerNode.reduce((a, b) => a + b, 0);
  const withTotal = out.reduce((acc, n) => acc + n.node.getContent().length, 0);

  console.log('Per node:');
  out.forEach((n, i) => {
    const rawLen = beforePerNode[i]!;
    const cmpLen = n.node.getContent().length;
    const rawT = rawLen >> 2;
    const cmpT = cmpLen >> 2;
    const title = n.node.metadata.title as string;
    console.log(
      `  ${title.padEnd(40)}${rawT.toLocaleString().padStart(7)} → ${cmpT.toLocaleString().padStart(6)} tokens (${pct(rawLen, cmpLen)}% smaller)`,
    );
  });
  console.log();
  console.log(
    `Without postprocessor: ${(withoutTotal >> 2).toLocaleString().padStart(7)} tokens reach the LLM`,
  );
  console.log(
    `With postprocessor:    ${(withTotal >> 2).toLocaleString().padStart(7)} tokens reach the LLM   (${pct(withoutTotal, withTotal)}% smaller overall)\n`,
  );

  const rawContext = rawTexts.join('\n\n');
  const cmpContext = out.map((n) => n.node.getContent()).join('\n\n');
  const SYS = 'You are a precise financial analyst. Use only the provided excerpts. Be concise (3-4 sentences) and cite numbers when present.';
  const [rawResp, cmpResp] = await Promise.all([
    oai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0,
      messages: [
        { role: 'system', content: SYS },
        { role: 'user', content: `Excerpts:\n${rawContext}\n\nQuestion: ${QUERY}` },
      ],
    }),
    oai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0,
      messages: [
        { role: 'system', content: SYS },
        { role: 'user', content: `Excerpts:\n${cmpContext}\n\nQuestion: ${QUERY}` },
      ],
    }),
  ]);
  const rawIn = rawResp.usage?.prompt_tokens ?? 0;
  const cmpIn = cmpResp.usage?.prompt_tokens ?? 0;
  console.log(`gpt-4o-mini input tokens: raw=${rawIn.toLocaleString()}  compresr=${cmpIn.toLocaleString()}  (${((1 - cmpIn / rawIn) * 100).toFixed(1)}% smaller)`);
  console.log(`$ saved / 1k requests at gpt-4o-mini ($0.15/M input): $${((rawIn - cmpIn) * 0.15 / 1000).toFixed(3)}\n`);
  console.log('--- Answer WITHOUT compression ---');
  console.log('  ' + (rawResp.choices[0]?.message.content?.trim() ?? '').replace(/\n/g, '\n  '));
  console.log('\n--- Answer WITH compression ---');
  console.log('  ' + (cmpResp.choices[0]?.message.content?.trim() ?? '').replace(/\n/g, '\n  '));
  console.log();
  printSavingsTable(rawIn, cmpIn);
  printCompresrDiff(rawContext, cmpContext);

  // ========================================================================
  // B. Tool wrapper — wrapToolWithCompresr
  // ========================================================================
  //
  // What it does: wraps any LlamaIndex tool so its return value is
  // compressed transparently before reaching the agent.
  //
  // When to use it: LlamaIndex agents (AgentRunner, ReActAgent, workflows)
  // where tools return long content. Zero-rewire — wrap the tool and
  // register the wrapped version.
  console.log('\n--- B. wrapToolWithCompresr ---');

  const TOOL_QUERY = 'What were Apple total operating expenses and R&D spend in fiscal 2024?';
  const filingTool = {
    metadata: { name: 'sec_filing_lookup', description: 'Fetch the full text of a SEC 10-K filing. Pass the analyst question as `query`.' },
    async call(_args: { query: string }) {
      return filingText;
    },
  };
  const wrapped = wrapToolWithCompresr(filingTool, {
    apiKey,
    compressionModel: 'latte_v2',
    queryArg: 'query',
    targetCompressionRatio: 0.3,
    minTokens: 500,
  });
  const rawOut = (await filingTool.call({ query: TOOL_QUERY })) as string;
  const cmpOut = (await wrapped.call({ query: TOOL_QUERY })) as string;
  console.log(
    `Without wrapper: ${toTokens(rawOut).toLocaleString().padStart(7)} tokens (raw tool output)`,
  );
  console.log(
    `With wrapper:    ${toTokens(cmpOut).toLocaleString().padStart(7)} tokens   (${pct(rawOut.length, cmpOut.length)}% smaller)`,
  );

  // ========================================================================
  // C. Memory — CompresrMemoryBlock
  // ========================================================================
  //
  // What it does: `BaseMemoryBlock` that aggregates messages into a buffer
  // and compresses on `get` once the buffer crosses `minTokens`.
  //
  // When to use it: long-running agents that flush short-term history
  // into long-term memory blocks. Token-level compression instead of an
  // LLM summarization call — faster, cheaper, preserves wording.
  console.log('\n--- C. CompresrMemoryBlock ---');

  // Without: raw buffer (every message kept verbatim).
  const withoutBlock = new CompresrMemoryBlock({
    apiKey,
    minTokens: 10_000_000, // effectively never compress
  });
  const memContent = filingText.slice(0, 30_000);
  await withoutBlock.put([
    { role: 'user', content: 'Walk me through the Consolidated Statements of Operations in the 10-K.' },
    { role: 'assistant', content: memContent },
  ]);
  const rawMem = String((await withoutBlock.get())[0]?.content ?? '');

  // With: same input, real compression.
  const withBlock = new CompresrMemoryBlock({
    apiKey,
    targetToken: 2_000,
    compressionModel: 'latte_v2',
  });
  await withBlock.put([
    { role: 'user', content: 'Walk me through the Consolidated Statements of Operations in the 10-K.' },
    { role: 'assistant', content: memContent },
  ]);
  const cmpMem = String((await withBlock.get())[0]?.content ?? '');

  console.log(
    `Without memory block: ${toTokens(rawMem).toLocaleString().padStart(7)} tokens (raw buffer)`,
  );
  console.log(
    `With memory block:    ${toTokens(cmpMem).toLocaleString().padStart(7)} tokens   (${pct(rawMem.length, cmpMem.length)}% smaller)`,
  );
  console.log();
  console.log('Wire it up:');
  console.log("  const memory = new Memory({ tokenLimit: 8_000, memoryBlocks: [block] });");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
