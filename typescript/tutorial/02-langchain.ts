/**
 * 02 — Compresr × LangChain.js
 *
 * Customer scenario: a competitive-intelligence analyst at a hedge fund.
 * Their LangChain agent fetches a long research corpus (~55k tokens
 * stitched from 5 real Wikipedia articles — Microsoft + Azure + History
 * of Microsoft + Bill Gates + Satya Nadella) and answers one narrow
 * question about Intelligent Cloud / Azure. Three groups of integrations
 * cover the three places the agent's prompt blows up:
 *
 *   A. Tool wrapper       — wrapToolWithCompression
 *   B. Middleware (3)     — compresrToolMiddleware
 *                           compresrSummarizationMiddleware
 *                           compresrPromptMiddleware
 *   C. Retriever          — CompresrExtractor
 *
 * Section A runs a real gpt-4o-mini call without and with compression and
 * prints a colored CLI diff showing exactly which words were dropped.
 *
 * Run from the SDK root:
 *   npx tsx --env-file=.env typescript/tutorial/02-langchain.ts
 *
 * Once published: import { ... } from '@compresr/sdk/integrations/langchain';
 */
import { Document } from '@langchain/core/documents';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import OpenAI from 'openai';
import { z } from 'zod';

import {
  CompresrExtractor,
  compresrPromptMiddleware,
  compresrSummarizationMiddleware,
  compresrToolMiddleware,
  wrapToolWithCompression,
} from '../src/integrations/langchain/index.js';
import { fetchCorpus, fetchWikipedia, printCompresrDiff, printSavingsTable } from './_demo_utils.js';

const apiKey = process.env.COMPRESR_API_KEY;
if (!apiKey) throw new Error('Set COMPRESR_API_KEY.');
if (!process.env.OPENAI_API_KEY) throw new Error('Set OPENAI_API_KEY.');

const CORPUS_TITLES = ['Microsoft', 'Microsoft Azure', 'History of Microsoft', 'Bill Gates', 'Satya Nadella'];
const COMPANY = 'Microsoft';
const QUERY = 'How large is Microsoft Intelligent Cloud / Azure and what has its growth been?';
const oai = new OpenAI();

const toTokens = (s: string) => s.length >> 2;
const pct = (raw: number, cmp: number) => ((1 - cmp / raw) * 100).toFixed(1);

// ---------- main -------------------------------------------------------------

async function main() {
  console.log('Compresr × LangChain.js — three groups, before / after.\n');

  console.log(`Fetching ${CORPUS_TITLES.length}-article research corpus on ${COMPANY}…`);
  const article = await fetchCorpus(CORPUS_TITLES);
  console.log(`Corpus: ${article.length.toLocaleString()} chars (~${toTokens(article).toLocaleString()} tokens)\n`);

  // ========================================================================
  // A. Tool wrapper — wrapToolWithCompression
  // ========================================================================
  //
  // What it does: wraps any LangChain `StructuredTool` so its output is
  // compressed before being returned.
  //
  // When to use it: the agent is already built; you want compression with
  // zero rewiring. Wrap the tool, swap it in, done.
  console.log('--- A. wrapToolWithCompression ---');

  const companyResearch = tool(
    async (_input: { query: string }) => fetchCorpus(CORPUS_TITLES),
    {
      name: 'company_research',
      description: 'Fetch a stitched research corpus on a public company.',
      schema: z.object({ query: z.string() }),
    },
  );

  const smartLookup = wrapToolWithCompression(companyResearch, {
    apiKey,
    compressionModel: 'latte_v2',
    queryArg: 'query',
    targetCompressionRatio: 0.5,
    minTokens: 100,
  });

  const rawOut = (await companyResearch.invoke({ query: COMPANY })) as string;
  const cmpOut = (await smartLookup.invoke({ query: COMPANY })) as string;

  const SYS = 'You are a precise equity analyst. Answer only from the provided corpus in 2-3 sentences.';
  const [rawResp, cmpResp] = await Promise.all([
    oai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0,
      messages: [
        { role: 'system', content: SYS },
        { role: 'user', content: `Corpus:\n${rawOut}\n\nQuestion: ${QUERY}` },
      ],
    }),
    oai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0,
      messages: [
        { role: 'system', content: SYS },
        { role: 'user', content: `Corpus:\n${cmpOut}\n\nQuestion: ${QUERY}` },
      ],
    }),
  ]);
  const rawIn = rawResp.usage?.prompt_tokens ?? 0;
  const cmpIn = cmpResp.usage?.prompt_tokens ?? 0;
  console.log(`Without wrapper: ${rawIn.toLocaleString().padStart(8)} prompt tokens`);
  console.log(
    `With wrapper:    ${cmpIn.toLocaleString().padStart(8)} prompt tokens  (${((1 - cmpIn / rawIn) * 100).toFixed(1)}% smaller)`,
  );
  console.log(`$ saved / 1k requests at gpt-4o-mini ($0.15/M input): $${((rawIn - cmpIn) * 0.15 / 1000).toFixed(3)}\n`);
  console.log('--- Answer WITHOUT compression ---');
  console.log('  ' + (rawResp.choices[0]?.message.content?.trim() ?? '').replace(/\n/g, '\n  '));
  console.log('\n--- Answer WITH compression ---');
  console.log('  ' + (cmpResp.choices[0]?.message.content?.trim() ?? '').replace(/\n/g, '\n  '));
  console.log();
  printSavingsTable(rawIn, cmpIn);
  printCompresrDiff(rawOut, cmpOut);

  // ========================================================================
  // B. Middleware — for `createAgent`
  // ========================================================================
  //
  // Three layers — per-tool, history, last-mile — that compose into one
  // `middleware: [...]` array on `createAgent`. Each one shown below with
  // its own before / after.

  // B.1 compresrToolMiddleware ---------------------------------------------
  //
  // When to use it: any agent with tools that return long content. Sits
  // between the tool node and agent state via `wrapToolCall`. Use
  // allowTools / ignoreTools to target only the noisy tools.
  console.log('\n--- B.1 compresrToolMiddleware ---');

  const toolMw = compresrToolMiddleware({
    apiKey,
    compressionModel: 'latte_v2',
    queryArg: 'query',
    allowTools: ['company_research'],
    targetCompressionRatio: 0.5,
  });

  const innerHandler = () =>
    new ToolMessage({ content: article, tool_call_id: 't1', name: 'company_research' });
  const result = (await toolMw.wrapToolCall!(
    {
      toolCall: { id: 't1', name: 'company_research', args: { query: QUERY } },
      messages: [new HumanMessage({ content: QUERY })],
    },
    innerHandler,
  )) as ToolMessage;
  const toolAfter = String(result.content);
  console.log(
    `Without middleware: ${toTokens(article).toLocaleString().padStart(8)} tokens (raw tool output)`,
  );
  console.log(
    `With middleware:    ${toTokens(toolAfter).toLocaleString().padStart(8)} tokens   (${pct(article.length, toolAfter.length)}% smaller)`,
  );

  // B.2 compresrSummarizationMiddleware ------------------------------------
  //
  // When to use it: long-running conversations that accumulate tool
  // outputs. When state crosses the threshold, old messages collapse into
  // one summary; recent tail stays untouched. KV-cache friendly because
  // the summary is stable across turns.
  console.log('\n--- B.2 compresrSummarizationMiddleware ---');

  const summaryMw = compresrSummarizationMiddleware({
    apiKey,
    compressionModel: 'latte_v2',
    maxTokensBeforeSummary: 4_000,
    messagesToKeep: 10,
  });

  const longToolOut = article.slice(0, 8_000);
  const msgs: unknown[] = [];
  for (let i = 0; i < 8; i++) {
    msgs.push(new HumanMessage({ content: `Question ${i}: tell me more.` }));
    msgs.push(new ToolMessage({ content: longToolOut, tool_call_id: `t${i}`, name: 'wiki' }));
  }
  const beforeTokens = msgs.reduce(
    (acc, m) => acc + toTokens(String((m as { content: unknown }).content ?? '')),
    0,
  );
  const out = await summaryMw.beforeModel!({ messages: msgs });
  const newMsgs = (out!.messages as unknown[]).slice(1); // index 0 is RemoveMessage marker
  const afterTokens = newMsgs.reduce(
    (acc, m) => acc + toTokens(String((m as { content: unknown }).content ?? '')),
    0,
  );
  console.log(
    `Without middleware: ${msgs.length.toString().padStart(2)} messages, ${beforeTokens.toLocaleString().padStart(7)} tokens (all retained)`,
  );
  console.log(
    `With middleware:    ${newMsgs.length.toString().padStart(2)} messages, ${afterTokens.toLocaleString().padStart(7)} tokens   (${pct(beforeTokens, afterTokens)}% smaller)`,
  );

  // B.3 compresrPromptMiddleware -------------------------------------------
  //
  // When to use it: last-mile safety net. Walks the messages
  // largest-first via wrapModelCall and compresses just enough to fit.
  // Doesn't pollute agent state — the shrink only affects what's handed
  // to the model.
  console.log('\n--- B.3 compresrPromptMiddleware ---');

  const promptMw = compresrPromptMiddleware({
    apiKey,
    maxTokens: 4_000,
    minTokens: 500,
    compressionModel: 'latte_v2',
    query: QUERY,
  });
  const messages = [
    new HumanMessage({ content: QUERY }),
    new ToolMessage({ content: article, tool_call_id: 't1', name: 'company_research' }),
  ];
  let captured: unknown[] = [];
  await promptMw.wrapModelCall!({ messages }, async (r) => {
    captured = r.messages;
    return null;
  });
  const promptBefore = messages.reduce((acc, m) => acc + toTokens(String(m.content)), 0);
  const promptAfter = captured.reduce(
    (acc, m) => acc + toTokens(String((m as { content: unknown }).content ?? '')),
    0,
  );
  console.log(
    `Without prompt cap: ${promptBefore.toLocaleString().padStart(7)} tokens  (would blow past 4k budget)`,
  );
  console.log(
    `With prompt cap:    ${promptAfter.toLocaleString().padStart(7)} tokens   (${pct(promptBefore, promptAfter)}% smaller — fits 4k budget)`,
  );

  // ========================================================================
  // C. Retriever — CompresrExtractor
  // ========================================================================
  //
  // What it does: `BaseDocumentCompressor` that batch-compresses every
  // retrieved document in one call.
  //
  // When to use it: RAG pipelines using `ContextualCompressionRetriever`.
  // Drop-in for `LLMChainExtractor` — same shape, no LLM call per doc.
  console.log('\n--- C. CompresrExtractor ---');

  const topics = ['Microsoft', 'Microsoft Azure', 'Amazon Web Services', 'Google Cloud Platform', 'Oracle Cloud'];
  const docs = await Promise.all(
    topics.map(async (t) => new Document({ pageContent: await fetchWikipedia(t), metadata: { title: t } })),
  );

  const extractor = new CompresrExtractor({
    apiKey,
    compressionModel: 'latte_v2',
    targetCompressionRatio: 0.5,
    minTokens: 100,
  });
  const analystQuery = 'Compare the size and growth of the major public cloud providers (Azure, AWS, GCP, Oracle).';
  const compressedDocs = await extractor.compressDocuments(docs, analystQuery);
  const withoutTotal = docs.reduce((acc, d) => acc + d.pageContent.length, 0);
  const withTotal = compressedDocs.reduce((acc, d) => acc + d.pageContent.length, 0);

  console.log('Per document:');
  compressedDocs.forEach((d, i) => {
    const rawT = toTokens(docs[i]!.pageContent);
    const cmpT = toTokens(d.pageContent);
    const title = d.metadata.title as string;
    console.log(
      `  ${title.padEnd(40)}${rawT.toLocaleString().padStart(7)} → ${cmpT.toLocaleString().padStart(6)} tokens (${pct(docs[i]!.pageContent.length, d.pageContent.length)}% smaller)`,
    );
  });
  console.log();
  console.log(`Without extractor: ${(withoutTotal >> 2).toLocaleString().padStart(7)} tokens total`);
  console.log(
    `With extractor:    ${(withTotal >> 2).toLocaleString().padStart(7)} tokens total (${pct(withoutTotal, withTotal)}% smaller overall)`,
  );

  console.log('\nHow they compose:');
  console.log('  createAgent({');
  console.log('    model, tools,');
  console.log('    middleware: [toolMw, summaryMw, promptMw],');
  console.log('  });');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
