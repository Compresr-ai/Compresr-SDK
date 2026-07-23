/**
 * 03 — Compresr × LangGraph.js
 *
 * Customer scenario: a corporate-strategy research graph. The graph fetches
 * a long research corpus (~45k tokens stitched from 12 Wikipedia articles
 * centered on Salesforce + Marc Benioff + Tableau + Slack + MuleSoft + the
 * surrounding CRM industry) and an analyst asks a focused M&A question.
 * Four groups of integrations cover the four places the graph's state and
 * storage blow up:
 *
 *   A. Graph node     — makeCompresrNode (drop into any StateGraph)
 *   B. Middleware     — compresrPromptMiddleware (others re-exported)
 *   C. Storage        — CompresrCheckpointSerializer + CompresrStore
 *   D. Multi-agent    — compresrHandoffTool
 *
 * Section A runs a real gpt-4o-mini call without and with the compression
 * node and prints a colored CLI diff showing exactly which words dropped.
 *
 * Run from the SDK root:
 *   npx tsx --env-file=.env typescript/tutorial/03-langgraph.ts
 */
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import { Annotation, END, InMemoryStore, START, StateGraph } from '@langchain/langgraph';
import OpenAI from 'openai';

import {
  CompresrCheckpointSerializer,
  CompresrStore,
  compresrHandoffTool,
  compresrPromptMiddleware,
  makeCompresrNode,
} from '../src/integrations/langgraph/index.js';
import { fetchCorpus, printCompresrDiff, printSavingsTable } from './_demo_utils.js';

const apiKey = process.env.COMPRESR_API_KEY;
if (!apiKey) throw new Error('Set COMPRESR_API_KEY.');
if (!process.env.OPENAI_API_KEY) throw new Error('Set OPENAI_API_KEY.');

const CORPUS_TITLES = [
  'Salesforce','Marc Benioff','Tableau Software','Slack Technologies','MuleSoft',
  'Customer relationship management','Heroku','Software industry','Enterprise software',
  'Software as a service','Cloud computing','Service-oriented architecture',
];
const COMPANY = 'Salesforce';
const QUERY = 'What are Salesforce major acquisitions and how have they shaped its product portfolio?';
const oai = new OpenAI();

// ---------- Shared helpers ---------------------------------------------------

const toTokens = (s: string) => s.length >> 2;
const pct = (raw: number, cmp: number) => ((1 - cmp / raw) * 100).toFixed(1);

// ---------- main -------------------------------------------------------------

const GraphState = Annotation.Root({
  user_question: Annotation<string>(),
  retrieved_text: Annotation<string>(),
});
type State = typeof GraphState.State;

async function main() {
  console.log('Compresr × LangGraph.js — four groups, before / after.\n');

  console.log(`Fetching ${CORPUS_TITLES.length}-article research corpus on ${COMPANY}…`);
  const article = await fetchCorpus(CORPUS_TITLES);
  console.log(`Corpus: ${article.length.toLocaleString()} chars (~${toTokens(article).toLocaleString()} tokens)\n`);

  // ========================================================================
  // A. Graph node — makeCompresrNode
  // ========================================================================
  //
  // What it does: a `StateGraph` node that compresses one state field
  // and writes the smaller version back.
  //
  // When to use it: custom graphs. Drop it between retrieval and the
  // LLM step — every iteration gets the smaller version.
  console.log('--- A. makeCompresrNode ---');

  async function retrieve(_state: State) {
    return { retrieved_text: article };
  }
  async function consume(_state: State) {
    return {};
  }
  const compress = makeCompresrNode<State>({
    apiKey,
    contextKey: 'retrieved_text',
    queryKey: 'user_question',
    compressionModel: 'latte_v2',
    targetCompressionRatio: 0.5,
  });

  // Without compression: retrieve → consume.
  const gRaw = new StateGraph(GraphState)
    .addNode('retrieve', retrieve)
    .addNode('consume', consume)
    .addEdge(START, 'retrieve')
    .addEdge('retrieve', 'consume')
    .addEdge('consume', END);
  const rawFinal = await gRaw.compile().invoke({ user_question: QUERY, retrieved_text: '' });

  // With compression: retrieve → compress → consume.
  const gCmp = new StateGraph(GraphState)
    .addNode('retrieve', retrieve)
    .addNode('compress', compress)
    .addNode('consume', consume)
    .addEdge(START, 'retrieve')
    .addEdge('retrieve', 'compress')
    .addEdge('compress', 'consume')
    .addEdge('consume', END);
  const cmpFinal = await gCmp.compile().invoke({ user_question: QUERY, retrieved_text: '' });

  const SYS = 'You are a precise strategy analyst. Answer only from the provided corpus in 3-4 sentences.';
  const [rawResp, cmpResp] = await Promise.all([
    oai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0,
      messages: [
        { role: 'system', content: SYS },
        { role: 'user', content: `Corpus:\n${rawFinal.retrieved_text}\n\nQuestion: ${QUERY}` },
      ],
    }),
    oai.chat.completions.create({
      model: 'gpt-4o-mini',
      temperature: 0,
      messages: [
        { role: 'system', content: SYS },
        { role: 'user', content: `Corpus:\n${cmpFinal.retrieved_text}\n\nQuestion: ${QUERY}` },
      ],
    }),
  ]);
  const rawIn = rawResp.usage?.prompt_tokens ?? 0;
  const cmpIn = cmpResp.usage?.prompt_tokens ?? 0;
  console.log(`Without node: ${rawIn.toLocaleString().padStart(7)} prompt tokens reach the LLM step`);
  console.log(
    `With node:    ${cmpIn.toLocaleString().padStart(7)} prompt tokens reach the LLM step   (${((1 - cmpIn / rawIn) * 100).toFixed(1)}% smaller)`,
  );
  console.log(`$ saved / 1k requests at gpt-4o-mini ($0.15/M input): $${((rawIn - cmpIn) * 0.15 / 1000).toFixed(3)}\n`);
  console.log('--- Answer WITHOUT compression ---');
  console.log('  ' + (rawResp.choices[0]?.message.content?.trim() ?? '').replace(/\n/g, '\n  '));
  console.log('\n--- Answer WITH compression ---');
  console.log('  ' + (cmpResp.choices[0]?.message.content?.trim() ?? '').replace(/\n/g, '\n  '));
  console.log();
  printSavingsTable(rawIn, cmpIn);
  printCompresrDiff(rawFinal.retrieved_text, cmpFinal.retrieved_text);

  // ========================================================================
  // B. Middleware — re-exports from LangChain
  // ========================================================================
  //
  // LangGraph.js 1.x uses the same middleware mechanism as `createAgent`,
  // so the three middlewares from tutorial 02 work here. Re-exports live
  // on `compresr.integrations.langgraph` for discoverability.
  //
  // Quick proof using compresrPromptMiddleware — see tutorial 02 for the
  // full per-middleware walkthrough.
  console.log('\n--- B. compresrPromptMiddleware (middleware re-exports) ---');

  const promptMw = compresrPromptMiddleware({
    apiKey,
    maxTokens: 4_000,
    minTokens: 500,
    compressionModel: 'latte_v2',
    query: QUERY,
  });
  const messages = [
    new HumanMessage({ content: QUERY }),
    new ToolMessage({ content: article, tool_call_id: 't1', name: 'wiki' }),
  ];
  let captured: unknown[] = [];
  await promptMw.wrapModelCall!({ messages }, async (r) => {
    captured = r.messages;
    return null;
  });
  const without = messages.reduce((acc, m) => acc + toTokens(String(m.content)), 0);
  const withMw = captured.reduce(
    (acc, m) => acc + toTokens(String((m as { content: unknown }).content ?? '')),
    0,
  );
  console.log(
    `Without middleware: ${without.toLocaleString().padStart(7)} tokens  (over 4k budget)`,
  );
  console.log(
    `With middleware:    ${withMw.toLocaleString().padStart(7)} tokens   (${pct(without, withMw)}% smaller — fits)`,
  );

  // ========================================================================
  // C. Storage — checkpoint serializer + store wrapper
  // ========================================================================
  //
  // Two integrations for LangGraph's persistence layer:
  //
  // - CompresrCheckpointSerializer: drop-in for JsonPlusSerializer.
  //   Lossy by design — use the `fields` allowlist in production.
  // - CompresrStore: wraps any `BaseStore` and compresses on `put`.
  //   Reads are passthrough.
  //
  // When to use them: when checkpoint / store size is a real cost —
  // Postgres rewrites the whole TOAST row on every state update.

  // C.1 CompresrCheckpointSerializer
  console.log('\n--- C.1 CompresrCheckpointSerializer ---');
  const serializer = new CompresrCheckpointSerializer({
    apiKey,
    fields: new Set(['retrieved_text', 'scratchpad']),
    minTokens: 500,
  });
  const state = { retrieved_text: article, topic: 'salesforce-strategy' };
  const [tag, encoded] = await serializer.dumpsTyped(state);
  const decoded = serializer.loadsTyped([tag, encoded]) as Record<string, unknown>;
  const ckBefore = toTokens(article);
  const ckAfter = toTokens(String((decoded.retrieved_text as { v: string }).v));
  console.log(
    `Checkpoint without serializer: ${ckBefore.toLocaleString().padStart(7)} tokens stored`,
  );
  console.log(
    `Checkpoint with serializer:    ${ckAfter.toLocaleString().padStart(7)} tokens stored   (${pct(article.length, String((decoded.retrieved_text as { v: string }).v).length)}% smaller)`,
  );

  // C.2 CompresrStore
  console.log('\n--- C.2 CompresrStore ---');
  const rawStore = new InMemoryStore();
  await rawStore.put(['users', 'alice'], 'notes', { retrieved_text: article, tag: 'crm' });
  const rawValue = String(
    ((await rawStore.get(['users', 'alice'], 'notes'))!.value as Record<string, unknown>)
      .retrieved_text,
  );

  const innerStore = new InMemoryStore();
  const compresrStore = new CompresrStore(innerStore, {
    apiKey,
    fields: new Set(['retrieved_text']),
    minTokens: 100,
  });
  await compresrStore.put(['users', 'alice'], 'notes', {
    retrieved_text: article,
    tag: 'crm',
  });
  const wrappedValue = String(
    ((await innerStore.get(['users', 'alice'], 'notes'))!.value as Record<string, unknown>)
      .retrieved_text,
  );

  console.log(
    `Without store wrapper: ${toTokens(rawValue).toLocaleString().padStart(7)} tokens stored`,
  );
  console.log(
    `With store wrapper:    ${toTokens(wrappedValue).toLocaleString().padStart(7)} tokens stored   (${pct(rawValue.length, wrappedValue.length)}% smaller)`,
  );

  // ========================================================================
  // D. Multi-agent — compresrHandoffTool
  // ========================================================================
  //
  // What it does: a tool that emits Command(goto=..., graph=PARENT) with
  // compressed `task_description` + `context` fields.
  //
  // When to use it: supervisor / subagent topologies. The supervisor
  // forwards a long context excerpt (a RAG snippet, an earlier scratchpad)
  // without bloating the subagent's starting prompt.
  console.log('\n--- D. compresrHandoffTool ---');

  const financialHandoff = compresrHandoffTool('financial_analyst', {
    apiKey,
    minTokens: 100,
  });
  const handoffCall = {
    name: 'transfer_to_financial_analyst',
    args: {
      task_description: `Estimate the impact of ${COMPANY} acquisitions on revenue mix.`,
      context: article,
    },
    type: 'tool_call' as const,
    id: 'tc1',
  };
  const cmd = (await financialHandoff.invoke(handoffCall)) as {
    goto: string | string[];
    update: Record<string, unknown>;
  };
  const handoffBefore = toTokens(article);
  const handoffAfter = toTokens(String(cmd.update.context));
  const goto = Array.isArray(cmd.goto) ? cmd.goto[0] : cmd.goto;
  console.log(`goto agent     : ${goto}   (graph = Command.PARENT)`);
  console.log(
    `Without handoff: ${handoffBefore.toLocaleString().padStart(7)} tokens of context forwarded`,
  );
  console.log(
    `With handoff:    ${handoffAfter.toLocaleString().padStart(7)} tokens of context forwarded   (${pct(article.length, String(cmd.update.context).length)}% smaller)`,
  );

  console.log('\nHow they compose:');
  console.log('  - Inside the graph:    makeCompresrNode');
  console.log('  - Around the model:    middleware (tool / summary / prompt)');
  console.log('  - At rest:             CompresrCheckpointSerializer, CompresrStore');
  console.log('  - Between agents:      compresrHandoffTool');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
