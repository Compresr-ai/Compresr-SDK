/**
 * 01 — Quickstart for the Compresr TypeScript SDK.
 *
 * Customer scenario: a SaaS support bot reading a long product / industry
 * knowledge corpus (~45k tokens, stitched live from 12 Wikipedia articles)
 * and answering one specific user question. We call OpenAI gpt-4o-mini twice
 * — once on the raw corpus, once on the Compresr-compressed corpus — and
 * print the measured input-token + dollar delta plus a colored CLI diff
 * showing exactly which tokens Compresr dropped.
 *
 * Run from the SDK root:
 *   npx tsx --env-file=.env typescript/tutorial/01-quickstart.ts
 *
 * Once `@compresr/sdk` is on npm, change the import below to:
 *   import { CompressionClient } from '@compresr/sdk';
 */
import OpenAI from 'openai';
import { CompressionClient } from '../src/index.js';
import { fetchCorpus, printCompresrDiff, printSavingsTable } from './_demo_utils.js';

const apiKey = process.env.COMPRESR_API_KEY;
if (!apiKey) throw new Error('Set COMPRESR_API_KEY in your environment.');
if (!process.env.OPENAI_API_KEY) throw new Error('Set OPENAI_API_KEY in your environment.');

const client = new CompressionClient({ apiKey });
const oai = new OpenAI();

const CORPUS_TITLES = [
  'Software as a service',
  'Cloud computing',
  'Subscription business model',
  'Multitenancy',
  'Customer relationship management',
  'Software industry',
  'Application service provider',
  'Enterprise software',
  'Web application',
  'Software development',
  'Information technology',
  'Computer software',
];
const QUESTION =
  'According to these articles, what happens to a SaaS business if a significant number of customers cancel their subscriptions, and why do SaaS companies offer freemium tiers?';

async function ask(corpus: string): Promise<{ answer: string; promptTokens: number }> {
  const resp = await oai.chat.completions.create({
    model: 'gpt-4o-mini',
    temperature: 0,
    messages: [
      { role: 'system', content: 'You are a SaaS customer-support analyst. Answer using ONLY the provided knowledge corpus. Be concise (2-3 sentences).' },
      { role: 'user', content: `CORPUS:\n${corpus}\n\nQUESTION: ${QUESTION}` },
    ],
  });
  return {
    answer: resp.choices[0]?.message.content?.trim() ?? '',
    promptTokens: resp.usage?.prompt_tokens ?? 0,
  };
}

async function main() {
  console.log(`Fetching ${CORPUS_TITLES.length} SaaS / cloud Wikipedia articles…`);
  const article = await fetchCorpus(CORPUS_TITLES);
  console.log(`Corpus: ${article.length.toLocaleString()} chars (~${Math.round(article.length / 4).toLocaleString()} tokens)`);
  console.log(`Question: ${QUESTION}\n`);

  console.log('--- 1. Query-aware compression (latte_v1) ---');
  const compressed = await client.compress({
    context: article,
    query: QUESTION,
    targetCompressionRatio: 0.5,
  });
  const cmpText = compressed.data?.compressed_context ?? '';
  console.log(`Original tokens:   ${compressed.data?.original_tokens?.toLocaleString()}`);
  console.log(`Compressed tokens: ${compressed.data?.compressed_tokens?.toLocaleString()}`);
  console.log(`Tokens saved:      ${compressed.data?.tokens_saved?.toLocaleString()}\n`);

  console.log('--- 2. gpt-4o-mini answers, raw vs compressed ---');
  const [raw, cmp] = await Promise.all([ask(article), ask(cmpText)]);
  const savedPct = ((1 - cmp.promptTokens / raw.promptTokens) * 100).toFixed(1);
  const savedDollars = ((raw.promptTokens - cmp.promptTokens) * 0.15 / 1000).toFixed(3);
  console.log(`Without compression: ${raw.promptTokens.toLocaleString()} prompt tokens`);
  console.log(`With compression:    ${cmp.promptTokens.toLocaleString()} prompt tokens  (${savedPct}% smaller)`);
  console.log(`$ saved / 1k requests at gpt-4o-mini ($0.15/M input): $${savedDollars}\n`);
  console.log('Without compression answer:\n  ' + raw.answer.replace(/\n/g, '\n  '));
  console.log('\nWith compression answer:\n  ' + cmp.answer.replace(/\n/g, '\n  '));

  console.log();
  printSavingsTable(raw.promptTokens, cmp.promptTokens);

  printCompresrDiff(article, cmpText);

  console.log('--- 3. Batch compression (convenience form) ---');
  const third = Math.floor(article.length / 3);
  const slices = [article.slice(0, third), article.slice(third, 2 * third), article.slice(2 * third)];
  const a = await client.compressBatch({
    contexts: slices,
    queries: QUESTION,
    targetCompressionRatio: 0.5,
  });
  console.log(`[A] ${a.data?.count} segments — saved ${a.data?.total_tokens_saved?.toLocaleString()} tokens`);

  console.log('\n--- 4. Batch compression (pair form, one query per segment) ---');
  const b = await client.compressBatch({
    inputs: [
      { context: slices[0]!, query: 'How is SaaS typically priced?' },
      { context: slices[1]!, query: 'What are the legal and compliance challenges of SaaS?' },
      { context: slices[2]!, query: 'What architectural patterns are common in SaaS?' },
    ],
    targetCompressionRatio: 0.5,
  });
  console.log(`[B] ${b.data?.count} segments — saved ${b.data?.total_tokens_saved?.toLocaleString()} tokens`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
