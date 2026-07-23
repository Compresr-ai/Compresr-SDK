const WIKI_UA = 'compresr-sdk-tutorial/1.0 (mailto:support@compresr.ai)';

export async function fetchWikipedia(title: string): Promise<string> {
  const url = new URL('https://en.wikipedia.org/w/api.php');
  url.search = new URLSearchParams({
    action: 'query',
    prop: 'extracts',
    explaintext: '1',
    exlimit: '1',
    titles: title,
    format: 'json',
    redirects: '1',
  }).toString();
  const r = await fetch(url, { headers: { 'User-Agent': WIKI_UA } });
  if (!r.ok) throw new Error(`Wikipedia fetch failed: ${r.status}`);
  const data = (await r.json()) as { query: { pages: Record<string, { extract?: string }> } };
  return Object.values(data.query.pages)[0]?.extract ?? '';
}

export async function fetchCorpus(titles: string[]): Promise<string> {
  const parts: string[] = [];
  for (const t of titles) {
    const text = await fetchWikipedia(t);
    if (text) parts.push(`${t}\n\n${text}`);
  }
  return parts.join('\n\n# ');
}

const CALLS_PER_DAY = 1_000;
const DAYS_PER_MONTH = 30;

export function printSavingsTable(rawTokens: number, cmpTokens: number): void {
  const prices: Record<string, number> = {
    'gpt-4o-mini': 0.15,
    'gpt-4o': 2.5,
    'claude-sonnet-4-6': 3.0,
    'gpt-5': 1.25,
  };
  console.log(
    `${'Model'.padEnd(20)}${'Raw $/mo'.padStart(14)}${'Compresr $/mo'.padStart(17)}${'Saved $/mo'.padStart(14)}`,
  );
  for (const [name, price] of Object.entries(prices)) {
    const raw = (rawTokens * CALLS_PER_DAY * DAYS_PER_MONTH * price) / 1_000_000;
    const cmp = (cmpTokens * CALLS_PER_DAY * DAYS_PER_MONTH * price) / 1_000_000;
    console.log(
      `${name.padEnd(20)}${raw.toFixed(2).padStart(14)}${cmp.toFixed(2).padStart(17)}${(raw - cmp).toFixed(2).padStart(14)}`,
    );
  }
}

function keyOf(w: string): string {
  return w.replace(/[^\p{L}\p{N}]/gu, '').toLowerCase();
}

const RED = '\x1b[91m';
const GREEN = '\x1b[32m';
const STRIKE = '\x1b[9m';
const RESET = '\x1b[0m';
const DIM = '\x1b[2m';

export function printCompresrDiff(raw: string, compressed: string, opts: { maxDisplayChars?: number } = {}): void {
  const maxDisplay = opts.maxDisplayChars ?? 8_000;

  const cmpCounter = new Map<string, number>();
  for (const w of compressed.split(/\s+/)) {
    const k = keyOf(w);
    if (k) cmpCounter.set(k, (cmpCounter.get(k) ?? 0) + 1);
  }

  let fullRawCount = 0;
  let fullKept = 0;
  const cmpFullCopy = new Map(cmpCounter);
  for (const w of raw.split(/\s+/)) {
    const k = keyOf(w);
    if (!k) continue;
    fullRawCount++;
    const c = cmpFullCopy.get(k) ?? 0;
    if (c > 0) {
      cmpFullCopy.set(k, c - 1);
      fullKept++;
    }
  }
  const overallPct = Math.floor((fullKept * 100) / Math.max(1, fullRawCount));

  console.log();
  console.log(
    `${DIM}Word-level diff:${RESET} ${fullKept.toLocaleString()} / ${fullRawCount.toLocaleString()} words kept (${overallPct}%).`,
  );
  console.log(`  ${GREEN}green = kept${RESET}    ${RED}${STRIKE}red strikethrough = dropped${RESET}`);
  console.log();

  const cmpDisplay = new Map(cmpCounter);
  const displayRaw = raw.length > maxDisplay ? raw.slice(0, maxDisplay) : raw;

  const out: string[] = [];
  for (const line of displayRaw.split('\n')) {
    const parts: string[] = [];
    for (const w of line.split(/\s+/).filter(Boolean)) {
      const k = keyOf(w);
      let kept = false;
      if (k) {
        const c = cmpDisplay.get(k) ?? 0;
        if (c > 0) {
          cmpDisplay.set(k, c - 1);
          kept = true;
        }
      }
      parts.push(kept ? `${GREEN}${w}${RESET}` : `${RED}${STRIKE}${w}${RESET}`);
    }
    out.push(parts.join(' '));
  }
  console.log(out.join('\n'));
  if (raw.length > maxDisplay) {
    console.log(`${DIM}… diff truncated; full corpus is ${raw.length.toLocaleString()} chars.${RESET}`);
  }
  console.log();
}
