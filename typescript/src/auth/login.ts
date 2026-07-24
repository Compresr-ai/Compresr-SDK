/** Browser-based login: opens the consent page, receives the key on a
 * loopback callback, stores it in ``~/.compresr/credentials.json``. */

import { createServer, IncomingMessage, Server, ServerResponse } from 'node:http';
import { randomBytes, timingSafeEqual } from 'node:crypto';

import { save, clear, DEFAULT_PROFILE } from './credentials.js';
import { openBrowser } from './browser.js';

const PORT_START = 9876;
const PORT_END = 9885;
const STATE_LEN = 64;
const POLL_INTERVAL_MS = 100;
const PROGRESS_INTERVAL_MS = 10_000;
const MAX_TIMEOUT_MS = 600_000;
const DEV_HOST_ALLOWED_ENV = 'COMPRESR_ALLOW_LOCAL_URLS';

const ALLOWED_APP_HOSTS = new Set([
  'compresr.ai',
  'www.compresr.ai',
  'staging.compresr.ai',
]);
const ALLOWED_BASE_HOSTS = new Set(['api.compresr.ai', 'api-staging.compresr.ai']);
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

const ERROR_CODE_WHITELIST = new Set([
  'state_mismatch',
  'no_token',
  'access_denied',
  'server_error',
  'invalid_request',
  'unauthorized_client',
  'temporarily_unavailable',
]);

const LANDING_CSS = `* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  display: flex; align-items: center; justify-content: center;
  min-height: 100vh; padding: 24px;
  background: radial-gradient(1200px 600px at 50% 0%, #eef2ff 0%, #ffffff 60%);
  color: #0f172a;
}
.card {
  max-width: 420px; width: 100%; background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.08); border-radius: 16px;
  box-shadow: 0 20px 40px -20px rgba(15, 23, 42, 0.15);
  padding: 32px 28px; text-align: center;
}
.badge {
  width: 64px; height: 64px; border-radius: 999px;
  display: inline-flex; align-items: center; justify-content: center;
  margin-bottom: 20px;
}
.badge svg { width: 32px; height: 32px; }
.badge-ok { background: rgba(16, 185, 129, 0.12); color: #059669; }
.badge-err { background: rgba(239, 68, 68, 0.12); color: #dc2626; }
h1 { font-size: 20px; font-weight: 600; margin-bottom: 8px; letter-spacing: -0.01em; }
p { font-size: 14px; color: #475569; line-height: 1.55; }
p + p { margin-top: 12px; }
.brand { margin-top: 24px; font-size: 12px; color: #94a3b8; letter-spacing: 0.05em; }
kbd {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
  padding: 2px 6px; border-radius: 4px; background: #f1f5f9; color: #0f172a;
  border: 1px solid rgba(15, 23, 42, 0.08);
}
@media (prefers-color-scheme: dark) {
  body { background: radial-gradient(1200px 600px at 50% 0%, #0b1220 0%, #030712 60%); color: #e2e8f0; }
  .card { background: #0f172a; border-color: rgba(255,255,255,0.08); }
  p { color: #94a3b8; }
  kbd { background: #1e293b; color: #e2e8f0; border-color: rgba(255,255,255,0.08); }
  .brand { color: #64748b; }
}`;

const CHECK_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>';
const X_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>';

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function sanitizeErrorCode(raw: string | null | undefined): string {
  if (!raw) return 'unknown_error';
  if (ERROR_CODE_WHITELIST.has(raw)) return raw;
  if (/^[A-Za-z0-9_-]{1,32}$/.test(raw)) return raw;
  return 'unknown_error';
}

export function landingHtml(opts: { ok: boolean; title: string; body: string }): string {
  const badgeClass = opts.ok ? 'badge-ok' : 'badge-err';
  const svg = opts.ok ? CHECK_SVG : X_SVG;
  const safeTitle = escapeHtml(opts.title);
  return (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">' +
    '<meta name="viewport" content="width=device-width,initial-scale=1">' +
    `<title>${safeTitle} — Compresr</title>` +
    `<style>${LANDING_CSS}</style></head><body>` +
    '<main class="card">' +
    `<div class="badge ${badgeClass}">${svg}</div>` +
    `<h1>${safeTitle}</h1>` +
    opts.body +
    '<div class="brand">compresr</div>' +
    '</main></body></html>'
  );
}

function isLoopback(host: string): boolean {
  return LOOPBACK_HOSTS.has(host.toLowerCase());
}

export function validateUrl(
  raw: string,
  allowedHosts: Set<string>,
  kind: 'app_url' | 'base_url'
): string {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch (e) {
    throw new Error(`Invalid ${kind}: ${(e as Error).message}`, { cause: e });
  }
  // Reject sneaky URLs: `https://user:pass@evil.com`, `https://x?y`,
  // `https://x#y`. Node's URL parser will happily accept them all.
  if (parsed.username || parsed.password) {
    throw new Error(
      `Refusing ${kind} '${raw}': URLs with userinfo are not permitted.`
    );
  }
  if (parsed.search || parsed.hash) {
    throw new Error(
      `Refusing ${kind} '${raw}': query/fragment not permitted here.`
    );
  }
  // rstrip trailing dot from hostname so 'evil.compresr.ai.' can't slip past
  // the allow-list via DNS root-zone normalization.
  const host = (parsed.hostname || '').toLowerCase().replace(/\.+$/, '');
  if (isLoopback(host)) return raw;
  if (process.env[DEV_HOST_ALLOWED_ENV] === '1') return raw;
  if (parsed.protocol !== 'https:') {
    throw new Error(
      `Refusing ${kind} '${raw}': scheme must be https. ` +
        `Set ${DEV_HOST_ALLOWED_ENV}=1 to allow non-loopback overrides.`
    );
  }
  if (!allowedHosts.has(host)) {
    throw new Error(
      `Refusing ${kind} '${raw}': host '${host}' not in allow-list ` +
        `${JSON.stringify([...allowedHosts].sort())}. Set ${DEV_HOST_ALLOWED_ENV}=1 to override.`
    );
  }
  return raw;
}

function stateEqual(received: string | null | undefined, expected: string): boolean {
  if (received?.length !== expected.length) return false;
  try {
    return timingSafeEqual(Buffer.from(received), Buffer.from(expected));
  } catch {
    return false;
  }
}

const SECURITY_HEADERS = {
  'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'",
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
  'Cache-Control': 'no-store',
} as const;

interface CallbackResult {
  token: string | null;
  state: string | null;
  error: string | null;
}

function makeHandler(
  result: CallbackResult,
  expectedState: string,
  port: number,
  consumedRef: { value: boolean }
): (req: IncomingMessage, res: ServerResponse) => void {
  return (req, res) => {
    const url = new URL(req.url ?? '/', `http://127.0.0.1:${port}`);
    if (url.pathname !== '/cb') {
      respondText(res, 404, 'not found');
      return;
    }
    const host = req.headers.host ?? '';
    if (host !== `127.0.0.1:${port}`) {
      respondText(res, 403, 'forbidden');
      return;
    }
    const secFetchSite = req.headers['sec-fetch-site'];
    if (
      typeof secFetchSite === 'string' &&
      secFetchSite !== 'none' &&
      secFetchSite !== 'same-origin'
    ) {
      const mode = req.headers['sec-fetch-mode'];
      const dest = req.headers['sec-fetch-dest'];
      if (mode !== 'navigate' || dest !== 'document') {
        respondText(res, 403, 'forbidden');
        return;
      }
    }
    if (consumedRef.value) {
      respondText(res, 410, 'gone');
      return;
    }
    const state = url.searchParams.get('state');
    const error = url.searchParams.get('error');
    const token = url.searchParams.get('token');
    // State first — bad state must not set result.error, otherwise any tab
    // could DoS active logins with random /cb?state=x.
    if (state?.length !== STATE_LEN || !stateEqual(state, expectedState)) {
      respondHtml(res, 400, 'State parameter mismatch', [
        `<p>The response didn't match this login session — possibly a stale tab or a replayed URL.</p>`,
        `<p>Retry <kbd>compresr-sdk login</kbd> from your terminal.</p>`,
      ].join(''), false);
      return;
    }

    consumedRef.value = true;

    if (error) {
      const code = sanitizeErrorCode(error);
      result.error = code;
      const safe = escapeHtml(code);
      respondHtml(res, 400, 'Authorization failed', [
        `<p>The provider responded with <kbd>${safe}</kbd>.</p>`,
        `<p>Return to your terminal and retry <kbd>compresr-sdk login</kbd>.</p>`,
      ].join(''), false);
      return;
    }

    if (!token) {
      result.error = 'no_token';
      respondHtml(res, 400, 'No token received',
        '<p>The authorization completed but no token came back.</p>', false);
      return;
    }

    result.token = token;
    result.state = state;
    respondHtml(res, 200, "You're logged in", [
      '<p>Your API key was delivered to the SDK.</p>',
      '<p>You can close this tab and return to your terminal.</p>',
    ].join(''), true);
  };
}

function respondHtml(res: ServerResponse, status: number, title: string, body: string, ok: boolean): void {
  const html = landingHtml({ ok, title, body });
  writeResponse(res, status, html, 'text/html; charset=utf-8');
}

function respondText(res: ServerResponse, status: number, text: string): void {
  writeResponse(res, status, text, 'text/plain; charset=utf-8');
}

function writeResponse(res: ServerResponse, status: number, body: string, contentType: string): void {
  const buf = Buffer.from(body, 'utf-8');
  res.writeHead(status, {
    'Content-Type': contentType,
    'Content-Length': String(buf.length),
    ...SECURITY_HEADERS,
  });
  try { res.end(buf); } catch { /* connection dropped */ }
}

async function bindServer(
  handlerFactory: (port: number) => (req: IncomingMessage, res: ServerResponse) => void
): Promise<{ server: Server; port: number }> {
  let lastErr: unknown = null;
  for (let port = PORT_START; port <= PORT_END; port++) {
    const server = createServer(handlerFactory(port));
    try {
      await new Promise<void>((resolve, reject) => {
        const onError = (e: NodeJS.ErrnoException) => { server.off('listening', onListen); reject(e); };
        const onListen = () => { server.off('error', onError); resolve(); };
        server.once('error', onError);
        server.once('listening', onListen);
        server.listen(port, '127.0.0.1');
      });
      return { server, port };
    } catch (e) {
      lastErr = e;
      continue;
    }
  }
  throw new Error(
    `No free port in ${PORT_START}-${PORT_END}: ${(lastErr as Error)?.message ?? 'unknown'}`
  );
}

function defaultAppUrl(): string {
  const raw = process.env.COMPRESR_APP_URL;
  return raw && raw.length > 0 ? raw : 'https://compresr.ai';
}

function defaultBaseUrl(): string {
  const raw = process.env.COMPRESR_BASE_URL;
  return raw && raw.length > 0 ? raw : 'https://api.compresr.ai';
}

function stderr(msg: string): void {
  process.stderr.write(`${msg}\n`);
}

export interface LoginOptions {
  appUrl?: string;
  baseUrl?: string;
  profile?: string;
  timeoutMs?: number;
  openBrowserOnStart?: boolean;
}

export async function login(opts: LoginOptions = {}): Promise<string> {
  const timeoutMs = opts.timeoutMs ?? 120_000;
  if (!(timeoutMs > 0 && timeoutMs <= MAX_TIMEOUT_MS)) {
    throw new RangeError(
      `timeoutMs must be in (0, ${MAX_TIMEOUT_MS}] (got ${timeoutMs})`
    );
  }

  const rawApp = (opts.appUrl ?? defaultAppUrl()).replace(/\/+$/, '');
  const rawBase = (opts.baseUrl ?? defaultBaseUrl()).replace(/\/+$/, '');
  const appUrl = validateUrl(rawApp, ALLOWED_APP_HOSTS, 'app_url');
  const baseUrl = validateUrl(rawBase, ALLOWED_BASE_HOSTS, 'base_url');

  const state = randomBytes(32).toString('hex');
  const result: CallbackResult = { token: null, state: null, error: null };
  const consumedRef = { value: false };

  const { server, port } = await bindServer((p) =>
    makeHandler(result, state, p, consumedRef)
  );
  const callback = `http://127.0.0.1:${port}/cb`;
  const authUrl = `${appUrl}/authorize?${new URLSearchParams({ state, callback }).toString()}`;

  let aborted = false;
  const onSignal = () => {
    aborted = true;
    result.error = 'aborted';
  };
  process.once('SIGINT', onSignal);
  process.once('SIGTERM', onSignal);

  let browserOpened = false;
  if (opts.openBrowserOnStart !== false) {
    browserOpened = openBrowser(authUrl);
  }
  if (browserOpened) {
    stderr(`Opened browser to ${authUrl}`);
  } else {
    stderr('Could not open a browser automatically. Copy this URL and open it manually:');
    stderr(`    ${authUrl}`);
  }

  try {
    const deadline = Date.now() + timeoutMs;
    let lastProgress = Date.now();
    while (Date.now() < deadline) {
      if (result.token || result.error) break;
      const now = Date.now();
      if (now - lastProgress >= PROGRESS_INTERVAL_MS) {
        const remaining = Math.max(0, Math.round((deadline - now) / 1000));
        stderr(`Waiting for browser callback… (${remaining}s remaining, Ctrl-C to abort)`);
        lastProgress = now;
      }
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    }
  } finally {
    process.off('SIGINT', onSignal);
    process.off('SIGTERM', onSignal);
    await new Promise<void>((resolve) => {
      const s = server as Server & { closeAllConnections?: () => void };
      s.closeAllConnections?.();
      server.close(() => resolve());
    }).catch(() => undefined);
  }

  if (aborted) {
    const err = new Error('Aborted.');
    err.name = 'AbortError';
    throw err;
  }

  if (result.error) {
    const code = sanitizeErrorCode(result.error);
    throw new Error(`Login failed: ${code}`);
  }
  const token = result.token;
  result.token = null;
  if (!token) {
    const seconds = (timeoutMs / 1000).toFixed(0);
    throw new Error(`Login timed out after ${seconds}s. Retry \`compresr-sdk login\`.`);
  }

  const path = save(token, { profile: opts.profile ?? DEFAULT_PROFILE, baseUrl });
  process.stdout.write(
    `Saved credentials to ${path} (profile: ${opts.profile ?? DEFAULT_PROFILE})\n`
  );
  return token;
}

export function logout(profile: string = DEFAULT_PROFILE): boolean {
  return clear(profile);
}
