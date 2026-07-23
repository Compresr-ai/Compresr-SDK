/**
 * Unit tests for src/auth/login.ts — browser flow, URL validation, and the
 * pure helpers (escapeHtml, sanitizeErrorCode, validateUrl, landingHtml).
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  escapeHtml,
  sanitizeErrorCode,
  validateUrl,
  login,
  logout,
} from '../../src/auth/login.js';
import { save, load } from '../../src/auth/credentials.js';

const ALLOWED_APP = new Set(['compresr.ai', 'staging.compresr.ai']);
const ALLOWED_BASE = new Set(['api.compresr.ai']);

describe('escapeHtml', () => {
  it('escapes the five reflected-XSS metacharacters', () => {
    expect(escapeHtml("<script>alert('x')</script>&\"y")).toBe(
      '&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;&amp;&quot;y'
    );
  });

  it('is idempotent for safe strings', () => {
    expect(escapeHtml('access_denied')).toBe('access_denied');
  });
});

describe('sanitizeErrorCode', () => {
  it('passes through whitelisted codes', () => {
    expect(sanitizeErrorCode('access_denied')).toBe('access_denied');
  });

  it('passes through short alphanum', () => {
    expect(sanitizeErrorCode('custom_code_9')).toBe('custom_code_9');
  });

  it('replaces dangerous strings', () => {
    expect(sanitizeErrorCode('<script>alert(1)</script>')).toBe('unknown_error');
  });

  it('replaces overly long strings', () => {
    expect(sanitizeErrorCode('a'.repeat(100))).toBe('unknown_error');
  });

  it('replaces null/empty', () => {
    expect(sanitizeErrorCode(null)).toBe('unknown_error');
    expect(sanitizeErrorCode(undefined)).toBe('unknown_error');
    expect(sanitizeErrorCode('')).toBe('unknown_error');
  });
});

describe('validateUrl', () => {
  const originalEnv = { ...process.env };
  afterEach(() => { process.env = { ...originalEnv }; });

  it('accepts loopback with any scheme', () => {
    expect(validateUrl('http://127.0.0.1:8000', ALLOWED_APP, 'app_url')).toBe('http://127.0.0.1:8000');
    expect(validateUrl('http://localhost:8000', ALLOWED_APP, 'app_url')).toBe('http://localhost:8000');
  });

  it('accepts allow-listed HTTPS', () => {
    expect(validateUrl('https://compresr.ai', ALLOWED_APP, 'app_url')).toBe('https://compresr.ai');
  });

  it('rejects non-loopback HTTP', () => {
    expect(() => validateUrl('http://evil.example', ALLOWED_APP, 'app_url')).toThrow(/scheme must be https/);
  });

  it('rejects non-allowlisted HTTPS', () => {
    expect(() => validateUrl('https://evil.example', ALLOWED_APP, 'app_url')).toThrow(/not in allow-list/);
  });

  it('COMPRESR_ALLOW_LOCAL_URLS=1 bypasses the allow-list', () => {
    process.env.COMPRESR_ALLOW_LOCAL_URLS = '1';
    expect(validateUrl('https://staging-preview.example', ALLOWED_APP, 'app_url')).toBe(
      'https://staging-preview.example'
    );
  });
});

describe('login (browser flow)', () => {
  let tmp: string;
  let credsFile: string;
  const originalEnv = { ...process.env };

  beforeEach(() => {
    tmp = join(tmpdir(), `compresr-login-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(tmp, { recursive: true });
    credsFile = join(tmp, 'creds.json');
    process.env.COMPRESR_CREDENTIALS_FILE = credsFile;
    delete process.env.COMPRESR_API_KEY;
    delete process.env.COMPRESR_APP_URL;
    delete process.env.COMPRESR_BASE_URL;
    delete process.env.COMPRESR_ALLOW_LOCAL_URLS;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    rmSync(tmp, { recursive: true, force: true });
  });

  it('rejects non-https app_url', async () => {
    await expect(
      login({
        appUrl: 'http://evil.example',
        baseUrl: 'http://127.0.0.1:8001',
        timeoutMs: 500,
        openBrowserOnStart: false,
      })
    ).rejects.toThrow(/scheme must be https/);
  });

  it('rejects off-allowlist HTTPS app_url', async () => {
    await expect(
      login({
        appUrl: 'https://evil.example',
        baseUrl: 'http://127.0.0.1:8001',
        timeoutMs: 500,
        openBrowserOnStart: false,
      })
    ).rejects.toThrow(/not in allow-list/);
  });

  it('rejects invalid timeoutMs', async () => {
    await expect(login({ timeoutMs: 0, openBrowserOnStart: false })).rejects.toThrow(/timeoutMs/);
    await expect(login({ timeoutMs: 10_000_000, openBrowserOnStart: false })).rejects.toThrow(/timeoutMs/);
  });

  it('times out cleanly when no callback arrives', async () => {
    await expect(
      login({
        appUrl: 'http://127.0.0.1:12345',
        baseUrl: 'http://127.0.0.1:54321',
        timeoutMs: 500,
        openBrowserOnStart: false,
      })
    ).rejects.toThrow(/timed out/);
  });

  it('happy path: correct state → token → key saved', async () => {
    // Kick login off, then drive the callback directly.
    let started: { authUrl?: string } = {};
    const captureStderr = intercept();
    const loginPromise = login({
      appUrl: 'http://127.0.0.1:12345',
      baseUrl: 'http://127.0.0.1:54321',
      timeoutMs: 5000,
      openBrowserOnStart: false,
    });
    // Grab the auth URL from stderr — we wrote it there.
    started = await waitForAuthUrl(captureStderr);
    const { port, state } = parseAuthUrl(started.authUrl!);
    const status = await hitCallback(port, { token: 'cmp_test_key_test_key_1234', state });
    expect(status).toBe(200);
    const key = await loginPromise;
    expect(key).toBe('cmp_test_key_test_key_1234');
    expect(load('default')).toBe('cmp_test_key_test_key_1234');
    captureStderr.stop();
  });

  it('bad state does NOT abort the flow', async () => {
    const captureStderr = intercept();
    const loginPromise = login({
      appUrl: 'http://127.0.0.1:12345',
      baseUrl: 'http://127.0.0.1:54321',
      timeoutMs: 1200,
      openBrowserOnStart: false,
    });
    const started = await waitForAuthUrl(captureStderr);
    const { port } = parseAuthUrl(started.authUrl!);
    // Wrong-length AND wrong-value states should be silently rejected.
    await hitCallback(port, { token: 'cmp_should_not_stick', state: 'a'.repeat(64) });
    await hitCallback(port, { token: 'cmp_should_not_stick', state: 'nope' });
    await expect(loginPromise).rejects.toThrow(/timed out/);
    expect(load('default')).toBeNull();
    captureStderr.stop();
  });

  it('correct state + server error → RuntimeError with sanitized code', async () => {
    const captureStderr = intercept();
    const loginPromise = login({
      appUrl: 'http://127.0.0.1:12345',
      baseUrl: 'http://127.0.0.1:54321',
      timeoutMs: 5000,
      openBrowserOnStart: false,
    });
    const started = await waitForAuthUrl(captureStderr);
    const { port, state } = parseAuthUrl(started.authUrl!);
    await hitCallback(port, { state, error: '<script>alert(1)</script>' });
    await expect(loginPromise).rejects.toThrow(/unknown_error/);
    captureStderr.stop();
  });

  it('non-/cb path is silently 404 without disturbing state', async () => {
    const captureStderr = intercept();
    const loginPromise = login({
      appUrl: 'http://127.0.0.1:12345',
      baseUrl: 'http://127.0.0.1:54321',
      timeoutMs: 1200,
      openBrowserOnStart: false,
    });
    const started = await waitForAuthUrl(captureStderr);
    const { port } = parseAuthUrl(started.authUrl!);
    const status = await rawGet(`http://127.0.0.1:${port}/favicon.ico`);
    expect(status).toBe(404);
    await expect(loginPromise).rejects.toThrow(/timed out/);
    captureStderr.stop();
  });

  it('rejects cross-site request via Sec-Fetch-Site', async () => {
    const captureStderr = intercept();
    const loginPromise = login({
      appUrl: 'http://127.0.0.1:12345',
      baseUrl: 'http://127.0.0.1:54321',
      timeoutMs: 1200,
      openBrowserOnStart: false,
    });
    const started = await waitForAuthUrl(captureStderr);
    const { port, state } = parseAuthUrl(started.authUrl!);
    const status = await hitCallback(port, {
      state,
      token: 'cmp_hostile_test_key_1234',
      headers: { 'sec-fetch-site': 'cross-site' },
    });
    expect(status).toBe(403);
    await expect(loginPromise).rejects.toThrow(/timed out/);
    captureStderr.stop();
  });
});

describe('logout', () => {
  let tmp: string;
  let credsFile: string;
  const originalEnv = { ...process.env };

  beforeEach(() => {
    tmp = join(tmpdir(), `compresr-logout-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(tmp, { recursive: true });
    credsFile = join(tmp, 'creds.json');
    process.env.COMPRESR_CREDENTIALS_FILE = credsFile;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    rmSync(tmp, { recursive: true, force: true });
  });

  it('removes a stored profile', () => {
    save('cmp_x_test_key_1234X', { profile: 'default' });
    expect(logout('default')).toBe(true);
    expect(load('default')).toBeNull();
  });

  it('returns false when no profile exists', () => {
    expect(logout('missing')).toBe(false);
  });
});

// ---------- test helpers ----------

interface Intercepted { lines: string[]; stop: () => void; }
function intercept(): Intercepted {
  const lines: string[] = [];
  const orig = process.stderr.write.bind(process.stderr);
  (process.stderr as unknown as { write: (b: string | Uint8Array) => boolean }).write = (
    b: string | Uint8Array
  ) => {
    lines.push(typeof b === 'string' ? b : Buffer.from(b).toString('utf-8'));
    return true;
  };
  return {
    lines,
    stop: () => {
      (process.stderr as unknown as { write: typeof orig }).write = orig;
    },
  };
}

async function waitForAuthUrl(cap: Intercepted, timeoutMs = 3000): Promise<{ authUrl: string }> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    for (const line of cap.lines) {
      const m = line.match(/https?:\/\/[^\s]+\/authorize\?[^\s]+/);
      if (m) return { authUrl: m[0] };
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error('login() never printed an auth URL');
}

function parseAuthUrl(authUrl: string): { port: number; state: string } {
  const u = new URL(authUrl);
  const state = u.searchParams.get('state') ?? '';
  const callback = u.searchParams.get('callback') ?? '';
  const cbUrl = new URL(callback);
  return { port: Number.parseInt(cbUrl.port, 10), state };
}

interface CallbackParams { token?: string; state?: string; error?: string; headers?: Record<string, string>; }

async function hitCallback(port: number, p: CallbackParams): Promise<number> {
  const q = new URLSearchParams();
  if (p.token) q.set('token', p.token);
  if (p.state) q.set('state', p.state);
  if (p.error) q.set('error', p.error);
  const res = await fetch(`http://127.0.0.1:${port}/cb?${q.toString()}`, {
    headers: p.headers,
  });
  return res.status;
}

async function rawGet(url: string): Promise<number> {
  const res = await fetch(url);
  return res.status;
}
