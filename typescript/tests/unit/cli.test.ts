/**
 * Unit tests for src/cli/index.ts — the compresr-sdk CLI.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { chmodSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { main } from '../../src/cli/index.js';
import { save } from '../../src/auth/credentials.js';

const isPosix = process.platform !== 'win32';

interface Capture {
  stdout: string[];
  stderr: string[];
  stop: () => void;
}

function capture(): Capture {
  const stdout: string[] = [];
  const stderr: string[] = [];
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  (process.stdout as unknown as { write: (b: string | Uint8Array) => boolean }).write = (
    b: string | Uint8Array
  ) => {
    stdout.push(typeof b === 'string' ? b : Buffer.from(b).toString('utf-8'));
    return true;
  };
  (process.stderr as unknown as { write: (b: string | Uint8Array) => boolean }).write = (
    b: string | Uint8Array
  ) => {
    stderr.push(typeof b === 'string' ? b : Buffer.from(b).toString('utf-8'));
    return true;
  };
  return {
    stdout,
    stderr,
    stop: () => {
      (process.stdout as unknown as { write: typeof origOut }).write = origOut;
      (process.stderr as unknown as { write: typeof origErr }).write = origErr;
    },
  };
}

describe('CLI', () => {
  let tmp: string;
  let credsFile: string;
  const originalEnv = { ...process.env };

  beforeEach(() => {
    tmp = join(tmpdir(), `compresr-cli-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    mkdirSync(tmp, { recursive: true });
    credsFile = join(tmp, 'creds.json');
    process.env.COMPRESR_CREDENTIALS_FILE = credsFile;
    delete process.env.COMPRESR_API_KEY;
    delete process.env.COMPRESR_PROFILE;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    rmSync(tmp, { recursive: true, force: true });
  });

  describe('--version', () => {
    it('prints compresr-sdk prefix', async () => {
      const cap = capture();
      try {
        const rc = await main(['--version']);
        expect(rc).toBe(0);
      } finally { cap.stop(); }
      expect(cap.stdout.join('')).toMatch(/^compresr-sdk /);
    });
  });

  describe('help', () => {
    it('no args → prints help + exit 1', async () => {
      const cap = capture();
      let rc: number;
      try { rc = await main([]); } finally { cap.stop(); }
      expect(rc).toBe(1);
      expect(cap.stdout.join('')).toMatch(/login/);
    });

    it('--help → prints help + exit 0', async () => {
      const cap = capture();
      let rc: number;
      try { rc = await main(['--help']); } finally { cap.stop(); }
      expect(rc).toBe(0);
    });
  });

  describe('whoami', () => {
    it('exit 1 with stderr message when not logged in', async () => {
      const cap = capture();
      let rc: number;
      try { rc = await main(['whoami']); } finally { cap.stop(); }
      expect(rc).toBe(1);
      expect(cap.stderr.join('')).toMatch(/Not logged in/);
      expect(cap.stdout.join('')).toBe('');
    });

    it('exit 0 with key preview when logged in', async () => {
      save('cmp_abcdefgh_test_XYZW', { profile: 'default' });
      const cap = capture();
      let rc: number;
      try { rc = await main(['whoami']); } finally { cap.stop(); }
      expect(rc).toBe(0);
      const stdout = cap.stdout.join('');
      expect(stdout).toMatch(/cmp_abcd/);
      expect(stdout).toMatch(/XYZW/);
      expect(cap.stderr.join('')).toBe('');
    });

    it.skipIf(!isPosix)('exit 1 with stderr when bad perms', async () => {
      save('cmp_x_test_key_1234X', { profile: 'default' });
      chmodSync(credsFile, 0o644);
      const cap = capture();
      let rc: number;
      try { rc = await main(['whoami']); } finally { cap.stop(); }
      expect(rc).toBe(1);
      expect(cap.stderr.join('')).toMatch(/group\/world/);
    });
  });

  describe('logout', () => {
    it('removes and reports success', async () => {
      save('cmp_x_test_key_1234X', { profile: 'default' });
      const cap = capture();
      let rc: number;
      try { rc = await main(['logout']); } finally { cap.stop(); }
      expect(rc).toBe(0);
      expect(cap.stdout.join('')).toMatch(/Removed credentials/);
    });

    it('idempotent when nothing to remove', async () => {
      const cap = capture();
      let rc: number;
      try { rc = await main(['logout']); } finally { cap.stop(); }
      expect(rc).toBe(0);
    });
  });

  describe('status', () => {
    it('reports missing file', async () => {
      const cap = capture();
      let rc: number;
      try { rc = await main(['status']); } finally { cap.stop(); }
      expect(rc).toBe(0);
      expect(cap.stdout.join('')).toMatch(/exists:\s+false/);
    });

    it('enumerates profiles', async () => {
      save('cmp_a_test_key_1234X', { profile: 'default' });
      save('cmp_b_test_key_1234X', { profile: 'work' });
      const cap = capture();
      let rc: number;
      try { rc = await main(['status']); } finally { cap.stop(); }
      expect(rc).toBe(0);
      expect(cap.stdout.join('')).toMatch(/default/);
      expect(cap.stdout.join('')).toMatch(/work/);
    });
  });

  describe('unknown command', () => {
    it('exit 2 with stderr message', async () => {
      const cap = capture();
      let rc: number;
      try { rc = await main(['flurb']); } finally { cap.stop(); }
      expect(rc).toBe(2);
      expect(cap.stderr.join('')).toMatch(/unknown command/);
    });
  });

  describe('login error routing', () => {
    it('surfaces app_url validation errors to stderr with exit 1', async () => {
      const cap = capture();
      let rc: number;
      try {
        rc = await main([
          'login',
          '--app-url', 'https://evil.example',
          '--base-url', 'http://127.0.0.1:8001',
          '--timeout', '0.1',
        ]);
      } finally { cap.stop(); }
      expect(rc).toBe(1);
      expect(cap.stderr.join('')).toMatch(/not in allow-list/);
    });

    it('unknown flag → exit 1 with stderr', async () => {
      const cap = capture();
      let rc: number;
      try { rc = await main(['login', '--nope']); } finally { cap.stop(); }
      expect(rc).toBe(1);
      expect(cap.stderr.join('')).toMatch(/unknown flag/);
    });
  });
});
