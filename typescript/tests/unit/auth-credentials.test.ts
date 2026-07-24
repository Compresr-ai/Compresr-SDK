/**
 * Unit tests for src/auth/credentials.ts — the ~/.compresr/credentials.json store.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { chmodSync, mkdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  DEFAULT_PROFILE,
  BadPermissionsError,
  credentialsPath,
  load,
  save,
  clear,
  listProfiles,
} from '../../src/auth/credentials.js';

const isPosix = process.platform !== 'win32';

function makeTmpDir(): string {
  const dir = join(tmpdir(), `compresr-auth-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

describe('credentials', () => {
  let tmp: string;
  let credsFile: string;
  const originalEnv = { ...process.env };

  beforeEach(() => {
    tmp = makeTmpDir();
    credsFile = join(tmp, 'creds.json');
    process.env.COMPRESR_CREDENTIALS_FILE = credsFile;
    delete process.env.COMPRESR_API_KEY;
    delete process.env.COMPRESR_PROFILE;
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    rmSync(tmp, { recursive: true, force: true });
  });

  describe('save + load', () => {
    it('roundtrips the api key', () => {
      save('cmp_abc123_test_key_1234', { profile: 'default', baseUrl: 'https://api.x' });
      expect(load('default')).toBe('cmp_abc123_test_key_1234');
    });

    it('returns null for missing file', () => {
      expect(load('default')).toBeNull();
    });

    it('returns null for missing profile', () => {
      save('cmp_a_test_key_1234X', { profile: 'work' });
      expect(load('default')).toBeNull();
      expect(load('work')).toBe('cmp_a_test_key_1234X');
    });

    it('overwrites the same profile', () => {
      save('cmp_old_test_key_1234', { profile: 'default' });
      save('cmp_new_test_key_1234', { profile: 'default' });
      expect(load('default')).toBe('cmp_new_test_key_1234');
    });

    it('keeps multiple profiles', () => {
      save('cmp_a_test_key_1234X', { profile: 'default' });
      save('cmp_b_test_key_1234X', { profile: 'work' });
      expect(load('default')).toBe('cmp_a_test_key_1234X');
      expect(load('work')).toBe('cmp_b_test_key_1234X');
    });

    it.skipIf(!isPosix)('sets 0600 mode on POSIX', () => {
      save('cmp_perm_test_key_1234', { profile: 'default' });
      const mode = statSync(credsFile).mode & 0o777;
      expect(mode).toBe(0o600);
    });

    it.skipIf(!isPosix)('load raises on world-readable file', () => {
      save('cmp_a_test_key_1234X', { profile: 'default' });
      chmodSync(credsFile, 0o644);
      expect(() => load('default')).toThrow(BadPermissionsError);
    });

    it('load returns null for corrupt JSON', () => {
      writeFileSync(credsFile, 'this is not { valid json', { mode: 0o600 });
      expect(load('default')).toBeNull();
    });

    it('save recovers from corrupt file and warns', () => {
      writeFileSync(credsFile, 'not json', { mode: 0o600 });
      const warnings: string[] = [];
      const orig = process.stderr.write.bind(process.stderr);
      // Vitest v-typed spy signature is fussy for process.stderr; capture directly.
      (process.stderr as unknown as { write: (b: string | Uint8Array) => boolean }).write = (
        b: string | Uint8Array
      ) => {
        warnings.push(typeof b === 'string' ? b : Buffer.from(b).toString('utf-8'));
        return true;
      };
      try {
        save('cmp_recovered_test_key_1234', { profile: 'default' });
      } finally {
        (process.stderr as unknown as { write: (b: string | Uint8Array) => boolean }).write = orig;
      }
      expect(load('default')).toBe('cmp_recovered_test_key_1234');
      expect(warnings.join('')).toMatch(/corrupt/);
    });
  });

  describe('clear', () => {
    it('removes the profile', () => {
      save('cmp_x_test_key_1234X', { profile: 'default' });
      expect(clear('default')).toBe(true);
      expect(load('default')).toBeNull();
    });

    it.skipIf(!isPosix)('preserves 0600 after clear', () => {
      save('cmp_a_test_key_1234X', { profile: 'default' });
      save('cmp_b_test_key_1234X', { profile: 'work' });
      clear('default');
      const mode = statSync(credsFile).mode & 0o777;
      expect(mode).toBe(0o600);
    });

    it('leaves other profiles intact', () => {
      save('cmp_a_test_key_1234X', { profile: 'default' });
      save('cmp_b_test_key_1234X', { profile: 'work' });
      clear('default');
      expect(load('work')).toBe('cmp_b_test_key_1234X');
    });

    it('returns false when missing', () => {
      expect(clear('nothing')).toBe(false);
    });

    it('returns false when file missing', () => {
      expect(clear('default')).toBe(false);
    });
  });

  describe('listProfiles', () => {
    it('returns empty for missing file', () => {
      expect(listProfiles()).toEqual([]);
    });

    it('returns saved profile names', () => {
      save('cmp_a_test_key_1234X', { profile: 'default' });
      save('cmp_b_test_key_1234X', { profile: 'work' });
      expect(listProfiles().sort()).toEqual(['default', 'work']);
    });
  });

  describe('credentialsPath', () => {
    it('honors the env override', () => {
      expect(credentialsPath()).toBe(credsFile);
    });

    it('exports DEFAULT_PROFILE', () => {
      expect(DEFAULT_PROFILE).toBe('default');
    });
  });
});
