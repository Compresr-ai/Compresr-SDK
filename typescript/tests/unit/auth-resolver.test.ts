/**
 * Unit tests for src/auth/resolver.ts — explicit > env > file precedence.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { chmodSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { save } from '../../src/auth/credentials.js';
import { resolveApiKey } from '../../src/auth/resolver.js';

const isPosix = process.platform !== 'win32';

describe('resolveApiKey', () => {
  let tmp: string;
  let credsFile: string;
  const originalEnv = { ...process.env };

  beforeEach(() => {
    tmp = join(tmpdir(), `compresr-res-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`);
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

  it('explicit wins over env + file', () => {
    process.env.COMPRESR_API_KEY = 'cmp_env_test_key_1234';
    save('cmp_file_test_key_1234', { profile: 'default' });
    expect(resolveApiKey('cmp_explicit_test_key_1234')).toBe('cmp_explicit_test_key_1234');
  });

  it('explicit empty string short-circuits (returns empty)', () => {
    process.env.COMPRESR_API_KEY = 'cmp_env_test_key_1234';
    save('cmp_file_test_key_1234', { profile: 'default' });
    // Downstream HttpClient will reject the empty; the resolver just
    // preserves the user's explicit intent.
    expect(resolveApiKey('')).toBe('');
  });

  it('env used when no explicit', () => {
    process.env.COMPRESR_API_KEY = 'cmp_env_test_key_1234';
    save('cmp_file_test_key_1234', { profile: 'default' });
    expect(resolveApiKey(undefined)).toBe('cmp_env_test_key_1234');
  });

  it('file used when no explicit and no env', () => {
    save('cmp_file_test_key_1234', { profile: 'default' });
    expect(resolveApiKey(undefined)).toBe('cmp_file_test_key_1234');
  });

  it('returns null when nothing is configured', () => {
    expect(resolveApiKey(undefined)).toBeNull();
  });

  it('COMPRESR_PROFILE selects the section', () => {
    save('cmp_a_test_key_1234X', { profile: 'default' });
    save('cmp_b_test_key_1234X', { profile: 'work' });
    process.env.COMPRESR_PROFILE = 'work';
    expect(resolveApiKey(undefined)).toBe('cmp_b_test_key_1234X');
  });

  it.skipIf(!isPosix)('returns null on bad-perms file rather than throwing', () => {
    save('cmp_a_test_key_1234X', { profile: 'default' });
    chmodSync(credsFile, 0o644);
    expect(resolveApiKey(undefined)).toBeNull();
  });
});
