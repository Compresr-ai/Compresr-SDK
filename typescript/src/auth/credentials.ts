/**
 * Persistent credentials store at ``~/.compresr/credentials.json``. Node-only.
 */

import { randomBytes, randomUUID } from 'node:crypto';
import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  existsSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeSync,
} from 'node:fs';
import { hostname, homedir } from 'node:os';
import { dirname, join } from 'node:path';

export const DEFAULT_PROFILE = 'default';

const REPLACE_RETRIES = 6;
const REPLACE_BACKOFF_MS = 100;
const LOCK_ATTEMPTS = 60;
const LOCK_BACKOFF_MS = 100;
const LOCK_STALE_MS = 30_000;

const S_IFLNK = 0o120000;
const S_IFMT = 0o170000;
const S_IRWXG = 0o070;
const S_IRWXO = 0o007;

const POSIX_TMP_FLAGS =
  fsConstants.O_WRONLY |
  fsConstants.O_CREAT |
  fsConstants.O_EXCL |
  (fsConstants.O_NOFOLLOW ?? 0);

export const API_KEY_RE = /^cmp_[A-Za-z0-9_-]{16,128}$/;

export interface CredentialsSection {
  api_key: string;
  saved_at: string;
  base_url?: string;
  account_type?: string;
}

interface CredentialsFile {
  [profile: string]: CredentialsSection;
}

export class BadPermissionsError extends Error {
  constructor(public readonly path: string) {
    super(
      `Refusing to read ${path}: file is group/world accessible. ` +
        'Run `chmod 600` on it.'
    );
    this.name = 'BadPermissionsError';
  }
}

export class LockAcquisitionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LockAcquisitionError';
  }
}

export function credentialsPath(): string {
  const override = process.env.COMPRESR_CREDENTIALS_FILE;
  if (override) {
    return override.startsWith('~')
      ? join(homedir(), override.slice(1))
      : override;
  }
  return join(homedir(), '.compresr', 'credentials.json');
}

function isPosix(): boolean {
  return process.platform !== 'win32';
}

function isSymlink(path: string): boolean {
  try {
    return (lstatSync(path).mode & S_IFMT) === S_IFLNK;
  } catch {
    return false;
  }
}

function assertNotSymlink(path: string, kind: string): void {
  if (isSymlink(path)) {
    throw new BadPermissionsError(
      `Refusing to use ${kind} at ${path}: symlink not permitted.`
    );
  }
}

function assertSafePerms(path: string): void {
  if (!isPosix()) return;
  if (lstatSync(path).mode & (S_IRWXG | S_IRWXO)) {
    throw new BadPermissionsError(path);
  }
}

function isSection(v: unknown): v is CredentialsSection {
  if (typeof v !== 'object' || v === null) return false;
  const o = v as Record<string, unknown>;
  return typeof o.api_key === 'string' && typeof o.saved_at === 'string';
}

function readFile(): CredentialsFile {
  const path = credentialsPath();
  if (!existsSync(path)) return {};
  assertNotSymlink(path, 'credentials file');
  assertSafePerms(path);
  const raw = readFileSync(path, 'utf-8');
  if (raw.trim() === '') return {};
  const parsed: unknown = JSON.parse(raw);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new SyntaxError('credentials file is not a JSON object');
  }
  for (const [profile, section] of Object.entries(parsed as Record<string, unknown>)) {
    if (!isSection(section)) {
      throw new SyntaxError(
        `credentials profile '${profile}' is malformed (missing api_key or saved_at)`
      );
    }
  }
  return parsed as CredentialsFile;
}

function safeReadOrFresh(): CredentialsFile {
  try {
    return readFile();
  } catch (e) {
    if (e instanceof BadPermissionsError) throw e;
    if (e instanceof SyntaxError) {
      process.stderr.write(
        `warning: credentials file at ${credentialsPath()} is corrupt ` +
          `(${e.message}); recreating.\n`
      );
      return {};
    }
    throw e;
  }
}

function sleep(ms: number): void {
  try {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
  } catch {
    const end = Date.now() + Math.min(ms, 50);
    while (Date.now() < end) {
      /* spin */
    }
  }
}

interface LockOwner {
  pid: number;
  uuid: string;
  hostname: string;
  ts: number;
}

function isLockOwnerLive(owner: LockOwner): boolean {
  if (owner.hostname !== hostname()) return true;
  try {
    process.kill(owner.pid, 0);
    return true;
  } catch {
    return false;
  }
}

function readLockOwner(lockPath: string): LockOwner | null {
  try {
    const parsed: unknown = JSON.parse(readFileSync(lockPath, 'utf-8'));
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      typeof (parsed as LockOwner).pid !== 'number' ||
      typeof (parsed as LockOwner).uuid !== 'string' ||
      typeof (parsed as LockOwner).hostname !== 'string'
    ) return null;
    return parsed as LockOwner;
  } catch {
    return null;
  }
}

function tryStealLock(lockPath: string, existingUuid: string): boolean {
  // CAS on the uuid — protects against a race with the original owner
  // refreshing the lock between our checks.
  const current = readLockOwner(lockPath);
  if (current?.uuid !== existingUuid) return false;
  try {
    unlinkSync(lockPath);
    return true;
  } catch {
    return false;
  }
}

function acquireLock<T>(path: string, work: () => T): T {
  const lockPath = `${path}.lock`;
  mkdirSync(dirname(lockPath), { recursive: true });

  const owner: LockOwner = {
    pid: process.pid,
    uuid: randomUUID(),
    hostname: hostname(),
    ts: Date.now(),
  };
  const ownerJson = JSON.stringify(owner);

  let held = false;
  for (let attempt = 0; attempt < LOCK_ATTEMPTS; attempt++) {
    try {
      const fd = openSync(lockPath, 'wx', 0o600);
      try {
        writeSync(fd, ownerJson);
      } finally {
        closeSync(fd);
      }
      held = true;
      break;
    } catch (e) {
      const err = e as NodeJS.ErrnoException;
      if (err.code !== 'EEXIST') throw e;
      const existing = readLockOwner(lockPath);
      const age = existing ? Date.now() - existing.ts : Number.MAX_SAFE_INTEGER;
      if (existing && !isLockOwnerLive(existing) && age > 1_000) {
        tryStealLock(lockPath, existing.uuid);
        continue;
      }
      if (existing && age > LOCK_STALE_MS) {
        tryStealLock(lockPath, existing.uuid);
        continue;
      }
      sleep(LOCK_BACKOFF_MS);
    }
  }

  if (!held) {
    throw new LockAcquisitionError(
      `Could not acquire ${lockPath} within ` +
        `${((LOCK_ATTEMPTS * LOCK_BACKOFF_MS) / 1000).toFixed(1)}s — ` +
        `another process may be writing, or a previous run left the lock stale.`
    );
  }

  try {
    return work();
  } finally {
    try {
      const current = readLockOwner(lockPath);
      if (!current || current.uuid === owner.uuid) {
        unlinkSync(lockPath);
      }
    } catch {
      /* ignore */
    }
  }
}

function fsyncDirectory(dir: string): void {
  if (!isPosix()) return;
  try {
    const fd = openSync(dir, 0);
    try {
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
  } catch {
    /* ENOTSUP on some FSes */
  }
}

function atomicWrite(data: CredentialsFile, path: string): void {
  const dir = dirname(path);
  mkdirSync(dir, { recursive: true });
  const defaultDir = join(homedir(), '.compresr');
  if (isPosix() && dir === defaultDir) {
    try { chmodSync(dir, 0o700); } catch { /* not fatal */ }
  }

  const tmpPath = join(
    dir,
    `.credentials-${process.pid}-${randomBytes(8).toString('hex')}`
  );
  const openFlag = isPosix() ? POSIX_TMP_FLAGS : 'wx';
  let written = false;
  let replaced = false;

  try {
    const fd = openSync(tmpPath, openFlag, 0o600);
    try {
      writeSync(fd, JSON.stringify(data, null, 2) + '\n');
      fsyncSync(fd);
      written = true;
    } finally {
      closeSync(fd);
    }
    if (isPosix()) {
      try { chmodSync(tmpPath, 0o600); } catch { /* not fatal */ }
    }

    if (isSymlink(path)) {
      throw new BadPermissionsError(
        `Refusing to replace symlink at ${path}; remove it first.`
      );
    }

    let lastErr: unknown = null;
    for (let attempt = 0; attempt < REPLACE_RETRIES; attempt++) {
      try {
        renameSync(tmpPath, path);
        replaced = true;
        break;
      } catch (e) {
        lastErr = e;
        sleep(REPLACE_BACKOFF_MS * 2 ** attempt);
      }
    }
    if (!replaced) throw lastErr;

    fsyncDirectory(dir);
  } finally {
    if (written && !replaced) {
      try { unlinkSync(tmpPath); } catch { /* ignore */ }
    }
  }
}

export function load(profile: string = DEFAULT_PROFILE): string | null {
  let data: CredentialsFile;
  try {
    data = readFile();
  } catch (e) {
    if (e instanceof BadPermissionsError) throw e;
    return null;
  }
  return data[profile]?.api_key || null;
}

export interface SaveOptions {
  profile?: string;
  baseUrl?: string;
  accountType?: string;
}

export function save(apiKey: string, opts: SaveOptions = {}): string {
  if (!API_KEY_RE.test(apiKey)) {
    throw new Error(
      'Refusing to save malformed api key: must match ^cmp_[A-Za-z0-9_-]{16,128}$.'
    );
  }
  const profile = opts.profile ?? DEFAULT_PROFILE;
  const path = credentialsPath();
  acquireLock(path, () => {
    const data = existsSync(path) ? safeReadOrFresh() : {};
    const section: CredentialsSection = {
      api_key: apiKey,
      saved_at: new Date().toISOString(),
    };
    if (opts.baseUrl) section.base_url = opts.baseUrl;
    if (opts.accountType) section.account_type = opts.accountType;
    data[profile] = section;
    atomicWrite(data, path);
  });
  return path;
}

export function clear(profile: string = DEFAULT_PROFILE): boolean {
  const path = credentialsPath();
  if (!existsSync(path)) return false;
  let removed = false;
  acquireLock(path, () => {
    let data: CredentialsFile;
    try {
      data = readFile();
    } catch (e) {
      if (e instanceof BadPermissionsError) throw e;
      return;
    }
    if (!(profile in data)) return;
    delete data[profile];
    atomicWrite(data, path);
    removed = true;
  });
  return removed;
}

export function listProfiles(): string[] {
  try {
    return Object.keys(readFile());
  } catch (e) {
    if (e instanceof BadPermissionsError) throw e;
    return [];
  }
}
