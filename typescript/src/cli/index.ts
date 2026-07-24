/** ``compresr-sdk`` CLI: login, logout, whoami, status. */

import { existsSync } from 'node:fs';

import { SDK_VERSION } from '../version.js';
import {
  BadPermissionsError,
  DEFAULT_PROFILE,
  credentialsPath,
  listProfiles,
  load,
  LockAcquisitionError,
} from '../auth/credentials.js';
import { login, logout } from '../auth/login.js';

interface CommonFlags {
  profile: string;
}
interface LoginFlags extends CommonFlags {
  appUrl?: string;
  baseUrl?: string;
  timeoutMs: number;
  openBrowserOnStart: boolean;
}

function err(msg: string): void {
  process.stderr.write(`${msg}\n`);
}

function out(msg: string): void {
  process.stdout.write(`${msg}\n`);
}

function parseCommonFlags(argv: string[]): CommonFlags {
  const flags: CommonFlags = { profile: DEFAULT_PROFILE };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--profile') {
      flags.profile = argv[++i] ?? DEFAULT_PROFILE;
    } else if (a?.startsWith('--')) {
      throw new Error(`unknown flag: ${a}`);
    } else {
      throw new Error(`unexpected argument: ${a}`);
    }
  }
  return flags;
}

function parseLoginFlags(argv: string[]): LoginFlags {
  const flags: LoginFlags = {
    profile: DEFAULT_PROFILE,
    timeoutMs: 120_000,
    openBrowserOnStart: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case '--profile': flags.profile = argv[++i] ?? DEFAULT_PROFILE; break;
      case '--app-url': flags.appUrl = argv[++i]; break;
      case '--base-url': flags.baseUrl = argv[++i]; break;
      case '--timeout': {
        const raw = argv[++i];
        const parsed = Number.parseFloat(raw ?? '');
        if (!Number.isFinite(parsed)) {
          throw new Error(`--timeout expects a number, got ${JSON.stringify(raw)}`);
        }
        flags.timeoutMs = Math.round(parsed * 1000);
        break;
      }
      case '--no-browser': flags.openBrowserOnStart = false; break;
      default:
        throw new Error(`unknown flag: ${a}`);
    }
  }
  return flags;
}

async function cmdLogin(argv: string[]): Promise<number> {
  let flags: LoginFlags;
  try {
    flags = parseLoginFlags(argv);
  } catch (e) {
    err(`error: ${(e as Error).message}`);
    return 1;
  }
  try {
    await login({
      appUrl: flags.appUrl,
      baseUrl: flags.baseUrl,
      profile: flags.profile,
      timeoutMs: flags.timeoutMs,
      openBrowserOnStart: flags.openBrowserOnStart,
    });
    return 0;
  } catch (e) {
    err(`error: ${(e as Error).message}`);
    if (e instanceof BadPermissionsError) {
      err('hint: run `chmod 600 ~/.compresr/credentials.json` and retry.');
    }
    return 1;
  }
}

function cmdLogout(argv: string[]): number {
  let flags: CommonFlags;
  try {
    flags = parseCommonFlags(argv);
  } catch (e) {
    err(`error: ${(e as Error).message}`);
    return 1;
  }
  try {
    if (logout(flags.profile)) {
      out(`Removed credentials for profile '${flags.profile}'.`);
    } else {
      out(`No credentials found for profile '${flags.profile}'.`);
    }
    return 0;
  } catch (e) {
    if (e instanceof BadPermissionsError || e instanceof LockAcquisitionError) {
      err(`error: ${e.message}`);
      return 1;
    }
    throw e;
  }
}

function cmdWhoami(argv: string[]): number {
  let flags: CommonFlags;
  try {
    flags = parseCommonFlags(argv);
  } catch (e) {
    err(`error: ${(e as Error).message}`);
    return 1;
  }
  let key: string | null;
  try {
    key = load(flags.profile);
  } catch (e) {
    if (e instanceof BadPermissionsError) {
      err(`error: ${e.message}`);
      return 1;
    }
    throw e;
  }
  if (!key) {
    err(`Not logged in (profile '${flags.profile}'). Run \`compresr-sdk login\`.`);
    return 1;
  }
  out(`profile:  ${flags.profile}`);
  out(`file:     ${credentialsPath()}`);
  out(`api_key:  ${key.slice(0, 8)}…${key.slice(-4)}`);
  return 0;
}

function cmdStatus(argv: string[]): number {
  if (argv.length > 0) {
    err(`error: status takes no arguments`);
    return 1;
  }
  const path = credentialsPath();
  const exists = existsSync(path);
  out(`credentials file: ${path}`);
  out(`exists:           ${exists}`);
  if (!exists) return 0;
  let profiles: string[];
  try {
    profiles = listProfiles();
  } catch (e) {
    if (e instanceof BadPermissionsError) {
      err(`error: ${e.message}`);
      return 1;
    }
    throw e;
  }
  out(`profiles:         ${profiles.length ? profiles.join(', ') : '(none)'}`);
  return 0;
}

function printHelp(): void {
  out('compresr-sdk — Compresr SDK CLI');
  out('');
  out('Commands:');
  out('  login    Log in via browser and store credentials');
  out('  logout   Remove stored credentials');
  out('  whoami   Show the stored API key preview');
  out('  status   Show credentials file location and profiles');
  out('');
  out('Flags:');
  out('  --version    Print version and exit');
  out('  --help       Print this help and exit');
}

export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
  if (argv.length === 0 || argv[0] === '--help' || argv[0] === '-h') {
    printHelp();
    return argv.length === 0 ? 1 : 0;
  }
  if (argv[0] === '--version') {
    out(`compresr-sdk ${SDK_VERSION}`);
    return 0;
  }
  const [cmd, ...rest] = argv;
  try {
    switch (cmd) {
      case 'login': return await cmdLogin(rest);
      case 'logout': return cmdLogout(rest);
      case 'whoami': return cmdWhoami(rest);
      case 'status': return cmdStatus(rest);
      default:
        err(`unknown command: ${cmd}`);
        printHelp();
        return 2;
    }
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') {
      err('\nAborted.');
      return 130;
    }
    throw e;
  }
}
