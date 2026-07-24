/** Resolve API key from (explicit → env → credentials file). Node-only. */

import { load, DEFAULT_PROFILE, BadPermissionsError } from './credentials.js';

let _badPermsWarned = false;

/**
 * An explicit non-undefined ``apiKey`` short-circuits fallbacks — even ``""``
 * flows through so downstream validation can distinguish "user passed empty"
 * from "no arg passed".
 */
export function resolveApiKey(
  explicit: string | undefined,
  profile?: string
): string | null {
  if (explicit !== undefined) return explicit;
  const env = process.env.COMPRESR_API_KEY;
  if (env) return env;
  try {
    return load(profile ?? process.env.COMPRESR_PROFILE ?? DEFAULT_PROFILE);
  } catch (e) {
    if (e instanceof BadPermissionsError) {
      if (!_badPermsWarned && process.env.COMPRESR_SUPPRESS_PERM_WARNING !== '1') {
        _badPermsWarned = true;
        process.stderr.write(`warning: ${e.message}\n`);
      }
      return null;
    }
    throw e;
  }
}
