/**
 * Auth subsystem — Node-only. Import via ``@compresr/sdk/auth``.
 *
 * The root ``@compresr/sdk`` entry stays browser-safe by not re-exporting
 * this module. ``CompressionClient`` uses a guarded dynamic import to pick
 * up ``resolveApiKey`` when it's available (Node) and no-ops silently on
 * browsers.
 */

export { login, logout, sanitizeErrorCode, escapeHtml, validateUrl } from './login.js';
export type { LoginOptions } from './login.js';
export {
  credentialsPath,
  load,
  save,
  clear,
  listProfiles,
  DEFAULT_PROFILE,
  BadPermissionsError,
} from './credentials.js';
export type { CredentialsSection, SaveOptions } from './credentials.js';
export { resolveApiKey } from './resolver.js';

import { CompressionClient, CompressionClientOptions } from '../clients/compression.js';
import { resolveApiKey } from './resolver.js';

/**
 * Node-only convenience: build a ``CompressionClient`` with automatic API
 * key pickup from (explicit → ``COMPRESR_API_KEY`` env →
 * ``~/.compresr/credentials.json``). This matches the Python SDK's default
 * ``CompressionClient()`` behavior.
 */
export function createClient(
  overrides: Partial<CompressionClientOptions> & { profile?: string } = {}
): CompressionClient {
  const { profile, ...rest } = overrides;
  const apiKey = resolveApiKey(rest.apiKey, profile);
  return new CompressionClient({ ...rest, apiKey: apiKey ?? '' });
}
