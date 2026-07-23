/**
 * Name-based eligibility filter for tool / middleware integrations.
 *
 * Mirrors Python `_shared/filters.py`.
 */

export interface FilterOptions {
  allow?: Iterable<string>;
  ignore?: Iterable<string>;
}

/**
 * Build a name predicate from allow- or ignore-lists.
 *
 * - `allow`:  only names in this set return true.
 * - `ignore`: every name except those in this set returns true.
 * - both undefined: every name returns true (no-op).
 *
 * Passing both throws — pick one mode.
 */
export function makeFilter(
  options: FilterOptions = {}
): (name?: string | null) => boolean {
  const { allow, ignore } = options;
  if (allow !== undefined && ignore !== undefined) {
    throw new Error('Pass `allow` OR `ignore`, not both.');
  }
  if (allow !== undefined) {
    const allowed = new Set(allow);
    return (name) => name != null && allowed.has(name);
  }
  if (ignore !== undefined) {
    const ignored = new Set(ignore);
    return (name) => name == null || !ignored.has(name);
  }
  return () => true;
}
