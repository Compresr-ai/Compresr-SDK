/**
 * Integration test configuration
 */
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables from parent .env file (SDK root)
config({ path: resolve(__dirname, '../../../.env') });

export const TEST_API_KEY = process.env.COMPRESR_API_KEY ?? '';

export function skipIfNoApiKey() {
  if (!TEST_API_KEY || !TEST_API_KEY.startsWith('cmp_')) {
    return true;
  }
  return false;
}
