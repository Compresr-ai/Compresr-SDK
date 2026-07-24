/** Base HTTP client for Compresr API. */
import {
  API_KEY_PREFIX,
  DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT,
  HEADERS,
} from '../config/constants.js';
import { getLogger } from '../logger.js';
import {
  AuthenticationError,
  CompresrError,
  ConnectionError,
  RateLimitError,
  ServiceUnavailableError,
} from '../errors/index.js';
import { handleHttpError, type ErrorBody } from './errors.js';
import {
  computeBackoffMs,
  resolveRetryConfig,
  sleep,
  type ResolvedRetryConfig,
  type RetryConfig,
} from './retry.js';
import { SDK_VERSION } from '../version.js';

const MAX_RESPONSE_BYTES = 50 * 1024 * 1024;

const LOCAL_HOSTNAMES = new Set(['localhost', '127.0.0.1', '::1']);
async function readBoundedJson<T>(response: Response): Promise<T> {
  const cl = response.headers.get('content-length');
  if (cl !== null) {
    const declared = Number(cl);
    if (Number.isFinite(declared) && declared > MAX_RESPONSE_BYTES) {
      throw new CompresrError(
        `Response too large: content-length ${declared} bytes exceeds ` +
          `cap (${MAX_RESPONSE_BYTES} bytes).`,
        'response_too_large'
      );
    }
  }

  if (!response.body) {
    const text = await response.text();
    if (text.length > MAX_RESPONSE_BYTES) {
      throw new CompresrError(
        `Response too large: ${text.length} bytes exceeds cap ` +
          `(${MAX_RESPONSE_BYTES} bytes).`,
        'response_too_large'
      );
    }
    return parseJson<T>(text);
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      total += value.byteLength;
      if (total > MAX_RESPONSE_BYTES) {
        try {
          await reader.cancel();
        } catch {
          // ignored — best-effort cleanup
        }
        throw new CompresrError(
          `Response too large: streamed ${total} bytes exceeds cap ` +
            `(${MAX_RESPONSE_BYTES} bytes).`,
          'response_too_large'
        );
      }
      chunks.push(value);
    }
  }
  const buf = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return parseJson<T>(new TextDecoder().decode(buf));
}

function parseJson<T>(text: string): T {
  if (text.length === 0) {
    return {} as T;
  }
  try {
    return JSON.parse(text) as T;
  } catch (err) {
    throw new CompresrError(
      `Failed to parse JSON response: ${err instanceof Error ? err.message : String(err)}`,
      'invalid_response'
    );
  }
}

export interface HttpClientOptions {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  retry?: RetryConfig;
}

export class HttpClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly timeout: number;
  private readonly retryConfig: ResolvedRetryConfig;

  constructor(options: HttpClientOptions) {
    if (!options.apiKey) {
      throw new AuthenticationError('API key is required');
    }
    if (!options.apiKey.startsWith(API_KEY_PREFIX)) {
      throw new AuthenticationError(
        `Invalid API key format. Keys must start with '${API_KEY_PREFIX}'`
      );
    }

    const rawBase = options.baseUrl ?? DEFAULT_BASE_URL;
    let parsed: URL;
    try {
      parsed = new URL(rawBase);
    } catch {
      throw new CompresrError(
        `Invalid baseUrl: '${rawBase}' is not a valid URL.`,
        'invalid_base_url'
      );
    }
    if (
      parsed.protocol === 'http:' &&
      !LOCAL_HOSTNAMES.has(parsed.hostname)
    ) {
      const allow =
        typeof process !== 'undefined' &&
        process.env?.COMPRESR_ALLOW_INSECURE === '1';
      if (!allow) {
        throw new CompresrError(
          `Refusing to send API key over cleartext http to '${rawBase}'. ` +
            "Set COMPRESR_ALLOW_INSECURE=1 to override (dev only).",
          'insecure_base_url'
        );
      }
      getLogger().warn(
        `baseUrl is ${parsed.protocol} — API key will be ` +
          'transmitted in cleartext. Use HTTPS in production.'
      );
    }

    this.apiKey = options.apiKey;
    this.baseUrl = rawBase.replace(/\/+$/, '');
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.retryConfig = resolveRetryConfig(options.retry);
  }

  private get headers(): Record<string, string> {
    return {
      [HEADERS.API_KEY]: this.apiKey,
      [HEADERS.CONTENT_TYPE]: 'application/json',
      [HEADERS.ACCEPT]: 'application/json',
      [HEADERS.USER_AGENT]: `compresr-typescript-sdk/${SDK_VERSION}`,
    };
  }

  private url(endpoint: string): string {
    if (!endpoint.startsWith('/') || endpoint.includes('://')) {
      throw new CompresrError(
        `Invalid endpoint: '${endpoint}' must be a relative path starting with '/'.`,
        'invalid_endpoint'
      );
    }
    return `${this.baseUrl}${endpoint}`;
  }

  private async attemptPost<T>(
    endpoint: string,
    data: Record<string, unknown>
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(this.url(endpoint), {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(data),
        signal: controller.signal,
      });

      const body = await readBoundedJson<T | ErrorBody>(response);

      if (!response.ok) {
        const errorBody = body as ErrorBody;
        if (errorBody.retry_after === undefined) {
          const header = response.headers.get('Retry-After');
          if (header !== null) {
            const parsed = Number(header);
            if (Number.isFinite(parsed)) {
              errorBody.retry_after = parsed;
            }
          }
        }
        handleHttpError(response.status, errorBody);
      }

      return body as T;
    } catch (error) {
      if (error instanceof CompresrError) {
        throw error;
      }
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new ConnectionError('Request timed out');
        }
        throw new ConnectionError(`Connection failed: ${error.message}`);
      }
      throw new CompresrError(`Request failed: ${String(error)}`);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async post<T>(endpoint: string, data: Record<string, unknown>): Promise<T> {
    const cfg = this.retryConfig;
    let attempt = 0;
    for (;;) {
      try {
        return await this.attemptPost<T>(endpoint, data);
      } catch (err) {
        const status =
          err instanceof RateLimitError
            ? 429
            : err instanceof ServiceUnavailableError
              ? 503
              : null;
        if (status === null || !cfg.retryOnStatus.has(status) || attempt >= cfg.maxRetries) {
          throw err;
        }
        const hint = (err as { retryAfter?: number }).retryAfter;
        const delay = computeBackoffMs(attempt, cfg, hint);
        if (delay > 0) await sleep(delay);
        attempt += 1;
      }
    }
  }

  async *stream(
    endpoint: string,
    data: Record<string, unknown>
  ): AsyncGenerator<string, void, undefined> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(this.url(endpoint), {
        method: 'POST',
        headers: {
          ...this.headers,
          [HEADERS.ACCEPT]: 'text/event-stream',
        },
        body: JSON.stringify(data),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await readBoundedJson<ErrorBody>(response);
        handleHttpError(response.status, body);
      }

      if (!response.body) {
        throw new CompresrError('No response body for stream');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const chunk = trimmed.slice(6);
            if (chunk === '[DONE]') {
              return;
            }
            try {
              const parsed = JSON.parse(chunk) as { content?: string };
              if (parsed.content) {
                yield parsed.content;
              }
            } catch {
              if (chunk) {
                yield chunk;
              }
            }
          }
        }
      }
    } catch (error) {
      if (error instanceof CompresrError) {
        throw error;
      }
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new ConnectionError('Stream timed out');
        }
        throw new ConnectionError(`Stream failed: ${error.message}`);
      }
      throw new CompresrError(`Stream failed: ${String(error)}`);
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
