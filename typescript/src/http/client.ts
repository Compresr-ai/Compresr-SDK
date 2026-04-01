/**
 * Base HTTP client for Compresr API
 *
 * Uses native fetch API for Node.js 18+ and browser compatibility.
 */
import {
  API_KEY_PREFIX,
  DEFAULT_BASE_URL,
  DEFAULT_TIMEOUT,
  STREAM_TIMEOUT,
  HEADERS,
} from '../config/constants.js';
import {
  AuthenticationError,
  CompresrError,
  ConnectionError,
} from '../errors/index.js';
import { handleHttpError, type ErrorBody } from './errors.js';

const SDK_VERSION = '1.0.0';

/**
 * HTTP client configuration options
 */
export interface HttpClientOptions {
  /** API key (required) - must start with "cmp_" */
  apiKey: string;
  /** Base URL for API (optional, defaults to production) */
  baseUrl?: string;
  /** Request timeout in milliseconds (optional) */
  timeout?: number;
}

/**
 * Internal HTTP client for all Compresr API requests
 */
export class HttpClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly timeout: number;

  constructor(options: HttpClientOptions) {
    // Validate API key
    if (!options.apiKey) {
      throw new AuthenticationError('API key is required');
    }
    if (!options.apiKey.startsWith(API_KEY_PREFIX)) {
      throw new AuthenticationError(
        `Invalid API key format. Keys must start with '${API_KEY_PREFIX}'`
      );
    }

    this.apiKey = options.apiKey;
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
  }

  /**
   * Get default headers for requests
   */
  private get headers(): Record<string, string> {
    return {
      [HEADERS.API_KEY]: this.apiKey,
      [HEADERS.CONTENT_TYPE]: 'application/json',
      [HEADERS.ACCEPT]: 'application/json',
      [HEADERS.USER_AGENT]: `compresr-typescript-sdk/${SDK_VERSION}`,
    };
  }

  /**
   * Build full URL from endpoint
   */
  private url(endpoint: string): string {
    return `${this.baseUrl}${endpoint}`;
  }

  /**
   * Make a POST request
   */
  async post<T>(endpoint: string, data: Record<string, unknown>): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(this.url(endpoint), {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(data),
        signal: controller.signal,
      });

      const body = (await response.json()) as T | ErrorBody;

      if (!response.ok) {
        handleHttpError(response.status, body as ErrorBody);
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

  /**
   * Make a GET request
   */
  async get<T>(endpoint: string): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(this.url(endpoint), {
        method: 'GET',
        headers: this.headers,
        signal: controller.signal,
      });

      const body = (await response.json()) as T | ErrorBody;

      if (!response.ok) {
        handleHttpError(response.status, body as ErrorBody);
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

  /**
   * Make a DELETE request
   */
  async delete<T>(endpoint: string): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(this.url(endpoint), {
        method: 'DELETE',
        headers: this.headers,
        signal: controller.signal,
      });

      const body = (await response.json()) as T | ErrorBody;

      if (!response.ok) {
        handleHttpError(response.status, body as ErrorBody);
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

  /**
   * Stream response from SSE endpoint
   */
  async *stream(
    endpoint: string,
    data: Record<string, unknown>
  ): AsyncGenerator<string, void, undefined> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), STREAM_TIMEOUT);

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
        const body = (await response.json()) as ErrorBody;
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
              // Yield raw content if not JSON
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

  /**
   * Upload multipart form data (for index creation)
   */
  async postMultipart<T>(
    endpoint: string,
    files: Record<string, { data: Blob; filename: string; contentType: string }>
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), STREAM_TIMEOUT);

    try {
      const formData = new FormData();
      for (const [key, file] of Object.entries(files)) {
        formData.append(key, file.data, file.filename);
      }

      const response = await fetch(this.url(endpoint), {
        method: 'POST',
        headers: {
          [HEADERS.API_KEY]: this.apiKey,
          [HEADERS.USER_AGENT]: `compresr-typescript-sdk/${SDK_VERSION}`,
        },
        body: formData,
        signal: controller.signal,
      });

      const body = (await response.json()) as T | ErrorBody;

      if (!response.ok) {
        handleHttpError(response.status, body as ErrorBody);
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
}
