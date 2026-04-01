/**
 * HTTP error handling utilities
 */
import { STATUS_CODES } from '../config/constants.js';
import {
  AuthenticationError,
  CompresrError,
  NotFoundError,
  RateLimitError,
  ScopeError,
  ServerError,
  ValidationError,
  type ErrorResponseData,
} from '../errors/index.js';

/**
 * Error body from API response
 */
export interface ErrorBody {
  error?: string;
  detail?: string | Array<{ loc?: string[]; msg?: string }>;
  message?: string;
  field?: string;
  retry_after?: number;
}

/**
 * Convert ErrorBody to ErrorResponseData (normalizes detail to string)
 */
function normalizeErrorBody(body: ErrorBody): Partial<ErrorResponseData> {
  let detail: string | undefined;
  
  if (body.detail) {
    if (Array.isArray(body.detail)) {
      detail = body.detail
        .map((e) => {
          const field = e.loc?.join('.') ?? '';
          const msg = e.msg ?? '';
          return field ? `${field}: ${msg}` : msg;
        })
        .join('; ') || 'Validation error';
    } else {
      detail = body.detail;
    }
  }

  return {
    error: body.error,
    detail,
    field: body.field,
    retry_after: body.retry_after,
  };
}

/**
 * Extract user-friendly error message from API response
 */
function extractErrorMessage(body: ErrorBody): string {
  if (body.error) {
    return body.error;
  }

  if (body.detail) {
    if (Array.isArray(body.detail)) {
      const messages = body.detail.map((e) => {
        const field = e.loc?.join('.') ?? '';
        const msg = e.msg ?? '';
        return field ? `${field}: ${msg}` : msg;
      });
      return messages.join('; ') || 'Validation error';
    }
    return body.detail;
  }

  if (body.message) {
    return body.message;
  }

  return 'Unknown error';
}

/**
 * Map HTTP status code and body to appropriate error class
 * @throws Always throws an appropriate CompresrError subclass
 */
export function handleHttpError(status: number, body: ErrorBody): never {
  const msg = extractErrorMessage(body);
  const normalized = normalizeErrorBody(body);

  switch (status) {
    case STATUS_CODES.UNAUTHORIZED:
      throw new AuthenticationError(
        `Authentication failed: ${msg}. Check your API key is valid.`,
        normalized
      );

    case STATUS_CODES.FORBIDDEN:
      throw new ScopeError(
        `Permission denied: ${msg}. Your API key may lack the required scope.`,
        undefined,
        normalized
      );

    case STATUS_CODES.NOT_FOUND:
      throw new NotFoundError(`Resource not found: ${msg}`, undefined, normalized);

    case STATUS_CODES.VALIDATION_ERROR:
      throw new ValidationError(`Invalid request: ${msg}`, body.field, normalized);

    case STATUS_CODES.RATE_LIMITED: {
      const retryMsg = body.retry_after
        ? ` Retry after ${body.retry_after} seconds.`
        : '';
      throw new RateLimitError(
        `Rate limit exceeded: ${msg}.${retryMsg}`,
        body.retry_after,
        normalized
      );
    }

    default:
      if (status >= 500) {
        throw new ServerError(
          `Server error: ${msg}. Please try again later or contact support.`,
          normalized
        );
      }
      throw new CompresrError(
        `Request failed (${status}): ${msg}`,
        'request_error',
        normalized
      );
  }
}
