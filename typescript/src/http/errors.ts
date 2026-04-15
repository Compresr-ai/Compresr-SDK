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
  // Budget & Credits
  InsufficientCreditsError,
  BudgetLimitError,
  DailyLimitError,
  ApiKeyBudgetError,
  // Model & Input
  ModelNotFoundError,
  ContextWindowExceededError,
  ContentPolicyError,
  // Service
  TimeoutError,
  ServiceUnavailableError,
  TargetAuthenticationError,
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
  code?: string;
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
 * Map error code from response body to appropriate error class
 * Returns null if code is not recognized (fall through to status-based handling)
 */
function handleErrorByCode(code: string, msg: string, body: ErrorBody, normalized: Partial<ErrorResponseData>): CompresrError | null {
  switch (code) {
    // Budget & Credits
    case 'insufficient_credits':
      return new InsufficientCreditsError(msg, undefined, undefined, normalized);
    case 'budget_limit_reached':
      return new BudgetLimitError(msg, undefined, undefined, normalized);
    case 'daily_limit_exceeded':
      return new DailyLimitError(msg, undefined, undefined, normalized);
    case 'api_key_budget_exceeded':
      return new ApiKeyBudgetError(msg, undefined, undefined, normalized);
    
    // Model & Input
    case 'model_not_found':
      return new ModelNotFoundError(msg, undefined, undefined, normalized);
    case 'context_window_exceeded':
      return new ContextWindowExceededError(msg, undefined, undefined, normalized);
    case 'content_policy_violation':
      return new ContentPolicyError(msg, undefined, normalized);
    
    // Service
    case 'timeout':
      return new TimeoutError(msg, undefined, normalized);
    case 'service_unavailable':
      return new ServiceUnavailableError(msg, undefined, body.retry_after, normalized);
    case 'target_authentication_error':
      return new TargetAuthenticationError(msg, undefined, normalized);
    
    // Auth & Permissions
    case 'authentication_error':
      return new AuthenticationError(`Authentication failed: ${msg}`, normalized);
    case 'scope_error':
      return new ScopeError(msg, undefined, normalized);
    
    // Resource
    case 'not_found':
      return new NotFoundError(msg, undefined, normalized);
    
    // Validation
    case 'validation_error':
      return new ValidationError(msg, body.field, normalized);
    
    // Rate limiting
    case 'rate_limit_exceeded':
      return new RateLimitError(msg, body.retry_after, normalized);

    default:
      return null;
  }
}

/**
 * Map HTTP status code and body to appropriate error class
 * First checks error code in body, then falls back to HTTP status
 * @throws Always throws an appropriate CompresrError subclass
 */
export function handleHttpError(status: number, body: ErrorBody): never {
  const msg = extractErrorMessage(body);
  const normalized = normalizeErrorBody(body);

  // First, try to map by error code if present
  if (body.code) {
    const codeError = handleErrorByCode(body.code, msg, body, normalized);
    if (codeError) {
      throw codeError;
    }
  }

  // Fall back to status-based error handling
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
