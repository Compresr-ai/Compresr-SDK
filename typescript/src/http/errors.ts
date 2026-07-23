/** HTTP error handling utilities. */
import { STATUS_CODES } from '../config/constants.js';
import {
  AuthenticationError,
  CompresrError,
  NotFoundError,
  RateLimitError,
  ScopeError,
  ServerError,
  ValidationError,
  InsufficientCreditsError,
  BudgetLimitError,
  DailyLimitError,
  ApiKeyBudgetError,
  ModelNotFoundError,
  ContextWindowExceededError,
  ContentPolicyError,
  TimeoutError,
  ServiceUnavailableError,
  TargetAuthenticationError,
  type ErrorResponseData,
} from '../errors/index.js';

export interface ErrorBody {
  error?: string;
  detail?: string | Array<{ loc?: string[]; msg?: string }>;
  message?: string;
  field?: string;
  retry_after?: number;
  code?: string;
}

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

function handleErrorByCode(code: string, msg: string, body: ErrorBody, normalized: Partial<ErrorResponseData>): CompresrError | null {
  switch (code) {
    case 'insufficient_credits':
      return new InsufficientCreditsError(msg, undefined, undefined, normalized);
    case 'budget_limit_reached':
      return new BudgetLimitError(msg, undefined, undefined, normalized);
    case 'daily_limit_exceeded':
      return new DailyLimitError(msg, undefined, undefined, normalized);
    case 'api_key_budget_exceeded':
      return new ApiKeyBudgetError(msg, undefined, undefined, normalized);
    case 'model_not_found':
      return new ModelNotFoundError(msg, undefined, undefined, normalized);
    case 'context_window_exceeded':
      return new ContextWindowExceededError(msg, undefined, undefined, normalized);
    case 'content_policy_violation':
      return new ContentPolicyError(msg, undefined, normalized);
    case 'timeout':
      return new TimeoutError(msg, undefined, normalized);
    case 'service_unavailable':
      return new ServiceUnavailableError(msg, undefined, body.retry_after, normalized);
    case 'target_authentication_error':
      return new TargetAuthenticationError(msg, undefined, normalized);
    case 'authentication_error':
      return new AuthenticationError(`Authentication failed: ${msg}`, normalized);
    case 'scope_error':
      return new ScopeError(msg, undefined, normalized);
    case 'not_found':
      return new NotFoundError(msg, undefined, normalized);
    case 'validation_error':
      return new ValidationError(msg, body.field, normalized);
    case 'rate_limit_exceeded':
      return new RateLimitError(msg, body.retry_after, normalized);

    default:
      return null;
  }
}

export function handleHttpError(status: number, body: ErrorBody): never {
  const msg = extractErrorMessage(body);
  const normalized = normalizeErrorBody(body);

  if (body.code) {
    const codeError = handleErrorByCode(body.code, msg, body, normalized);
    if (codeError) {
      throw codeError;
    }
  }

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
      if (status === 503) {
        throw new ServiceUnavailableError(
          `Service temporarily unavailable: ${msg}`,
          undefined,
          body.retry_after,
          normalized
        );
      }
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
