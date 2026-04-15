/**
 * Compresr SDK Exceptions
 *
 * Exception classes for API error handling, mirroring the Python SDK.
 */

/**
 * Response data structure for errors
 */
export interface ErrorResponseData {
  success: boolean;
  error: string;
  detail?: string;
  code?: string;
  field?: string;
  retry_after?: number;
}

/**
 * Base error class for all Compresr SDK errors
 */
export class CompresrError extends Error {
  readonly code: string;
  readonly responseData: ErrorResponseData;

  constructor(
    message: string,
    code = 'compresr_error',
    responseData: Partial<ErrorResponseData> = {}
  ) {
    super(message);
    this.name = 'CompresrError';
    this.code = code;
    this.responseData = { success: false, error: message, ...responseData };
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Invalid or missing API key
 */
export class AuthenticationError extends CompresrError {
  constructor(
    message = 'Authentication failed',
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'authentication_error', responseData);
    this.name = 'AuthenticationError';
  }
}

/**
 * Request validation failed (invalid parameters, missing required fields)
 */
export class ValidationError extends CompresrError {
  readonly field?: string;

  constructor(
    message: string,
    field?: string,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'validation_error', { ...responseData, field });
    this.name = 'ValidationError';
    this.field = field;
  }
}

/**
 * Rate limit exceeded - too many requests
 */
export class RateLimitError extends CompresrError {
  readonly retryAfter?: number;

  constructor(
    message: string,
    retryAfter?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'rate_limit_exceeded', {
      ...responseData,
      retry_after: retryAfter,
    });
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

/**
 * API key lacks required permissions/scope
 */
export class ScopeError extends CompresrError {
  readonly requiredScope?: string;

  constructor(
    message: string,
    requiredScope?: string,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'scope_error', responseData);
    this.name = 'ScopeError';
    this.requiredScope = requiredScope;
  }
}

/**
 * Internal server error
 */
export class ServerError extends CompresrError {
  constructor(
    message = 'Internal server error',
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'server_error', responseData);
    this.name = 'ServerError';
  }
}

/**
 * Connection error - failed to connect to API
 */
export class ConnectionError extends CompresrError {
  readonly service?: string;

  constructor(
    message: string,
    service?: string,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'connection_error', responseData);
    this.name = 'ConnectionError';
    this.service = service;
  }
}

/**
 * Resource not found (e.g., index doesn't exist)
 */
export class NotFoundError extends CompresrError {
  readonly resource?: string;

  constructor(
    message: string,
    resource?: string,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'not_found', responseData);
    this.name = 'NotFoundError';
    this.resource = resource;
  }
}

// =============================================================================
// Budget & Credits Errors
// =============================================================================

/**
 * User has insufficient credits to complete the request
 */
export class InsufficientCreditsError extends CompresrError {
  readonly creditsRequired?: number;
  readonly creditsRemaining?: number;

  constructor(
    message = 'Insufficient credits',
    creditsRequired?: number,
    creditsRemaining?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'insufficient_credits', responseData);
    this.name = 'InsufficientCreditsError';
    this.creditsRequired = creditsRequired;
    this.creditsRemaining = creditsRemaining;
  }
}

/**
 * User's budget limit has been reached
 */
export class BudgetLimitError extends CompresrError {
  readonly currentBudget?: number;
  readonly budgetUsed?: number;

  constructor(
    message = 'Budget limit reached',
    currentBudget?: number,
    budgetUsed?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'budget_limit_reached', responseData);
    this.name = 'BudgetLimitError';
    this.currentBudget = currentBudget;
    this.budgetUsed = budgetUsed;
  }
}

/**
 * Daily request limit exceeded
 */
export class DailyLimitError extends CompresrError {
  readonly dailyLimit?: number;
  readonly requestsUsed?: number;

  constructor(
    message = 'Daily limit exceeded',
    dailyLimit?: number,
    requestsUsed?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'daily_limit_exceeded', responseData);
    this.name = 'DailyLimitError';
    this.dailyLimit = dailyLimit;
    this.requestsUsed = requestsUsed;
  }
}

/**
 * Per-API-key budget limit exceeded
 */
export class ApiKeyBudgetError extends CompresrError {
  readonly apiKeyBudget?: number;
  readonly apiKeyUsed?: number;

  constructor(
    message = 'API key budget exceeded',
    apiKeyBudget?: number,
    apiKeyUsed?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'api_key_budget_exceeded', responseData);
    this.name = 'ApiKeyBudgetError';
    this.apiKeyBudget = apiKeyBudget;
    this.apiKeyUsed = apiKeyUsed;
  }
}

// =============================================================================
// Model & Input Errors
// =============================================================================

/**
 * Requested model does not exist
 */
export class ModelNotFoundError extends CompresrError {
  readonly modelName?: string;
  readonly availableModels?: string[];

  constructor(
    message: string,
    modelName?: string,
    availableModels?: string[],
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'model_not_found', responseData);
    this.name = 'ModelNotFoundError';
    this.modelName = modelName;
    this.availableModels = availableModels ?? [];
  }
}

/**
 * Input exceeds model's context window size
 */
export class ContextWindowExceededError extends CompresrError {
  readonly maxTokens?: number;
  readonly actualTokens?: number;

  constructor(
    message = 'Context window exceeded',
    maxTokens?: number,
    actualTokens?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'context_window_exceeded', responseData);
    this.name = 'ContextWindowExceededError';
    this.maxTokens = maxTokens;
    this.actualTokens = actualTokens;
  }
}

/**
 * Content violates provider's content policy
 */
export class ContentPolicyError extends CompresrError {
  readonly provider?: string;

  constructor(
    message = 'Content policy violation',
    provider?: string,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'content_policy_violation', responseData);
    this.name = 'ContentPolicyError';
    this.provider = provider;
  }
}

// =============================================================================
// Service Errors
// =============================================================================

/**
 * Request timed out
 */
export class TimeoutError extends CompresrError {
  readonly timeoutSeconds?: number;

  constructor(
    message = 'Request timed out',
    timeoutSeconds?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'timeout', responseData);
    this.name = 'TimeoutError';
    this.timeoutSeconds = timeoutSeconds;
  }
}

/**
 * Service is temporarily unavailable
 */
export class ServiceUnavailableError extends CompresrError {
  readonly service?: string;
  readonly retryAfter?: number;

  constructor(
    message = 'Service temporarily unavailable',
    service?: string,
    retryAfter?: number,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'service_unavailable', responseData);
    this.name = 'ServiceUnavailableError';
    this.service = service;
    this.retryAfter = retryAfter;
  }
}

/**
 * User's target LLM API key is invalid
 */
export class TargetAuthenticationError extends CompresrError {
  readonly provider?: string;

  constructor(
    message = 'Invalid target API key',
    provider?: string,
    responseData?: Partial<ErrorResponseData>
  ) {
    super(message, 'target_authentication_error', responseData);
    this.name = 'TargetAuthenticationError';
    this.provider = provider;
  }
}
