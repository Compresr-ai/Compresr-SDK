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
