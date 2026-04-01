/**
 * Unit tests for error classes
 */
import { describe, it, expect } from 'vitest';
import {
  CompresrError,
  AuthenticationError,
  ValidationError,
  RateLimitError,
  ScopeError,
  ServerError,
  ConnectionError,
  NotFoundError,
} from '../../src/errors/index.js';

describe('CompresrError', () => {
  it('should create base error with message', () => {
    const error = new CompresrError('Something went wrong');
    
    expect(error).toBeInstanceOf(Error);
    expect(error).toBeInstanceOf(CompresrError);
    expect(error.message).toBe('Something went wrong');
    expect(error.code).toBe('compresr_error');
    expect(error.responseData.success).toBe(false);
    expect(error.responseData.error).toBe('Something went wrong');
  });

  it('should create error with custom code', () => {
    const error = new CompresrError('Custom error', 'custom_code');
    
    expect(error.code).toBe('custom_code');
  });

  it('should include response data', () => {
    const responseData = { detail: 'More info' };
    const error = new CompresrError('Error', 'code', responseData);
    
    expect(error.responseData.detail).toBe('More info');
  });
});

describe('AuthenticationError', () => {
  it('should create authentication error with default message', () => {
    const error = new AuthenticationError();
    
    expect(error).toBeInstanceOf(CompresrError);
    expect(error).toBeInstanceOf(AuthenticationError);
    expect(error.message).toBe('Authentication failed');
    expect(error.code).toBe('authentication_error');
    expect(error.name).toBe('AuthenticationError');
  });

  it('should create authentication error with custom message', () => {
    const error = new AuthenticationError('Invalid API key');
    
    expect(error.message).toBe('Invalid API key');
  });
});

describe('ValidationError', () => {
  it('should create validation error with field', () => {
    const error = new ValidationError('Invalid value', 'context');
    
    expect(error).toBeInstanceOf(CompresrError);
    expect(error).toBeInstanceOf(ValidationError);
    expect(error.message).toBe('Invalid value');
    expect(error.code).toBe('validation_error');
    expect(error.field).toBe('context');
    expect(error.name).toBe('ValidationError');
  });

  it('should create validation error without field', () => {
    const error = new ValidationError('Invalid request');
    
    expect(error.field).toBeUndefined();
  });
});

describe('RateLimitError', () => {
  it('should create rate limit error with retry_after', () => {
    const error = new RateLimitError('Too many requests', 30);
    
    expect(error).toBeInstanceOf(CompresrError);
    expect(error).toBeInstanceOf(RateLimitError);
    expect(error.message).toBe('Too many requests');
    expect(error.code).toBe('rate_limit_exceeded');
    expect(error.retryAfter).toBe(30);
    expect(error.name).toBe('RateLimitError');
  });

  it('should create rate limit error without retry_after', () => {
    const error = new RateLimitError('Rate limited');
    
    expect(error.retryAfter).toBeUndefined();
  });
});

describe('ScopeError', () => {
  it('should create scope error with required scope', () => {
    const error = new ScopeError('Missing permission', 'admin');
    
    expect(error).toBeInstanceOf(CompresrError);
    expect(error).toBeInstanceOf(ScopeError);
    expect(error.message).toBe('Missing permission');
    expect(error.code).toBe('scope_error');
    expect(error.requiredScope).toBe('admin');
    expect(error.name).toBe('ScopeError');
  });
});

describe('ServerError', () => {
  it('should create server error with default message', () => {
    const error = new ServerError();
    
    expect(error).toBeInstanceOf(CompresrError);
    expect(error).toBeInstanceOf(ServerError);
    expect(error.message).toBe('Internal server error');
    expect(error.code).toBe('server_error');
    expect(error.name).toBe('ServerError');
  });
});

describe('ConnectionError', () => {
  it('should create connection error with service', () => {
    const error = new ConnectionError('Failed to connect', 'api');
    
    expect(error).toBeInstanceOf(CompresrError);
    expect(error).toBeInstanceOf(ConnectionError);
    expect(error.message).toBe('Failed to connect');
    expect(error.code).toBe('connection_error');
    expect(error.service).toBe('api');
    expect(error.name).toBe('ConnectionError');
  });
});

describe('NotFoundError', () => {
  it('should create not found error with resource', () => {
    const error = new NotFoundError('Index not found', 'index');
    
    expect(error).toBeInstanceOf(CompresrError);
    expect(error).toBeInstanceOf(NotFoundError);
    expect(error.message).toBe('Index not found');
    expect(error.code).toBe('not_found');
    expect(error.resource).toBe('index');
    expect(error.name).toBe('NotFoundError');
  });
});
