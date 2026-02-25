"""
Compresr SDK Tests

Integration tests for the Compresr Python SDK.

Test Modules:
    - test_compression_client: Tests for CompressionClient (token-level compression)
    - test_qs_compression_client: Tests for FilterClient (chunk-level filtering)

Running Tests:
    pytest tests/ -v                    # Run all tests
    pytest tests/ -v -m "not slow"      # Skip slow tests
    pytest tests/integration/test_compression_client.py -v  # Run compression tests only
    pytest tests/integration/test_qs_compression_client.py -v  # Run filter tests only

Environment Variables:
    COMPRESR_API_KEY: API key for compression operations
    COMPRESR_BASE_URL: Base URL for the API (default: http://localhost:8000)
"""
