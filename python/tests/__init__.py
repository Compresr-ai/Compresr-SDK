"""
Compresr SDK Tests

Integration tests for the Compresr Python SDK.

Test Modules:
    - test_completion_client: Tests for CompletionClient (compression + LLM)
    - test_compression_client: Tests for CompressionClient (compression only)

Running Tests:
    pytest tests/ -v                    # Run all tests
    pytest tests/ -v -m "not slow"      # Skip slow tests
    pytest tests/test_completion_client.py -v   # Run completion tests only
    pytest tests/test_compression_client.py -v  # Run compression tests only

Environment Variables:
    COMPLETION_API_KEY: API key for completion operations
    COMPRESSION_API_KEY: API key for compression operations
    COMPRESR_BASE_URL: Base URL for the API (default: http://localhost:8000)
"""
