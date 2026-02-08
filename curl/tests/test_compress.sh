#!/bin/bash
set -e

# Check if API key is already set (CI secret), if not try .env (local)
if [ -z "$COMPRESR_API_KEY" ] && [ -f "../../.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "../../.env"
    set +a
fi

# Skip if still no API key
if [ -z "$COMPRESR_API_KEY" ]; then
    echo "✓ COMPRESR_API_KEY not set, skipping compression test (optional)"
    exit 0
fi

echo "Testing compression endpoint..."

# Create a test context
TEST_CONTEXT="This is a long test context that should be compressed. It contains multiple sentences and provides enough text for the compression model to work with. The goal is to reduce the token count while maintaining semantic meaning."

# Run compression
RESPONSE=$(bash ../compress.sh "$TEST_CONTEXT" 2>&1)

# Check if response contains expected fields
if echo "$RESPONSE" | grep -q "compressed_context" && echo "$RESPONSE" | grep -q "tokens_saved"; then
    echo "✓ Compression test passed"
    exit 0
else
    echo "✗ Compression test failed"
    echo "$RESPONSE"
    exit 1
fi
