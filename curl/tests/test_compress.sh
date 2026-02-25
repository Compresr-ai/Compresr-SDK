#!/bin/bash
set -e

# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

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
