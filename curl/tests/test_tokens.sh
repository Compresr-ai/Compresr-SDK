#!/bin/bash
# Test Token Counting Endpoint
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
    echo "✓ COMPRESR_API_KEY not set, skipping tokens test (optional)"
    exit 0
fi

echo "Testing tokens endpoint..."

# Run tokens request
RESPONSE=$(bash ../tokens.sh 2>&1)

# Check if response contains token count
if echo "$RESPONSE" | grep -q "tokens"; then
    echo "✓ Tokens test passed"
    exit 0
else
    echo "✗ Tokens test failed"
    echo "$RESPONSE"
    exit 1
fi
