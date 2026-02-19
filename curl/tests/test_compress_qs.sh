#!/bin/bash
# Test Question-Specific Compression
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
    echo "✓ COMPRESR_API_KEY not set, skipping QS compression test (optional)"
    exit 0
fi

echo "Testing QS compression endpoint..."

# Run QS compression
RESPONSE=$(bash ../compress_qs.sh 2>&1)

# Check if response contains expected fields
if echo "$RESPONSE" | grep -q "compressed_context" && echo "$RESPONSE" | grep -q "tokens_saved"; then
    echo "✓ QS compression test passed"
    exit 0
else
    echo "✗ QS compression test failed"
    echo "$RESPONSE"
    exit 1
fi
