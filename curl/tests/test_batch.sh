#!/bin/bash
# Test Batch Compression
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
    echo "✓ COMPRESR_API_KEY not set, skipping batch compression test (optional)"
    exit 0
fi

echo "Testing batch compression endpoint..."

# Run batch compression
RESPONSE=$(bash ../compress_batch.sh 2>&1)

# Check if response contains expected batch result fields
if echo "$RESPONSE" | grep -q "results" && echo "$RESPONSE" | grep -q "total_tokens_saved"; then
    echo "✓ Batch compression test passed"
    exit 0
else
    echo "✗ Batch compression test failed"
    echo "$RESPONSE"
    exit 1
fi
