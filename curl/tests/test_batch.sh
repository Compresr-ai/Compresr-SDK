#!/bin/bash
# Test Batch Compression
set -e

# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

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
