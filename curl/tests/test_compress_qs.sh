#!/bin/bash
# Test Question-Specific Compression
set -e

# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

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
