#!/bin/bash
# Test Streaming Compression
set -e

# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

echo "Testing streaming compression endpoint..."

# Run streaming compression and capture output
RESPONSE=$(bash ../compress_stream.sh 2>&1)

# Check if response contains SSE data or stream content
if echo "$RESPONSE" | grep -q "data:" || echo "$RESPONSE" | grep -q "event:"; then
    echo "✓ Streaming compression test passed"
    exit 0
else
    echo "✗ Streaming compression test failed"
    echo "$RESPONSE"
    exit 1
fi
