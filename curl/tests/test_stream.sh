#!/bin/bash
# Test Streaming Compression
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
    echo "✓ COMPRESR_API_KEY not set, skipping stream compression test (optional)"
    exit 0
fi

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
