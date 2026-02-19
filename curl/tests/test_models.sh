#!/bin/bash
# Test Models Endpoint
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
    echo "✓ COMPRESR_API_KEY not set, skipping models test (optional)"
    exit 0
fi

echo "Testing models endpoint..."

# Run models request
RESPONSE=$(bash ../models.sh 2>&1)

# Check if response contains expected model names
if echo "$RESPONSE" | grep -q "A_CMPRSR_V1" && echo "$RESPONSE" | grep -q "QS_CMPRSR_V1"; then
    echo "✓ Models test passed"
    exit 0
else
    echo "✗ Models test failed"
    echo "$RESPONSE"
    exit 1
fi
