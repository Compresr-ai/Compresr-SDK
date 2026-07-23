#!/bin/bash
# Test Models Endpoint
set -e

# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

echo "Testing models endpoint..."

# Run models request
RESPONSE=$(bash ../models.sh 2>&1)

# Check if response contains expected model name.
if echo "$RESPONSE" | grep -q "latte_v2"; then
    echo "✓ Models test passed"
    exit 0
else
    echo "✗ Models test failed"
    echo "$RESPONSE"
    exit 1
fi
