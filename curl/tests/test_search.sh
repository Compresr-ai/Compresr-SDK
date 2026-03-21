#!/bin/bash
# Test Agentic Search
set -e

# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

echo "Testing agentic search endpoint..."

# Run agentic search
RESPONSE=$(bash ../search.sh 2>&1)

# Check if response contains expected fields
if echo "$RESPONSE" | grep -q "chunks" && echo "$RESPONSE" | grep -q "chunks_count"; then
    echo "✓ Agentic search test passed"
    exit 0
else
    echo "✗ Agentic search test failed"
    echo "$RESPONSE"
    exit 1
fi
