#!/bin/bash
set -e

echo "Testing health endpoint..."

# Run health check
RESPONSE=$(bash ../health.sh 2>&1)

# Check if response contains "status"
if echo "$RESPONSE" | grep -q "status"; then
    echo "✓ Health check passed"
    exit 0
else
    echo "✗ Health check failed"
    echo "$RESPONSE"
    exit 1
fi
