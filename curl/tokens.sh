#!/bin/bash
# Compresr curl SDK - Token Counting

set -e

# Load .env from parent directory
if [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$(dirname "$0")/../.env"
    set +a
fi

BASE_URL="${COMPRESR_BASE_URL:-https://api.compresr.ai}"

echo "Counting tokens..."
echo ""

curl -s -X POST "$BASE_URL/api/tokens/count" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quick brown fox jumps over the lazy dog. This is a sample text for token counting."
  }' | jq .
