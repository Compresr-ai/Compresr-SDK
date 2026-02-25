#!/bin/bash
# Compresr curl SDK - List Models

set -e

# Load .env from parent directory
if [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$(dirname "$0")/../.env"
    set +a
fi

BASE_URL="${COMPRESR_BASE_URL:-https://api.compresr.ai}"

echo "Fetching available compression models..."
echo ""

curl -s "$BASE_URL/api/compress/question-agnostic/models" | jq .
