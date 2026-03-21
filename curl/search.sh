#!/bin/bash
# Compresr curl SDK - Agentic Search

set -e

# Load .env from parent directory
if [ -f "$(dirname "$0")/../.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$(dirname "$0")/../.env"
    set +a
fi

BASE_URL="${COMPRESR_BASE_URL:-https://api.compresr.ai}"
API_KEY="${COMPRESR_API_KEY:?Error: Set COMPRESR_API_KEY in ../.env}"

echo "Agentic Search..."
echo "This searches a pre-indexed knowledge base using multi-round LLM reasoning."
echo ""

curl -s -X POST "$BASE_URL/api/compress/search/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "query": "What is machine learning?",
    "index_name": "my-knowledge-base",
    "compression_model_name": "macchiato_v1",
    "max_time_s": 4.5,
    "source": "sdk:curl"
  }' | jq .
