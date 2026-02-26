#!/bin/bash
# Compresr curl SDK - Streaming Compression

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

echo "Streaming compression..."
echo ""

curl -s -N -X POST "$BASE_URL/api/compress/question-agnostic/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "Accept: text/event-stream" \
  -d '{
    "context": "Artificial intelligence and machine learning are transforming how we build software applications. These technologies enable computers to learn from data and make intelligent decisions without being explicitly programmed.",
    "compression_model_name": "espresso_v1",
    "target_compression_ratio": 0.5,
    "source": "sdk:curl"
  }'

echo ""
