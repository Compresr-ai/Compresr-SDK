#!/bin/bash
# Compresr curl SDK - Query-Specific Compression

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

echo "Query-Specific Compression..."
echo "This compresses context while preserving information relevant to the query."
echo ""

curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "context": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand human language. Computer vision allows machines to interpret images. Reinforcement learning trains agents through rewards and penalties.",
    "query": "What is machine learning?",
    "compression_model_name": "qs_gemfilter_v1",
    "target_compression_ratio": 0.5,
    "source": "sdk:curl"
  }' | jq .
