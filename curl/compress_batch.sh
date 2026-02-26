#!/bin/bash
# Compresr curl SDK - Batch Compression

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

echo "Batch compressing 3 contexts..."
echo ""

curl -s -X POST "$BASE_URL/api/compress/question-agnostic/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "inputs": [
      {"context": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing."},
      {"context": "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience."},
      {"context": "Context compression helps reduce API costs by intelligently removing redundant information while preserving semantic meaning."}
    ],
    "compression_model_name": "espresso_v1",
    "target_compression_ratio": 0.5
  }' | jq .
