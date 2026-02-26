#!/bin/bash
# Test Query-Specific Batch Compression
set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

echo "Testing QS batch compression endpoint..."

# Test batch QS compression with multiple contexts
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "inputs": [
      {
        "context": "Python is a high-level programming language. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python has a large standard library.",
        "query": "What is Python?"
      },
      {
        "context": "JavaScript is primarily used for web development. It runs in web browsers and on servers via Node.js. JavaScript is event-driven and supports asynchronous programming.",
        "query": "What is JavaScript?"
      },
      {
        "context": "Rust is a systems programming language focused on safety and performance. It prevents memory errors through its ownership system. Rust is used for operating systems and embedded systems.",
        "query": "What is Rust?"
      }
    ],
    "compression_model_name": "qs_gemfilter_v1",
    "target_compression_ratio": 0.5,
    "source": "sdk:curl"
  }')

# Check if response contains batch result fields
if echo "$RESPONSE" | jq -e '.data.results' > /dev/null 2>&1; then
    RESULT_COUNT=$(echo "$RESPONSE" | jq '.data.results | length')
    if [ "$RESULT_COUNT" = "3" ]; then
        echo "✓ QS batch compression test passed (3 results)"
        exit 0
    else
        echo "✗ QS batch compression test failed (expected 3 results, got $RESULT_COUNT)"
        echo "$RESPONSE"
        exit 1
    fi
else
    echo "✗ QS batch compression test failed"
    echo "$RESPONSE"
    exit 1
fi
