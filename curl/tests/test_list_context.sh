#!/bin/bash
# Test single vs batch endpoint separation
# - Single endpoints: context: str only
# - Batch endpoints: inputs: List[{context, query}] format

set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"
FAILED=0
PASSED=0

echo "========================================="
echo "Testing Single vs Batch Endpoint Separation"
echo "========================================="
echo ""

# Test 1: QS single endpoint with string context
echo "1. Testing QS single endpoint (string context + query)..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Python was created by Guido van Rossum. Java was created by James Gosling.",
    "query": "Who created Python?",
    "compression_model_name": "latte_v2",
    "source": "sdk:curl"
  }')

if echo "$RESPONSE" | jq -e '.data.compressed_context | type == "string"' > /dev/null 2>&1; then
    echo "   ✓ Single QS returns string (passed)"
    ((++PASSED))
else
    echo "   ✗ Single QS should return string (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 2: QS batch endpoint
echo "2. Testing QS batch endpoint..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "inputs": [
      {"context": "Python was created by Guido van Rossum in 1991.", "query": "Who created Python?"},
      {"context": "JavaScript was created by Brendan Eich.", "query": "Who created JavaScript?"},
      {"context": "Java was developed by James Gosling.", "query": "Who created Java?"}
    ],
    "compression_model_name": "latte_v2",
    "source": "sdk:curl"
  }')

if echo "$RESPONSE" | jq -e '.data.results | length == 3' > /dev/null 2>&1; then
    echo "   ✓ Batch QS returns 3 results (passed)"
    ((++PASSED))
else
    echo "   ✗ Batch QS should return 3 results (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

echo "========================================="
echo "Results: $PASSED passed, $FAILED failed"
echo "========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi

echo "✓ All single/batch separation tests passed!"
exit 0
