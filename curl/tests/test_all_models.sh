#!/bin/bash
# Test all supported compression models (espresso_v1, latte_v1, coldbrew_v1)

set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"
FAILED=0
PASSED=0

echo "========================================="
echo "Testing all supported compression models"
echo "========================================="
echo ""

# Test context
CONTEXT="Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand human language. Computer vision allows machines to interpret images."
QUERY="What is machine learning?"

# Test 1: espresso_v1 (question agnostic)
echo "1. Testing espresso_v1..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"compression_model_name\": \"espresso_v1\",
    \"target_compression_ratio\": 0.5,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_context' > /dev/null 2>&1; then
    echo "   ✓ espresso_v1 passed"
    ((++PASSED))
else
    echo "   ✗ espresso_v1 failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 2: latte_v1 (question specific)
echo "2. Testing latte_v1..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"latte_v1\",
    \"target_compression_ratio\": 0.5,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_context' > /dev/null 2>&1; then
    echo "   ✓ latte_v1 passed"
    ((++PASSED))
else
    echo "   ✗ latte_v1 failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 3: coldbrew_v1 (filter model - uses context parameter)
echo "3. Testing coldbrew_v1..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"coldbrew_v1\",
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_context' > /dev/null 2>&1; then
    echo "   ✓ coldbrew_v1 passed"
    ((++PASSED))
else
    echo "   ✗ coldbrew_v1 failed"
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

echo "✓ All model tests passed!"
exit 0
