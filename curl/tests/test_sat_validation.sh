#!/bin/bash
# Test model validation and coarse parameter

set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"
FAILED=0
PASSED=0

echo "========================================="
echo "Testing Model Validation & Coarse Param"
echo "========================================="
echo ""

CONTEXT="Machine learning is a subset of AI. Deep learning uses neural networks. NLP processes language."
QUERY="What is machine learning?"

# Test 1: latte_v1 WITHOUT compression_ratio (should succeed)
echo "1. Testing latte_v1 WITHOUT compression_ratio..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"latte_v1\",
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null 2>&1; then
    echo "   ✓ latte_v1 without ratio succeeds (passed)"
    ((++PASSED))
else
    echo "   ✗ latte_v1 without ratio should succeed (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 2: latte_v1 WITH compression_ratio (should succeed)
echo "2. Testing latte_v1 WITH compression_ratio..."
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

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null 2>&1; then
    echo "   ✓ latte_v1 with ratio succeeds (passed)"
    ((++PASSED))
else
    echo "   ✗ latte_v1 with ratio should succeed (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 3: latte_v1 WITH coarse=true (should succeed)
echo "3. Testing latte_v1 WITH coarse=true (paragraph-level)..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"latte_v1\",
    \"coarse\": true,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null 2>&1; then
    echo "   ✓ latte_v1 with coarse=true succeeds (passed)"
    ((++PASSED))
else
    echo "   ✗ latte_v1 with coarse=true should succeed (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 4: latte_v1 WITH coarse=false (should succeed)
echo "4. Testing latte_v1 WITH coarse=false (token-level)..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"latte_v1\",
    \"coarse\": false,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null 2>&1; then
    echo "   ✓ latte_v1 with coarse=false succeeds (passed)"
    ((++PASSED))
else
    echo "   ✗ latte_v1 with coarse=false should succeed (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 5: espresso_v1 (agnostic, no query)
echo "5. Testing espresso_v1 (agnostic, no query)..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"compression_model_name\": \"espresso_v1\",
    \"target_compression_ratio\": 0.5,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null 2>&1; then
    echo "   ✓ espresso_v1 succeeds (passed)"
    ((++PASSED))
else
    echo "   ✗ espresso_v1 should succeed (failed)"
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

echo "✓ All validation tests passed!"
exit 0
