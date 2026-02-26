#!/bin/bash
# Test SaT model validation (binary filter - NO compression_ratio)

set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"
FAILED=0
PASSED=0

echo "========================================="
echo "Testing SaT Model Validation"
echo "========================================="
echo ""

CONTEXT="Machine learning is a subset of AI. Deep learning uses neural networks. NLP processes language."
QUERY="What is machine learning?"

# Test 1: SaT WITHOUT compression_ratio (should succeed)
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
    echo "   ✓ SaT without ratio succeeds (passed)"
    ((++PASSED))
else
    echo "   ✗ SaT without ratio should succeed (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 2: latte_v1 WITH compression_ratio (should succeed - latte supports compression_ratio)
echo "2. Testing latte_v1 WITH compression_ratio (should succeed)..."
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

# Test 3: coldbrew_v1 WITH compression_ratio (should fail with 422 - coldbrew doesn't support ratio)
echo "3. Testing coldbrew_v1 WITH compression_ratio (should fail)..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"coldbrew_v1\",
    \"target_compression_ratio\": 0.5,
    \"source\": \"sdk:curl\"
  }")

HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "422" ]; then
    echo "   ✓ coldbrew with ratio returns 422 (passed)"
    ((++PASSED))
else
    echo "   ✗ coldbrew with ratio should return 422 (got $HTTP_CODE)"
    echo "   Response: $BODY"
    ((++FAILED))
fi
echo ""

# Test 4: coldbrew_v1 WITHOUT compression_ratio (should succeed)
echo "4. Testing coldbrew_v1 WITHOUT compression_ratio (should succeed)..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"coldbrew_v1\",
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null 2>&1; then
    echo "   ✓ coldbrew without ratio succeeds (passed)"
    ((++PASSED))
else
    echo "   ✗ coldbrew without ratio should succeed (failed)"
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

echo "✓ All SaT validation tests passed!"
exit 0
