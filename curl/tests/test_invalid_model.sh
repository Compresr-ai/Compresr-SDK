#!/bin/bash
# Test error handling for invalid model names

set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

echo "========================================="
echo "Testing invalid model name error handling"
echo "========================================="
echo ""

CONTEXT="Test context for compression."

# Test 1: Invalid model name for question-agnostic endpoint
echo "1. Testing invalid model name (question-agnostic)..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "'"$CONTEXT"'",
    "compression_model_name": "invalid_model_12345",
    "target_compression_ratio": 0.5,
    "source": "sdk:curl"
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "422" ] || [ "$HTTP_CODE" = "400" ]; then
    if echo "$BODY" | jq -e '.detail' > /dev/null 2>&1; then
        echo "   ✓ Correctly rejected invalid model with error:"
        echo "   $(echo "$BODY" | jq -r '.detail')"
    else
        echo "   ✓ Correctly rejected invalid model (HTTP $HTTP_CODE)"
    fi
else
    echo "   ✗ Expected 422 or 400, got HTTP $HTTP_CODE"
    echo "   Response: $BODY"
    exit 1
fi
echo ""

# Test 2: Invalid model name for question-specific endpoint
echo "2. Testing invalid model name (question-specific)..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "'"$CONTEXT"'",
    "query": "test query",
    "compression_model_name": "nonexistent_model",
    "target_compression_ratio": 0.5,
    "source": "sdk:curl"
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "422" ] || [ "$HTTP_CODE" = "400" ]; then
    if echo "$BODY" | jq -e '.detail' > /dev/null 2>&1; then
        echo "   ✓ Correctly rejected invalid model with error:"
        echo "   $(echo "$BODY" | jq -r '.detail')"
    else
        echo "   ✓ Correctly rejected invalid model (HTTP $HTTP_CODE)"
    fi
else
    echo "   ✗ Expected 422 or 400, got HTTP $HTTP_CODE"
    echo "   Response: $BODY"
    exit 1
fi
echo ""

# Test 3: Wrong model type (using QS model on agnostic endpoint)
echo "3. Testing wrong model type (QS model on agnostic endpoint)..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "'"$CONTEXT"'",
    "compression_model_name": "latte_v1",
    "target_compression_ratio": 0.5,
    "source": "sdk:curl"
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "422" ] || [ "$HTTP_CODE" = "400" ]; then
    if echo "$BODY" | jq -e '.detail' > /dev/null 2>&1; then
        echo "   ✓ Correctly rejected wrong model type with error:"
        echo "   $(echo "$BODY" | jq -r '.detail')"
    else
        echo "   ✓ Correctly rejected wrong model type (HTTP $HTTP_CODE)"
    fi
else
    echo "   ✗ Expected rejection, got HTTP $HTTP_CODE"
    echo "   Response: $BODY"
    exit 1
fi
echo ""

echo "========================================="
echo "✓ All invalid model tests passed!"
echo "========================================="
exit 0
