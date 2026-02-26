#!/bin/bash
# Test list context support (context can be str or List[str])

set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"
FAILED=0
PASSED=0

echo "========================================="
echo "Testing List Context Support"
echo "========================================="
echo ""

# Test 1: Agnostic with string context
echo "1. Testing agnostic compression with string context..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Machine learning is a powerful tool for data analysis.",
    "compression_model_name": "espresso_v1",
    "source": "sdk:curl"
  }')

if echo "$RESPONSE" | jq -e '.data.compressed_context | type == "string"' > /dev/null 2>&1; then
    echo "   ✓ String context returns string (passed)"
    ((++PASSED))
else
    echo "   ✗ String context should return string (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 2: Agnostic with list context
echo "2. Testing agnostic compression with list context..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": ["Machine learning enables data-driven decisions.", "Deep learning uses neural networks.", "NLP processes human language."],
    "compression_model_name": "espresso_v1",
    "source": "sdk:curl"
  }')

if echo "$RESPONSE" | jq -e '.data.compressed_context | type == "array"' > /dev/null 2>&1; then
    echo "   ✓ List context returns list (passed)"
    ((++PASSED))
else
    echo "   ✗ List context should return list (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 3: QS with string context
echo "3. Testing QS compression with string context..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Python was created by Guido van Rossum. Java was created by James Gosling.",
    "query": "Who created Python?",
    "compression_model_name": "coldbrew_v1",
    "source": "sdk:curl"
  }')

if echo "$RESPONSE" | jq -e '.data.compressed_context | type == "string"' > /dev/null 2>&1; then
    echo "   ✓ QS string context returns string (passed)"
    ((++PASSED))
else
    echo "   ✗ QS string context should return string (failed)"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 4: QS with list context
echo "4. Testing QS compression with list context..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": ["Python was created by Guido van Rossum in 1991.", "JavaScript was created by Brendan Eich.", "Java was developed by James Gosling."],
    "query": "Who created Python?",
    "compression_model_name": "coldbrew_v1",
    "source": "sdk:curl"
  }')

if echo "$RESPONSE" | jq -e '.data.compressed_context | type == "array"' > /dev/null 2>&1; then
    echo "   ✓ QS list context returns list (passed)"
    ((++PASSED))
else
    echo "   ✗ QS list context should return list (failed)"
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

echo "✓ All list context tests passed!"
exit 0
