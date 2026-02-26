#!/bin/bash
# Test all 7 API models

set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"
FAILED=0
PASSED=0

echo "========================================="
echo "Testing all 7 API models"
echo "========================================="
echo ""

# Test context
CONTEXT="Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand human language. Computer vision allows machines to interpret images."
QUERY="What is machine learning?"

# Test 1: agnostic_compressor_1 (query agnostic)
echo "1. Testing agnostic_compressor_1..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-agnostic/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"compression_model_name\": \"agnostic_compressor_1\",
    \"target_compression_ratio\": 0.5,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_context' > /dev/null 2>&1; then
    echo "   ✓ agnostic_compressor_1 passed"
    ((++PASSED))
else
    echo "   ✗ agnostic_compressor_1 failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 2: qs_gemfilter_v1 (query specific)
echo "2. Testing qs_gemfilter_v1..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"qs_gemfilter_v1\",
    \"target_compression_ratio\": 0.5,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_context' > /dev/null 2>&1; then
    echo "   ✓ qs_gemfilter_v1 passed"
    ((++PASSED))
else
    echo "   ✗ qs_gemfilter_v1 failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 3: qs_sat_v1 (query specific)
echo "3. Testing qs_sat_v1..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/question-specific/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"context\": \"$CONTEXT\",
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"qs_sat_v1\",
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_context' > /dev/null 2>&1; then
    echo "   ✓ qs_sat_v1 passed"
    ((++PASSED))
else
    echo "   ✗ qs_sat_v1 failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 4: agentic_tool_output_gemfilter
echo "4. Testing agentic_tool_output_gemfilter..."
TOOL_OUTPUT="Tool execution result: The database query returned 42 rows with the following columns: id, name, email, created_at. The query took 0.023 seconds to execute. Memory usage was 2.3 MB. All rows were successfully fetched without errors."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/tool-output/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"tool_output\": \"$TOOL_OUTPUT\",
    \"query\": \"$QUERY\",
    \"tool_name\": \"database_query\",
    \"compression_model_name\": \"agentic_tool_output_gemfilter\",
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_output' > /dev/null 2>&1; then
    echo "   ✓ agentic_tool_output_gemfilter passed"
    ((++PASSED))
else
    echo "   ✗ agentic_tool_output_gemfilter failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 5: agentic_tool_discovery_sat
echo "5. Testing agentic_tool_discovery_sat..."
TOOLS='[
  {"name": "search_db", "description": "Search the database for records"},
  {"name": "send_email", "description": "Send an email to a user"},
  {"name": "calculate_sum", "description": "Calculate the sum of numbers"}
]'
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/tool-discovery/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"tools\": $TOOLS,
    \"query\": \"$QUERY\",
    \"compression_model_name\": \"agentic_tool_discovery_sat\",
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.relevant_tools' > /dev/null 2>&1; then
    echo "   ✓ agentic_tool_discovery_sat passed"
    ((++PASSED))
else
    echo "   ✗ agentic_tool_discovery_sat failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 6: agentic_tool_output_lingua (NO query)
echo "6. Testing agentic_tool_output_lingua..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/tool-output/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"tool_output\": \"$TOOL_OUTPUT\",
    \"tool_name\": \"database_query\",
    \"compression_model_name\": \"agentic_tool_output_lingua\",
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.compressed_output' > /dev/null 2>&1; then
    echo "   ✓ agentic_tool_output_lingua passed"
    ((++PASSED))
else
    echo "   ✗ agentic_tool_output_lingua failed"
    echo "   Response: $RESPONSE"
    ((++FAILED))
fi
echo ""

# Test 7: agentic_history_lingua
echo "7. Testing agentic_history_lingua..."
MESSAGES='[
  {"role": "user", "content": "Hello, can you help me?"},
  {"role": "assistant", "content": "Of course! I would be happy to help you with whatever you need."},
  {"role": "user", "content": "I need information about machine learning."}
]'
RESPONSE=$(curl -s -X POST "$BASE_URL/api/compress/history/" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d "{
    \"messages\": $MESSAGES,
    \"compression_model_name\": \"agentic_history_lingua\",
    \"target_compression_ratio\": 0.5,
    \"source\": \"sdk:curl\"
  }")

if echo "$RESPONSE" | jq -e '.data.summary' > /dev/null 2>&1; then
    echo "   ✓ agentic_history_lingua passed"
    ((++PASSED))
else
    echo "   ✗ agentic_history_lingua failed"
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
