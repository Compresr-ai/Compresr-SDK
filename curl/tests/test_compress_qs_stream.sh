#!/bin/bash
# Test Query-Specific Streaming Compression
set -e

# Load test configuration
# shellcheck source=./config.sh
. "$(dirname "$0")/config.sh"

echo "Testing QS streaming compression endpoint..."

# Test QS streaming compression
RESPONSE=$(curl -s -N -X POST "$BASE_URL/api/compress/question-specific/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $COMPRESR_API_KEY" \
  -d '{
    "context": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand human language. Computer vision allows machines to interpret images and videos. Reinforcement learning trains agents through trial and error with rewards.",
    "query": "What is machine learning?",
    "compression_model_name": "latte_v1",
    "target_compression_ratio": 0.5,
    "source": "sdk:curl"
  }')

# Check if response contains SSE data or stream content
if echo "$RESPONSE" | grep -q "data:" || echo "$RESPONSE" | grep -q "event:"; then
    echo "✓ QS streaming compression test passed"
    exit 0
else
    # Fallback: check if we got a complete response (some servers may not use SSE format)
    if echo "$RESPONSE" | grep -q "compressed"; then
        echo "✓ QS streaming compression test passed (non-SSE response)"
        exit 0
    else
        echo "✗ QS streaming compression test failed"
        echo "$RESPONSE"
        exit 1
    fi
fi
