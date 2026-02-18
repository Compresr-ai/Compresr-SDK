#!/usr/bin/env python
"""
Debug script to test QS compression and see what's being sent.
"""
import os
import json
from compresr import QSCompressionClient
from compresr.exceptions import ServerError

# Load API key
api_key = os.getenv("COMPRESR_API_KEY")
if not api_key:
    print("❌ COMPRESR_API_KEY not set")
    exit(1)

print(f"🔑 API Key: {api_key[:15]}...")

# Create client
client = QSCompressionClient(api_key=api_key)
print(f"✅ QSCompressionClient created")

# Test simple compression
context = "Machine learning is AI. Python is a language. JavaScript is for web."
question = "What is machine learning?"
model = "QS_CMPRSR_V1"

print(f"\n📝 Testing QS compression:")
print(f"   Context: {context}")
print(f"   Question: {question}")
print(f"   Model: {model}")

try:
    # Build the request to see what's being sent
    req = client._build_request(context, model, question)
    print(f"\n📤 Request payload:")
    print(json.dumps(req.model_dump(exclude_none=True), indent=2))
    
    # Try to compress
    print(f"\n🔄 Sending request...")
    response = client.compress(
        context=context,
        question=question,
        compression_model_name=model
    )
    
    print(f"\n✅ SUCCESS!")
    print(f"   Original tokens: {response.data.original_tokens}")
    print(f"   Compressed tokens: {response.data.compressed_tokens}")
    print(f"   Compressed: {response.data.compressed_context}")
    
except ServerError as e:
    print(f"\n❌ ServerError: {e}")
    if hasattr(e, 'response_data') and e.response_data:
        print(f"   Response data: {json.dumps(e.response_data, indent=2)}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
