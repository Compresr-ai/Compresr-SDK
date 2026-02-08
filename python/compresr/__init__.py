"""
Compresr Python SDK

Compress context to reduce LLM costs.

Quick Start:
    from compresr import CompressionClient

    client = CompressionClient(api_key="cmp_...")
    response = client.compress(context="Your long context...")
    print(response.data.compressed_context)
"""

from .clients import CompressionClient

__version__ = "1.0.22"
__all__ = ["CompressionClient"]
