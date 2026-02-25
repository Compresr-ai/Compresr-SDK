"""
Compresr Python SDK

Compress context to reduce LLM costs.

Two client types:
1. CompressionClient - Token-level compression, customizable compression_ratio
2. FilterClient - Coarse-grained chunk selection, keeps/drops entire chunks

Quick Start - Compression:
    from compresr import CompressionClient, MODELS

    client = CompressionClient(api_key="cmp_...")

    # Agnostic compression (espresso_v1, default)
    response = client.compress(context="Your long context...")
    print(response.data.compressed_context)  # str

    # Query-specific compression (latte_v1)
    response = client.compress(
        context="Your long context...",
        query="What is the main conclusion?",
        compression_model_name="latte_v1",
    )

Quick Start - Filtering:
    from compresr import FilterClient, MODELS

    client = FilterClient(api_key="cmp_...")

    response = client.filter(
        chunks=["Chunk 1...", "Chunk 2...", "Chunk 3..."],
        query="What is relevant?",
    )
    print(response.data.compressed_context)  # List[str]
"""

from .clients import CompressionClient, FilterClient
from .config import MODELS

__version__ = "2.0.0"
__all__ = [
    "CompressionClient",
    "FilterClient",
    "MODELS",
]
