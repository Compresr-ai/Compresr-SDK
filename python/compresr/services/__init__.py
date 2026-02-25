"""
Compresr Services

Usage:
    from compresr import CompressionClient, FilterClient

    # Token-level compression
    client = CompressionClient(api_key="cmp_...")
    result = client.compress(context="Your long context...")

    # Query-specific compression
    result = client.compress(
        context="Your context...",
        query="What is...?",
        compression_model_name="latte_v1",
    )

    # Chunk-level filtering
    filter_client = FilterClient(api_key="cmp_...")
    result = filter_client.filter(
        chunks=["Chunk 1...", "Chunk 2..."],
        query="What is relevant?",
    )
"""

from .compression import CompressionClient
from .filter import FilterClient

__all__ = [
    "CompressionClient",
    "FilterClient",
]
