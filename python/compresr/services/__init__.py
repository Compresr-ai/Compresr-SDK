"""
Compresr Services

Usage:
    from compresr import CompressionClient

    # Token-level compression
    client = CompressionClient(api_key="cmp_...")
    result = client.compress(context="Your long context...")

    # Query-specific compression
    result = client.compress(
        context="Your context...",
        query="What is...?",
        compression_model_name="latte_v1",
    )

    # Query-specific compression with coarse (faster)
    result = client.compress(
        context="Your context...",
        query="What is...?",
        compression_model_name="latte_v1",
        coarse=True,  # paragraph-level (faster)
    )
"""

from .compression import CompressionClient

__all__ = [
    "CompressionClient",
]
